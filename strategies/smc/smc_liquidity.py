"""
SMC Liquidity Mapping & 3R Veto Engine (smc_liquidity.py)
=========================================================
Identifies institutional liquidity pools and validates setups against the strict 3R Veto Rule:
  - Internal Liquidity: Opposing FVGs, minor internal swing pivots (TP1: 40%)
  - Primary Structural Target: Opposing HTF FVG or impulse origin (TP2: 40%)
  - External Liquidity: PDH, PDL, Equal Highs (EQH), Equal Lows (EQL) (TP3: 20% Runner)
  - 3R Minimum Veto Rule: Rejects any setup whose logical structural target offers < 3.0R
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd

from .smc_structure import SwingPoint, SwingType
from .smc_fvg import FairValueGap


class LiquidityType(str, Enum):
    INTERNAL = "INTERNAL"    # TP1: Minor pivot or nearest opposing FVG
    PRIMARY = "PRIMARY"      # TP2: Structural swing target / major opposing FVG
    EXTERNAL = "EXTERNAL"    # TP3: PDH, PDL, EQH, EQL runner target


@dataclass
class LiquidityPool:
    """Represents a targeted liquidity pool level."""
    price: float
    pool_type: LiquidityType
    name: str
    description: str
    timestamp: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.price,
            "pool_type": self.pool_type.value,
            "name": self.name,
            "description": self.description,
            "timestamp": str(self.timestamp) if self.timestamp else None,
        }


def detect_equal_highs_lows(
    pivots: List[SwingPoint],
    tolerance_pct: float = 0.0015  # 0.15% tolerance
) -> Tuple[List[LiquidityPool], List[LiquidityPool]]:
    """
    Detects Equal Highs (EQH) and Equal Lows (EQL) where retail stops accumulate.
    """
    eqh_pools: List[LiquidityPool] = []
    eql_pools: List[LiquidityPool] = []

    high_pivots = [p for p in pivots if p.swing_type == SwingType.HIGH and not p.swept]
    low_pivots = [p for p in pivots if p.swing_type == SwingType.LOW and not p.swept]

    # Equal Highs (Double / Triple tops)
    for i in range(len(high_pivots)):
        for j in range(i + 1, len(high_pivots)):
            p1 = high_pivots[i]
            p2 = high_pivots[j]
            diff = abs(p1.price - p2.price) / max(p1.price, p2.price)
            if diff <= tolerance_pct:
                avg_price = (p1.price + p2.price) / 2.0
                eqh_pools.append(
                    LiquidityPool(
                        price=avg_price,
                        pool_type=LiquidityType.EXTERNAL,
                        name="EQH",
                        description=f"Equal Highs at ${avg_price:.2f} (bars {p1.bar_idx} & {p2.bar_idx})",
                        timestamp=p2.timestamp,
                    )
                )

    # Equal Lows (Double / Triple bottoms)
    for i in range(len(low_pivots)):
        for j in range(i + 1, len(low_pivots)):
            p1 = low_pivots[i]
            p2 = low_pivots[j]
            diff = abs(p1.price - p2.price) / max(p1.price, p2.price)
            if diff <= tolerance_pct:
                avg_price = (p1.price + p2.price) / 2.0
                eql_pools.append(
                    LiquidityPool(
                        price=avg_price,
                        pool_type=LiquidityType.EXTERNAL,
                        name="EQL",
                        description=f"Equal Lows at ${avg_price:.2f} (bars {p1.bar_idx} & {p2.bar_idx})",
                        timestamp=p2.timestamp,
                    )
                )

    return eqh_pools, eql_pools


def extract_pdh_pdl(df_5m: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts Previous Day High (PDH) and Previous Day Low (PDL) from 5-minute RTH data.
    """
    if df_5m.empty:
        return None, None

    # Resample daily
    daily = df_5m.resample("1D").agg({
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    if len(daily) >= 2:
        prev_day = daily.iloc[-2]
        return float(prev_day["high"]), float(prev_day["low"])
    elif len(daily) == 1:
        prev_day = daily.iloc[0]
        return float(prev_day["high"]), float(prev_day["low"])
    return None, None


def find_liquidity_pools(
    current_price: float,
    side: str,  # "BUY" or "SELL"
    pivots: List[SwingPoint],
    opposing_fvgs: List[FairValueGap],
    pdh: Optional[float] = None,
    pdl: Optional[float] = None
) -> Tuple[Optional[LiquidityPool], Optional[LiquidityPool], Optional[LiquidityPool]]:
    """
    Pre-plans the 3-Tier Take Profit Liquidity Pools for a prospective trade:
      - TP1: Internal Liquidity (First opposing FVG or closest swing pivot)
      - TP2: Primary Structural Target (Major swing extreme or opposing HTF FVG)
      - TP3: External Liquidity Pool (PDH/PDL or EQH/EQL runner target)
    """
    tp1: Optional[LiquidityPool] = None
    tp2: Optional[LiquidityPool] = None
    tp3: Optional[LiquidityPool] = None

    eqh, eql = detect_equal_highs_lows(pivots)

    if side.upper() == "BUY":
        # Target higher liquidity pools
        # 1. TP1: Nearest opposing bearish FVG or nearest swing high above current price
        candidate_fvgs = [f for f in opposing_fvgs if f.bottom > current_price and not f.mitigated]
        candidate_pivots = [p for p in pivots if p.swing_type == SwingType.HIGH and p.price > current_price]

        if candidate_fvgs:
            nearest_fvg = min(candidate_fvgs, key=lambda f: f.bottom)
            tp1 = LiquidityPool(
                price=nearest_fvg.bottom,
                pool_type=LiquidityType.INTERNAL,
                name="Opposing FVG (TP1)",
                description=f"Opposing Bearish FVG bottom at ${nearest_fvg.bottom:.2f}",
                timestamp=nearest_fvg.timestamp
            )
        elif candidate_pivots:
            nearest_pivot = min(candidate_pivots, key=lambda p: p.price)
            tp1 = LiquidityPool(
                price=nearest_pivot.price,
                pool_type=LiquidityType.INTERNAL,
                name="Internal Swing High (TP1)",
                description=f"Nearest Swing High at ${nearest_pivot.price:.2f}",
                timestamp=nearest_pivot.timestamp
            )

        # 2. TP2: Major structural swing high or secondary opposing FVG
        if candidate_pivots:
            highest_pivot = max(candidate_pivots, key=lambda p: p.price)
            tp2 = LiquidityPool(
                price=highest_pivot.price,
                pool_type=LiquidityType.PRIMARY,
                name="Structural Swing High (TP2)",
                description=f"Major Swing High target at ${highest_pivot.price:.2f}",
                timestamp=highest_pivot.timestamp
            )

        # 3. TP3: External Liquidity (PDH or EQH)
        external_targets: List[Tuple[float, str]] = []
        if pdh and pdh > current_price:
            external_targets.append((pdh, f"Previous Day High (PDH: ${pdh:.2f})"))
        for pool in eqh:
            if pool.price > current_price:
                external_targets.append((pool.price, pool.description))

        if external_targets:
            furthest_ext = max(external_targets, key=lambda x: x[0])
            tp3 = LiquidityPool(
                price=furthest_ext[0],
                pool_type=LiquidityType.EXTERNAL,
                name="External Liquidity Runner (TP3)",
                description=furthest_ext[1],
            )
        elif tp2:
            # If no further external pool, extend beyond TP2
            tp3 = LiquidityPool(
                price=tp2.price * 1.01,
                pool_type=LiquidityType.EXTERNAL,
                name="External Extension (TP3)",
                description=f"1% Extension above ${tp2.price:.2f}",
            )

    else:
        # Target lower liquidity pools (SELL / SHORT)
        candidate_fvgs = [f for f in opposing_fvgs if f.top < current_price and not f.mitigated]
        candidate_pivots = [p for p in pivots if p.swing_type == SwingType.LOW and p.price < current_price]

        # 1. TP1: Nearest opposing bullish FVG or nearest swing low below current price
        if candidate_fvgs:
            nearest_fvg = max(candidate_fvgs, key=lambda f: f.top)
            tp1 = LiquidityPool(
                price=nearest_fvg.top,
                pool_type=LiquidityType.INTERNAL,
                name="Opposing FVG (TP1)",
                description=f"Opposing Bullish FVG top at ${nearest_fvg.top:.2f}",
                timestamp=nearest_fvg.timestamp
            )
        elif candidate_pivots:
            nearest_pivot = max(candidate_pivots, key=lambda p: p.price)
            tp1 = LiquidityPool(
                price=nearest_pivot.price,
                pool_type=LiquidityType.INTERNAL,
                name="Internal Swing Low (TP1)",
                description=f"Nearest Swing Low at ${nearest_pivot.price:.2f}",
                timestamp=nearest_pivot.timestamp
            )

        # 2. TP2: Major structural swing low
        if candidate_pivots:
            lowest_pivot = min(candidate_pivots, key=lambda p: p.price)
            tp2 = LiquidityPool(
                price=lowest_pivot.price,
                pool_type=LiquidityType.PRIMARY,
                name="Structural Swing Low (TP2)",
                description=f"Major Swing Low target at ${lowest_pivot.price:.2f}",
                timestamp=lowest_pivot.timestamp
            )

        # 3. TP3: External Liquidity (PDL or EQL)
        external_targets: List[Tuple[float, str]] = []
        if pdl and pdl < current_price:
            external_targets.append((pdl, f"Previous Day Low (PDL: ${pdl:.2f})"))
        for pool in eql:
            if pool.price < current_price:
                external_targets.append((pool.price, pool.description))

        if external_targets:
            lowest_ext = min(external_targets, key=lambda x: x[0])
            tp3 = LiquidityPool(
                price=lowest_ext[0],
                pool_type=LiquidityType.EXTERNAL,
                name="External Liquidity Runner (TP3)",
                description=lowest_ext[1],
            )
        elif tp2:
            tp3 = LiquidityPool(
                price=tp2.price * 0.99,
                pool_type=LiquidityType.EXTERNAL,
                name="External Extension (TP3)",
                description=f"1% Extension below ${tp2.price:.2f}",
            )

    return tp1, tp2, tp3


def evaluate_3r_veto(
    entry_price: float,
    stop_loss: float,
    primary_target: float,
    side: str,
    min_rr: float = 3.0
) -> Tuple[bool, float, str]:
    """
    Enforces Part 4, Rule 3: The 3R Minimum & The Veto Rule:
      - 1R = distance from Entry to Stop Loss.
      - Primary Target = must yield at least 3R.
      - If next logical liquidity target yields < 3R, skip the trade entirely.
      - Never arbitrarily widen target or tighten stop.
    """
    if side.upper() == "BUY":
        risk = entry_price - stop_loss
        reward = primary_target - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - primary_target

    if risk <= 0:
        return False, 0.0, f"Invalid risk distance: risk={risk:.4f} <= 0"

    rr_ratio = reward / risk

    if rr_ratio < min_rr:
        return False, float(rr_ratio), (
            f"VETO RULE TRIGGERED: Logical primary target at ${primary_target:.2f} "
            f"only offers {rr_ratio:.2f}R (< {min_rr:.1f}R minimum). Trade discarded."
        )

    return True, float(rr_ratio), f"APPROVED: Primary target offers {rr_ratio:.2f}R (>= {min_rr:.1f}R)."
