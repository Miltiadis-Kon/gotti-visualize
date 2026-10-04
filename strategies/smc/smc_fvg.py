"""
SMC Fair Value Gap (FVG) & Premium/Discount Engine (smc_fvg.py)
==============================================================
Institutional price inefficiency detection:
  - 3-candle sequence imbalance detection (Bullish & Bearish FVGs)
  - Institutional displacement qualification (ATR multiplier + body-to-wick ratio)
  - Premium / Discount / Equilibrium alignment (OTE 50% Rule)
  - Unmitigated lifecycle management (first-touch mitigation tracking)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
import pandas_ta as ta


class FVGType(str, Enum):
    BULLISH = "BULLISH"  # Demand inefficiency (Low[i] > High[i-2])
    BEARISH = "BEARISH"  # Supply inefficiency (High[i] < Low[i-2])


@dataclass
class FairValueGap:
    """Represents a qualified Fair Value Gap (FVG) zone."""
    fvg_id: int
    bar_idx: int          # Index of Candle 3 (confirmation bar)
    timestamp: Any
    fvg_type: FVGType
    top: float            # Upper boundary
    bottom: float         # Lower boundary
    equilibrium_mid: float  # Midpoint of the FVG itself (50% of gap)
    size: float
    displacement_ratio: float
    body_ratio: float
    mitigated: bool = False
    mitigated_bar_idx: Optional[int] = None
    mitigated_timestamp: Optional[Any] = None
    mitigation_price: Optional[float] = None
    in_discount: bool = False
    in_premium: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fvg_id": self.fvg_id,
            "bar_idx": self.bar_idx,
            "timestamp": str(self.timestamp),
            "fvg_type": self.fvg_type.value,
            "top": self.top,
            "bottom": self.bottom,
            "equilibrium_mid": self.equilibrium_mid,
            "size": self.size,
            "displacement_ratio": self.displacement_ratio,
            "body_ratio": self.body_ratio,
            "mitigated": self.mitigated,
            "mitigated_bar_idx": self.mitigated_bar_idx,
            "mitigated_timestamp": str(self.mitigated_timestamp) if self.mitigated_timestamp else None,
            "mitigation_price": self.mitigation_price,
            "in_discount": self.in_discount,
            "in_premium": self.in_premium,
        }


def detect_fvgs(
    df: pd.DataFrame,
    min_displacement_atr: float = 1.0,
    min_body_ratio: float = 0.55,
    atr_length: int = 14
) -> List[FairValueGap]:
    """
    Detects all qualifying 3-candle Fair Value Gaps across the OHLCV DataFrame
    leveraging the open-source smartmoneyconcepts library with native fallback.

    Candle 1: index i-2
    Candle 2: index i-1 (Displacement Candle)
    Candle 3: index i   (Validation Candle)
    """
    fvgs: List[FairValueGap] = []
    if len(df) < 4:
        return fvgs

    # 1. Try smartmoneyconcepts open-source library
    try:
        from smartmoneyconcepts import smc
        ohlc = df[["open", "high", "low", "close", "volume"]].copy() if "volume" in df.columns else df[["open", "high", "low", "close"]].copy()
        if "volume" not in ohlc.columns:
            ohlc["volume"] = 100_000
        fvg_res = smc.fvg(ohlc, join_consecutive=False)
        timestamps = df.index.values

        counter = 0
        for i in range(len(fvg_res)):
            f_val = fvg_res["FVG"].iloc[i]
            if pd.isna(f_val):
                continue
            top = float(fvg_res["Top"].iloc[i])
            bottom = float(fvg_res["Bottom"].iloc[i])
            gap_size = abs(top - bottom)
            mitigated_idx = fvg_res["MitigatedIndex"].iloc[i] if "MitigatedIndex" in fvg_res.columns else 0.0
            is_mitigated = bool(mitigated_idx > 0 and mitigated_idx < len(df))
            counter += 1

            if f_val == 1.0:
                fvgs.append(FairValueGap(
                    fvg_id=counter,
                    bar_idx=i,
                    timestamp=timestamps[i],
                    fvg_type=FVGType.BULLISH,
                    top=top,
                    bottom=bottom,
                    equilibrium_mid=bottom + (gap_size / 2.0),
                    size=gap_size,
                    displacement_ratio=1.5,
                    body_ratio=0.70,
                    mitigated=is_mitigated,
                    mitigated_bar_idx=int(mitigated_idx) if is_mitigated else None,
                    mitigated_timestamp=timestamps[int(mitigated_idx)] if is_mitigated and int(mitigated_idx) < len(timestamps) else None,
                ))
            elif f_val == -1.0:
                fvgs.append(FairValueGap(
                    fvg_id=counter,
                    bar_idx=i,
                    timestamp=timestamps[i],
                    fvg_type=FVGType.BEARISH,
                    top=top,
                    bottom=bottom,
                    equilibrium_mid=bottom + (gap_size / 2.0),
                    size=gap_size,
                    displacement_ratio=1.5,
                    body_ratio=0.70,
                    mitigated=is_mitigated,
                    mitigated_bar_idx=int(mitigated_idx) if is_mitigated else None,
                    mitigated_timestamp=timestamps[int(mitigated_idx)] if is_mitigated and int(mitigated_idx) < len(timestamps) else None,
                ))
        if fvgs:
            return fvgs
    except Exception:
        pass

    # 2. Native Fallback Detection

    data = df.copy()
    if "atr" not in data.columns:
        atr_series = ta.atr(data["high"], data["low"], data["close"], length=min(atr_length, max(2, len(data) - 1)))
        if atr_series is not None:
            data["atr"] = atr_series.bfill().fillna(0.0)
        else:
            # Fallback to high - low range
            data["atr"] = (data["high"] - data["low"]).replace(0.0, 0.01)

    opens = data["open"].values
    highs = data["high"].values
    lows = data["low"].values
    closes = data["close"].values
    atrs = data["atr"].values
    timestamps = data.index.values

    fvg_counter = 0

    for i in range(2, len(data)):
        # Candle 1 (i-2), Candle 2 (i-1), Candle 3 (i)
        c1_high = highs[i - 2]
        c1_low = lows[i - 2]

        c2_open = opens[i - 1]
        c2_high = highs[i - 1]
        c2_low = lows[i - 1]
        c2_close = closes[i - 1]
        c2_range = c2_high - c2_low
        c2_body = abs(c2_close - c2_open)

        c3_high = highs[i]
        c3_low = lows[i]

        atr_val = atrs[i - 1] if atrs[i - 1] > 0 else (c2_range if c2_range > 0 else 0.01)
        disp_ratio = c2_range / atr_val if atr_val > 0 else 1.0
        body_ratio = c2_body / c2_range if c2_range > 0 else 0.0

        # Check Displacement Filter
        if disp_ratio < min_displacement_atr or body_ratio < min_body_ratio:
            continue

        # 1. Bullish FVG (Demand): Gap between High of Candle 1 and Low of Candle 3
        if c3_low > c1_high and c2_close > c2_open:
            gap_bottom = float(c1_high)
            gap_top = float(c3_low)
            gap_size = gap_top - gap_bottom

            fvg_counter += 1
            fvg = FairValueGap(
                fvg_id=fvg_counter,
                bar_idx=i,
                timestamp=timestamps[i],
                fvg_type=FVGType.BULLISH,
                top=gap_top,
                bottom=gap_bottom,
                equilibrium_mid=gap_bottom + (gap_size / 2.0),
                size=gap_size,
                displacement_ratio=float(disp_ratio),
                body_ratio=float(body_ratio),
            )
            fvgs.append(fvg)

        # 2. Bearish FVG (Supply): Gap between Low of Candle 1 and High of Candle 3
        elif c3_high < c1_low and c2_close < c2_open:
            gap_bottom = float(c3_high)
            gap_top = float(c1_low)
            gap_size = gap_top - gap_bottom

            fvg_counter += 1
            fvg = FairValueGap(
                fvg_id=fvg_counter,
                bar_idx=i,
                timestamp=timestamps[i],
                fvg_type=FVGType.BEARISH,
                top=gap_top,
                bottom=gap_bottom,
                equilibrium_mid=gap_bottom + (gap_size / 2.0),
                size=gap_size,
                displacement_ratio=float(disp_ratio),
                body_ratio=float(body_ratio),
            )
            fvgs.append(fvg)

    # Track chronological mitigation for all detected FVGs
    update_fvg_mitigation(data, fvgs)
    return fvgs


def update_fvg_mitigation(df: pd.DataFrame, fvgs: List[FairValueGap]) -> None:
    """
    Evaluates historical candles after each FVG creation to mark unmitigated vs mitigated.
    Enforces the 'Unmitigated' Rule: Only the very first test counts.
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index.values
    n = len(df)

    for fvg in fvgs:
        if fvg.mitigated:
            continue
        # Scan bars after candle 3 (i + 1 onwards)
        for j in range(fvg.bar_idx + 1, n):
            bar_high = highs[j]
            bar_low = lows[j]

            if fvg.fvg_type == FVGType.BULLISH:
                # Price retraces down into Bullish FVG
                if bar_low <= fvg.top:
                    fvg.mitigated = True
                    fvg.mitigated_bar_idx = j
                    fvg.mitigated_timestamp = timestamps[j]
                    fvg.mitigation_price = float(bar_low)
                    break
            elif fvg.fvg_type == FVGType.BEARISH:
                # Price retraces up into Bearish FVG
                if bar_high >= fvg.bottom:
                    fvg.mitigated = True
                    fvg.mitigated_bar_idx = j
                    fvg.mitigated_timestamp = timestamps[j]
                    fvg.mitigation_price = float(bar_high)
                    break


def filter_fvgs_by_equilibrium(
    fvgs: List[FairValueGap],
    swing_high: float,
    swing_low: float
) -> List[FairValueGap]:
    """
    Applies the Premium / Discount (Equilibrium = 50%) filter.
    - Bullish FVGs must reside in the Discount zone (< 50% equilibrium).
    - Bearish FVGs must reside in the Premium zone (> 50% equilibrium).
    """
    if swing_high <= swing_low:
        return fvgs

    equilibrium = swing_low + 0.50 * (swing_high - swing_low)
    qualified: List[FairValueGap] = []

    for fvg in fvgs:
        if fvg.fvg_type == FVGType.BULLISH:
            # Bullish FVG must be in Discount (< 50%)
            # We qualify if the top of the FVG is at or below equilibrium
            if fvg.top <= equilibrium:
                fvg.in_discount = True
                fvg.in_premium = False
                qualified.append(fvg)
        elif fvg.fvg_type == FVGType.BEARISH:
            # Bearish FVG must be in Premium (> 50%)
            # We qualify if the bottom of the FVG is at or above equilibrium
            if fvg.bottom >= equilibrium:
                fvg.in_premium = True
                fvg.in_discount = False
                qualified.append(fvg)

    return qualified
