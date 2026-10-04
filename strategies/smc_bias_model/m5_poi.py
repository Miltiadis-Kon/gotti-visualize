"""
SMC Bias Model: 5-Minute POI Selection & Confluence (m5_poi.py)
===============================================================
Rule 5 of the 5-Rule SMC Intraday Bias Model:
- Timeframe: 5-Minute (M5) Execution & Trigger
- Point of Interest (POI) Menu:
  1. Fair Value Gap (FVG): 3-bar imbalance surrounding a displacement candle.
  2. Order Block (OB): The final counter-trend candle before the aggressive impulse
     move that broke structure / triggered CHoCH.
  3. Inverted Fair Value Gap (IFVG): A prior opposing FVG that was violated with
     displacement, flipping from support to resistance (or vice versa).
- Confluence Stacking:
  The highest-probability setups occur when an M5 Order Block overlaps an M5 FVG or IFVG.
- Order Placement:
  Place a Limit Order at the proximate boundary of the chosen M5 POI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any
import pandas as pd
import numpy as np


class POIType(Enum):
    ORDER_BLOCK = "ORDER_BLOCK"
    FAIR_VALUE_GAP = "FAIR_VALUE_GAP"
    INVERTED_FVG = "INVERTED_FVG"
    CONFLUENCE_OB_FVG = "CONFLUENCE_OB_FVG"


@dataclass
class PointOfInterest:
    poi_type: POIType
    bias: str  # "BULLISH" or "BEARISH"
    proximate_price: float  # Limit entry price (closest edge to current price)
    distal_price: float     # Far edge (near structural invalidation)
    bar_idx: int
    timestamp: Any
    confluence_score: float  # 1.0 = standard, 2.0+ = stacked confluence
    description: str
    mitigated: bool = False

    @property
    def top(self) -> float:
        return max(self.proximate_price, self.distal_price)

    @property
    def bottom(self) -> float:
        return min(self.proximate_price, self.distal_price)

    def contains_price(self, price: float) -> bool:
        return self.bottom <= price <= self.top


class M5POISelector:
    """
    Identifies 5-minute institutional Points of Interest: Order Blocks, FVGs,
    Inverted FVGs (IFVGs), and detects multi-factor confluence.
    """

    def __init__(self, min_displacement_ratio: float = 1.0):
        self.min_displacement_ratio = min_displacement_ratio

    def detect_5m_fvgs(self, df_m5: pd.DataFrame) -> List[PointOfInterest]:
        """
        Detects 3-bar Fair Value Gaps on the 5-minute chart using the open-source
        smartmoneyconcepts library, with built-in native fallback.
        """
        if len(df_m5) < 3:
            return []

        # 1. Try smartmoneyconcepts open-source library
        try:
            from smartmoneyconcepts import smc
            ohlc = df_m5[["open", "high", "low", "close", "volume"]].copy() if "volume" in df_m5.columns else df_m5[["open", "high", "low", "close"]].copy()
            if "volume" not in ohlc.columns:
                ohlc["volume"] = 100_000
            fvg_res = smc.fvg(ohlc, join_consecutive=False)
            fvgs_smc: List[PointOfInterest] = []
            timestamps = df_m5.index.values

            for i in range(len(fvg_res)):
                f_val = fvg_res["FVG"].iloc[i]
                if pd.isna(f_val):
                    continue
                top = float(fvg_res["Top"].iloc[i])
                bottom = float(fvg_res["Bottom"].iloc[i])
                mitigated_idx = fvg_res["MitigatedIndex"].iloc[i] if "MitigatedIndex" in fvg_res.columns else 0.0
                is_mitigated = bool(mitigated_idx > 0 and mitigated_idx < len(df_m5))

                if f_val == 1.0:  # Bullish FVG
                    fvgs_smc.append(
                        PointOfInterest(
                            poi_type=POIType.FAIR_VALUE_GAP,
                            bias="BULLISH",
                            proximate_price=top,
                            distal_price=bottom,
                            bar_idx=i,
                            timestamp=timestamps[i],
                            confluence_score=1.0,
                            description=f"5M Bullish FVG [{bottom:.2f} - {top:.2f}]",
                            mitigated=is_mitigated
                        )
                    )
                elif f_val == -1.0:  # Bearish FVG
                    fvgs_smc.append(
                        PointOfInterest(
                            poi_type=POIType.FAIR_VALUE_GAP,
                            bias="BEARISH",
                            proximate_price=bottom,
                            distal_price=top,
                            bar_idx=i,
                            timestamp=timestamps[i],
                            confluence_score=1.0,
                            description=f"5M Bearish FVG [{bottom:.2f} - {top:.2f}]",
                            mitigated=is_mitigated
                        )
                    )

            if fvgs_smc:
                return fvgs_smc
        except Exception:
            pass

        # 2. Native Fallback Detection
        highs = df_m5["high"].values
        lows = df_m5["low"].values
        closes = df_m5["close"].values
        opens = df_m5["open"].values
        timestamps = df_m5.index.values
        fvgs: List[PointOfInterest] = []

        for i in range(2, len(df_m5)):
            c1_high = highs[i - 2]
            c1_low = lows[i - 2]
            c2_open = opens[i - 1]
            c2_close = closes[i - 1]
            c3_high = highs[i]
            c3_low = lows[i]

            # Bullish FVG: Bar 1 High < Bar 3 Low
            if c3_low > c1_high:
                gap_size = c3_low - c1_high
                if gap_size > 0.01:
                    fvgs.append(
                        PointOfInterest(
                            poi_type=POIType.FAIR_VALUE_GAP,
                            bias="BULLISH",
                            proximate_price=float(c3_low),   # Entry at top of gap
                            distal_price=float(c1_high),     # Stop near bottom of gap
                            bar_idx=i - 1,
                            timestamp=timestamps[i - 1],
                            confluence_score=1.0,
                            description=f"5M Bullish FVG [{c1_high:.2f} - {c3_low:.2f}]"
                        )
                    )

            # Bearish FVG: Bar 1 Low > Bar 3 High
            elif c1_low > c3_high:
                gap_size = c1_low - c3_high
                if gap_size > 0.01:
                    fvgs.append(
                        PointOfInterest(
                            poi_type=POIType.FAIR_VALUE_GAP,
                            bias="BEARISH",
                            proximate_price=float(c3_high),  # Entry at bottom of gap
                            distal_price=float(c1_low),      # Stop near top of gap
                            bar_idx=i - 1,
                            timestamp=timestamps[i - 1],
                            confluence_score=1.0,
                            description=f"5M Bearish FVG [{c3_high:.2f} - {c1_low:.2f}]"
                        )
                    )

        return fvgs

    def detect_5m_order_blocks(self, df_m5: pd.DataFrame, bias: str) -> List[PointOfInterest]:
        """
        Detects Order Blocks using the open-source smartmoneyconcepts library,
        with built-in native fallback.
        - Bearish OB: Counter-trend candle before aggressive downward displacement.
        - Bullish OB: Counter-trend candle before aggressive upward displacement.
        """
        if len(df_m5) < 5:
            return []

        # 1. Try smartmoneyconcepts open-source library
        try:
            from smartmoneyconcepts import smc
            ohlc = df_m5[["open", "high", "low", "close", "volume"]].copy() if "volume" in df_m5.columns else df_m5[["open", "high", "low", "close"]].copy()
            if "volume" not in ohlc.columns:
                ohlc["volume"] = 100_000
            swings = smc.swing_highs_lows(ohlc, swing_length=3)
            ob_res = smc.ob(ohlc, swings)
            obs_smc: List[PointOfInterest] = []
            timestamps = df_m5.index.values

            for i in range(len(ob_res)):
                ob_val = ob_res["OB"].iloc[i]
                if pd.isna(ob_val):
                    continue
                top = float(ob_res["Top"].iloc[i])
                bottom = float(ob_res["Bottom"].iloc[i])
                mitigated_idx = ob_res["MitigatedIndex"].iloc[i] if "MitigatedIndex" in ob_res.columns else 0.0
                is_mitigated = bool(mitigated_idx > 0 and mitigated_idx < len(df_m5))
                strength_pct = float(ob_res["Percentage"].iloc[i]) if "Percentage" in ob_res.columns and not pd.isna(ob_res["Percentage"].iloc[i]) else 50.0
                conf_score = 1.0 + (strength_pct / 100.0) * 0.5

                if ob_val == 1.0 and bias == "BULLISH":
                    obs_smc.append(
                        PointOfInterest(
                            poi_type=POIType.ORDER_BLOCK,
                            bias="BULLISH",
                            proximate_price=top,
                            distal_price=bottom,
                            bar_idx=i,
                            timestamp=timestamps[i],
                            confluence_score=conf_score,
                            description=f"5M Bullish Order Block @ ${bottom:.2f}-${top:.2f} ({strength_pct:.0f}%)",
                            mitigated=is_mitigated
                        )
                    )
                elif ob_val == -1.0 and bias == "BEARISH":
                    obs_smc.append(
                        PointOfInterest(
                            poi_type=POIType.ORDER_BLOCK,
                            bias="BEARISH",
                            proximate_price=bottom,
                            distal_price=top,
                            bar_idx=i,
                            timestamp=timestamps[i],
                            confluence_score=conf_score,
                            description=f"5M Bearish Order Block @ ${bottom:.2f}-${top:.2f} ({strength_pct:.0f}%)",
                            mitigated=is_mitigated
                        )
                    )

            if obs_smc:
                return obs_smc
        except Exception:
            pass

        # 2. Native Fallback Detection
        opens = df_m5["open"].values
        highs = df_m5["high"].values
        lows = df_m5["low"].values
        closes = df_m5["close"].values
        timestamps = df_m5.index.values
        obs: List[PointOfInterest] = []

        n = len(df_m5)
        ranges = highs - lows
        avg_range = float(np.mean(ranges[-15:])) if len(ranges) >= 15 else 0.05
        avg_range = max(avg_range, 0.01)

        if bias == "BEARISH":
            for i in range(1, n - 1):
                if closes[i] > opens[i]:
                    disp_move = opens[i + 1] - closes[i + 1]
                    if closes[i + 1] < opens[i + 1] and disp_move >= 1.0 * avg_range and closes[i + 1] < lows[i]:
                        obs.append(
                            PointOfInterest(
                                poi_type=POIType.ORDER_BLOCK,
                                bias="BEARISH",
                                proximate_price=float(lows[i]),   # Entry at candle low
                                distal_price=float(highs[i]),     # Far edge at candle high
                                bar_idx=i,
                                timestamp=timestamps[i],
                                confluence_score=1.2,
                                description=f"5M Bearish Order Block @ ${lows[i]:.2f}-${highs[i]:.2f}"
                            )
                        )

        elif bias == "BULLISH":
            for i in range(1, n - 1):
                if closes[i] < opens[i]:
                    disp_move = closes[i + 1] - opens[i + 1]
                    if closes[i + 1] > opens[i + 1] and disp_move >= 1.0 * avg_range and closes[i + 1] > highs[i]:
                        obs.append(
                            PointOfInterest(
                                poi_type=POIType.ORDER_BLOCK,
                                bias="BULLISH",
                                proximate_price=float(highs[i]),  # Entry at candle high
                                distal_price=float(lows[i]),      # Far edge at candle low
                                bar_idx=i,
                                timestamp=timestamps[i],
                                confluence_score=1.2,
                                description=f"5M Bullish Order Block @ ${lows[i]:.2f}-${highs[i]:.2f}"
                            )
                        )

        return obs

    def detect_5m_inverted_fvgs(self, df_m5: pd.DataFrame, prior_fvgs: List[PointOfInterest], bias: str) -> List[PointOfInterest]:
        """
        Detects Inverted Fair Value Gaps (IFVGs):
        A prior opposing FVG that was invalidated by a solid body close through its distal boundary,
        flipping it into support (for bullish bias) or resistance (for bearish bias).
        """
        if not prior_fvgs or len(df_m5) < 3:
            return []

        closes = df_m5["close"].values
        timestamps = df_m5.index.values
        cur_close = closes[-1]
        cur_ts = timestamps[-1]
        ifvgs: List[PointOfInterest] = []

        for fvg in prior_fvgs:
            # Bullish bias: Look for prior Bearish FVG broken above
            if bias == "BULLISH" and fvg.bias == "BEARISH":
                if cur_close > fvg.top:
                    ifvgs.append(
                        PointOfInterest(
                            poi_type=POIType.INVERTED_FVG,
                            bias="BULLISH",
                            proximate_price=fvg.top,     # Flipped to support
                            distal_price=fvg.bottom,
                            bar_idx=len(df_m5) - 1,
                            timestamp=cur_ts,
                            confluence_score=1.3,
                            description=f"5M Bullish Inverted FVG (flipped from Bearish FVG)"
                        )
                    )
            # Bearish bias: Look for prior Bullish FVG broken below
            elif bias == "BEARISH" and fvg.bias == "BULLISH":
                if cur_close < fvg.bottom:
                    ifvgs.append(
                        PointOfInterest(
                            poi_type=POIType.INVERTED_FVG,
                            bias="BEARISH",
                            proximate_price=fvg.bottom,  # Flipped to resistance
                            distal_price=fvg.top,
                            bar_idx=len(df_m5) - 1,
                            timestamp=cur_ts,
                            confluence_score=1.3,
                            description=f"5M Bearish Inverted FVG (flipped from Bullish FVG)"
                        )
                    )

        return ifvgs

    def select_best_poi_with_confluence(
        self,
        df_m5: pd.DataFrame,
        bias: str,
        prior_opposing_fvgs: Optional[List[PointOfInterest]] = None,
        lookback_bars: int = 15,
        structural_extreme: Optional[float] = None,
        m15_target: Optional[float] = None,
        choch_broken_level: Optional[float] = None,
    ) -> Optional[PointOfInterest]:
        """
        Extracts M5 POIs from the recent displacement leg, identifies OB + FVG overlaps
        (confluence stacking), enforces structural alignment, and returns the highest-probability
        POI for limit order placement.
        """
        recent_df = df_m5.iloc[-lookback_bars:] if len(df_m5) > lookback_bars else df_m5

        fvgs = [f for f in self.detect_5m_fvgs(recent_df) if f.bias == bias]
        obs = self.detect_5m_order_blocks(recent_df, bias)
        ifvgs = self.detect_5m_inverted_fvgs(recent_df, prior_opposing_fvgs or [], bias)

        all_candidates: List[PointOfInterest] = []

        # 1. Check for Order Block + FVG Confluence
        for ob in obs:
            has_overlap = False
            for f in fvgs:
                overlap_low = max(ob.bottom, f.bottom)
                overlap_high = min(ob.top, f.top)
                if overlap_high >= overlap_low:
                    has_overlap = True
                    confluent_poi = PointOfInterest(
                        poi_type=POIType.CONFLUENCE_OB_FVG,
                        bias=bias,
                        proximate_price=ob.proximate_price,
                        distal_price=ob.distal_price,
                        bar_idx=ob.bar_idx,
                        timestamp=ob.timestamp,
                        confluence_score=2.5,
                        description=f"CONFLUENCE: 5M OB + FVG @ ${ob.proximate_price:.2f}"
                    )
                    all_candidates.append(confluent_poi)
                    break

            if not has_overlap:
                all_candidates.append(ob)

        all_candidates.extend(fvgs)
        all_candidates.extend(ifvgs)

        # 2. Filter candidates by structural validity relative to invalidation extreme and macro target
        valid_candidates: List[PointOfInterest] = []
        for cand in all_candidates:
            if structural_extreme is not None:
                if bias == "BULLISH" and cand.proximate_price <= structural_extreme:
                    continue
                if bias == "BEARISH" and cand.proximate_price >= structural_extreme:
                    continue

            if m15_target is not None:
                if bias == "BULLISH" and cand.proximate_price >= m15_target:
                    continue
                if bias == "BEARISH" and cand.proximate_price <= m15_target:
                    continue

            valid_candidates.append(cand)

        # 3. Fallback to CHoCH Breaker level if no qualified POIs in recent bars
        if not valid_candidates and choch_broken_level is not None:
            cur_ts = recent_df.index[-1] if not recent_df.empty else None
            is_valid_breaker = True
            if structural_extreme is not None:
                if bias == "BULLISH" and choch_broken_level <= structural_extreme:
                    is_valid_breaker = False
                elif bias == "BEARISH" and choch_broken_level >= structural_extreme:
                    is_valid_breaker = False

            if m15_target is not None:
                if bias == "BULLISH" and choch_broken_level >= m15_target:
                    is_valid_breaker = False
                elif bias == "BEARISH" and choch_broken_level <= m15_target:
                    is_valid_breaker = False

            if is_valid_breaker:
                valid_candidates.append(
                    PointOfInterest(
                        poi_type=POIType.ORDER_BLOCK,
                        bias=bias,
                        proximate_price=choch_broken_level,
                        distal_price=structural_extreme if structural_extreme is not None else choch_broken_level,
                        bar_idx=len(recent_df) - 1,
                        timestamp=cur_ts,
                        confluence_score=1.0,
                        description=f"5M CHoCH Breaker Level @ ${choch_broken_level:.2f}"
                    )
                )

        if not valid_candidates:
            return None

        # Sort by confluence score descending, then recency
        valid_candidates.sort(key=lambda p: (p.confluence_score, p.bar_idx), reverse=True)
        return valid_candidates[0]
