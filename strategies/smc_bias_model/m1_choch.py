"""
SMC Bias Model: 1-Minute Reversal Confirmation (m1_choch.py)
=============================================================
Rule 4 of the 5-Rule SMC Intraday Bias Model:
- Timeframe: 1-Minute (M1) Structural Confirmation
- Mechanism:
  1. Observe the Pullback: After the session liquidation sweep (Rule 3), price retraces
     into the higher timeframe supply/demand area. On the 1-minute chart, this forms
     a counter-trend (e.g. higher highs and higher lows in a bearish environment).
  2. The Trigger Event (M1 CHoCH): Wait for the 1-minute chart to break the low (for short)
     or high (for long) that produced the most recent extreme.
  3. Body Close Filter: The break must occur via an impulsive **candle body close**.
     Wick breaks are classified as fakeouts and rejected.
  4. Invalidation Extreme: The extreme wick that formed prior to the CHoCH becomes the
     anchor for the structural stop loss (1-2 pips/ticks beyond).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import pandas as pd
import numpy as np


@dataclass
class MicroSwingPoint:
    bar_idx: int
    timestamp: Any
    price: float
    is_high: bool
    broken: bool = False


@dataclass
class M1CHoCHConfirmation:
    event_type: str  # "M1_CHOCH_BULLISH" or "M1_CHOCH_BEARISH"
    bar_idx: int
    timestamp: Any
    trigger_price: float
    broken_level: float
    origin_swing: MicroSwingPoint
    structural_extreme_price: float  # Invalidation extreme
    displacement_ratio: float


class M1CHoCHDetector:
    """
    Tracks 1-minute micro-swings and identifies impulsive Change of Character (CHoCH)
    confirmations following session liquidation sweeps.
    """

    def __init__(self, left_bars: int = 2, right_bars: int = 2):
        self.left_bars = left_bars
        self.right_bars = right_bars

    def find_micro_pivots(self, df_m1: pd.DataFrame) -> List[MicroSwingPoint]:
        """Identifies microscopic 1-minute swing points."""
        if len(df_m1) < (self.left_bars + self.right_bars + 1):
            return []

        highs = df_m1["high"].values
        lows = df_m1["low"].values
        timestamps = df_m1.index.values
        n = len(df_m1)
        pivots: List[MicroSwingPoint] = []

        for i in range(self.left_bars, n - self.right_bars):
            cur_h = highs[i]
            cur_l = lows[i]

            is_h = True
            for off in range(-self.left_bars, self.right_bars + 1):
                if off < 0 and highs[i + off] > cur_h:
                    is_h = False
                    break
                elif off > 0 and highs[i + off] >= cur_h:
                    is_h = False
                    break
            if is_h:
                pivots.append(MicroSwingPoint(bar_idx=i, timestamp=timestamps[i], price=float(cur_h), is_high=True))

            is_l = True
            for off in range(-self.left_bars, self.right_bars + 1):
                if off < 0 and lows[i + off] < cur_l:
                    is_l = False
                    break
                elif off > 0 and lows[i + off] <= cur_l:
                    is_l = False
                    break
            if is_l:
                pivots.append(MicroSwingPoint(bar_idx=i, timestamp=timestamps[i], price=float(cur_l), is_high=False))

        # If not enough pivots found with strict lookback, try 1-bar micro pivots
        if (not [p for p in pivots if p.is_high] or not [p for p in pivots if not p.is_high]) and self.left_bars > 1:
            for i in range(1, n - 1):
                if highs[i] > highs[i - 1] and highs[i] >= highs[i + 1]:
                    if not any(p.bar_idx == i and p.is_high for p in pivots):
                        pivots.append(MicroSwingPoint(bar_idx=i, timestamp=timestamps[i], price=float(highs[i]), is_high=True))
                if lows[i] < lows[i - 1] and lows[i] <= lows[i + 1]:
                    if not any(p.bar_idx == i and not p.is_high for p in pivots):
                        pivots.append(MicroSwingPoint(bar_idx=i, timestamp=timestamps[i], price=float(lows[i]), is_high=False))

        pivots.sort(key=lambda p: p.bar_idx)
        return pivots

    def detect_m1_choch(
        self,
        df_m1: pd.DataFrame,
        bias: str,
        sweep_time: Optional[Any] = None,
        sweep_level: Optional[float] = None
    ) -> Optional[M1CHoCHConfirmation]:
        """
        Scans recent 1-minute candles for an impulsive Change of Character in the direction
        of the M15 bias following the session liquidation sweep.

        - For Bearish Bias: We look for price to break below the most recent M1 Higher Low
          with a solid candle body close. The structural invalidation extreme is the Highest High
          made during the sweep.
        - For Bullish Bias: We look for price to break above the most recent M1 Lower High
          with a solid candle body close. The structural invalidation extreme is the Lowest Low
          made during the sweep.
        """
        if len(df_m1) < 10:
            return None

        pivots = self.find_micro_pivots(df_m1)
        if not pivots:
            return None

        high_pivots = [p for p in pivots if p.is_high]
        low_pivots = [p for p in pivots if not p.is_high]

        closes = df_m1["close"].values
        opens = df_m1["open"].values
        highs = df_m1["high"].values
        lows = df_m1["low"].values
        timestamps = df_m1.index.values
        n = len(df_m1)

        # Average candle body size for displacement filter
        body_sizes = np.abs(closes - opens)
        avg_body = float(np.mean(body_sizes[-20:])) if len(body_sizes) >= 20 else 0.01
        avg_body = max(avg_body, 0.005)

        if bias == "BEARISH":
            # Looking for M1 CHoCH Bearish: Price makes a peak (HH), then breaks below the key Higher Low (HL)
            if not low_pivots or not high_pivots:
                return None

            hh = high_pivots[-1]
            hl_cand = [p for p in low_pivots if p.bar_idx < hh.bar_idx]
            if not hl_cand:
                hl_cand = low_pivots
            key_low = hl_cand[-1]
            level_to_break = key_low.price

            search_start = max(hh.bar_idx + 1, n - 6)
            for i in range(search_start, n):
                if closes[i] < level_to_break:
                    displacement = abs(closes[i] - opens[i]) / avg_body
                    return M1CHoCHConfirmation(
                        event_type="M1_CHOCH_BEARISH",
                        bar_idx=i,
                        timestamp=timestamps[i],
                        trigger_price=float(closes[i]),
                        broken_level=level_to_break,
                        origin_swing=key_low,
                        structural_extreme_price=hh.price,
                        displacement_ratio=displacement
                    )

        elif bias == "BULLISH":
            # Looking for M1 CHoCH Bullish: Price makes a trough (LL), then breaks above the key Lower High (LH)
            if not high_pivots or not low_pivots:
                return None

            ll = low_pivots[-1]
            lh_cand = [p for p in high_pivots if p.bar_idx < ll.bar_idx]
            if not lh_cand:
                lh_cand = high_pivots
            key_high = lh_cand[-1]
            level_to_break = key_high.price

            search_start = max(ll.bar_idx + 1, n - 6)
            for i in range(search_start, n):
                if closes[i] > level_to_break:
                    displacement = abs(closes[i] - opens[i]) / avg_body
                    return M1CHoCHConfirmation(
                        event_type="M1_CHOCH_BULLISH",
                        bar_idx=i,
                        timestamp=timestamps[i],
                        trigger_price=float(closes[i]),
                        broken_level=level_to_break,
                        origin_swing=key_high,
                        structural_extreme_price=ll.price,
                        displacement_ratio=displacement
                    )

        return None
