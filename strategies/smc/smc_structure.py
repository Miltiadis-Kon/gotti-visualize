"""
SMC Market Structure Engine (smc_structure.py)
==============================================
Institutional market structure tracking:
  - Fractal swing pivot detection (Swing Highs & Swing Lows)
  - Trend classification (HH, HL, LH, LL)
  - Liquidity Sweep (Fakeout) filter: Candle wick pierces level, but body closes inside
  - Valid CHoCH (Change of Character): Solid candle body close past key structural pivot
  - Break of Structure (BOS): Trend-continuation candle body breaks
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class StructureType(str, Enum):
    HH = "HH"  # Higher High
    LH = "LH"  # Lower High
    HL = "HL"  # Higher Low
    LL = "LL"  # Lower Low


class TrendBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class SwingPoint:
    """Represents a structural swing high or swing low pivot."""
    bar_idx: int
    timestamp: Any
    price: float
    swing_type: SwingType
    classification: Optional[StructureType] = None
    swept: bool = False
    broken_by_body: bool = False
    broken_bar_idx: Optional[int] = None
    broken_timestamp: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_idx": self.bar_idx,
            "timestamp": str(self.timestamp),
            "price": self.price,
            "swing_type": self.swing_type.value,
            "classification": self.classification.value if self.classification else None,
            "swept": self.swept,
            "broken_by_body": self.broken_by_body,
            "broken_bar_idx": self.broken_bar_idx,
            "broken_timestamp": str(self.broken_timestamp) if self.broken_timestamp else None,
        }


@dataclass
class StructureShiftEvent:
    """Records a Break of Structure (BOS), Liquidity Sweep, or Change of Character (CHoCH)."""
    event_type: str  # "CHOCH_BULLISH", "CHOCH_BEARISH", "BOS_BULLISH", "BOS_BEARISH", "SWEEP_HIGH", "SWEEP_LOW"
    bar_idx: int
    timestamp: Any
    trigger_price: float  # Close or Wick price triggering event
    level_price: float    # The swing pivot level tested
    pivot: SwingPoint
    origin_swing: Optional[SwingPoint] = None  # The swing origin that initiated the shift leg


class FractalPivotDetector:
    """
    Detects fractal swing highs and lows and classifies market structure.
    """

    def __init__(self, left_bars: int = 3, right_bars: int = 3):
        self.left_bars = left_bars
        self.right_bars = right_bars

    def find_pivots(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Scans OHLCV DataFrame and locates all fractal swing highs and lows.
        Ensures strict no-lookahead: right_bars must have completed.
        """
        pivots: List[SwingPoint] = []
        if len(df) < self.left_bars + self.right_bars + 1:
            return pivots

        highs = df["high"].values
        lows = df["low"].values
        timestamps = df.index.values

        n = len(df)
        for i in range(self.left_bars, n - self.right_bars):
            cur_high = highs[i]
            cur_low = lows[i]

            # Swing High condition
            is_swing_high = True
            for offset in range(-self.left_bars, self.right_bars + 1):
                if offset != 0 and highs[i + offset] >= cur_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                pivots.append(
                    SwingPoint(
                        bar_idx=i,
                        timestamp=timestamps[i],
                        price=float(cur_high),
                        swing_type=SwingType.HIGH,
                    )
                )

            # Swing Low condition
            is_swing_low = True
            for offset in range(-self.left_bars, self.right_bars + 1):
                if offset != 0 and lows[i + offset] <= cur_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                pivots.append(
                    SwingPoint(
                        bar_idx=i,
                        timestamp=timestamps[i],
                        price=float(cur_low),
                        swing_type=SwingType.LOW,
                    )
                )

        # Sort pivots by bar index
        pivots.sort(key=lambda p: p.bar_idx)
        self._classify_pivots(pivots)
        return pivots

    def _classify_pivots(self, pivots: List[SwingPoint]) -> None:
        """Classifies pivots into HH, HL, LH, LL."""
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None

        for p in pivots:
            if p.swing_type == SwingType.HIGH:
                if last_high is None:
                    p.classification = StructureType.HH
                elif p.price > last_high.price:
                    p.classification = StructureType.HH
                else:
                    p.classification = StructureType.LH
                last_high = p
            else:
                if last_low is None:
                    p.classification = StructureType.LL
                elif p.price < last_low.price:
                    p.classification = StructureType.LL
                else:
                    p.classification = StructureType.HL
                last_low = p


class MarketStructure:
    """
    Maintains real-time market structure, identifying CHoCH vs Wick Sweeps.
    """

    def __init__(self, left_bars: int = 3, right_bars: int = 3):
        self.detector = FractalPivotDetector(left_bars, right_bars)
        self.pivots: List[SwingPoint] = []
        self.trend: TrendBias = TrendBias.NEUTRAL
        self.events: List[StructureShiftEvent] = []
        
        # Key reference levels
        self.last_key_hl: Optional[SwingPoint] = None  # Last HL responsible for the highest high
        self.last_key_lh: Optional[SwingPoint] = None  # Last LH responsible for the lowest low
        self.recent_high: Optional[SwingPoint] = None
        self.recent_low: Optional[SwingPoint] = None

    def analyze(self, df: pd.DataFrame) -> Tuple[TrendBias, List[StructureShiftEvent]]:
        """
        Runs comprehensive structural analysis across the provided DataFrame.
        """
        self.pivots = self.detector.find_pivots(df)
        self.events.clear()
        self.trend = TrendBias.NEUTRAL

        if len(self.pivots) < 2:
            return self.trend, self.events

        # Identify baseline trend from first 2 pivots of each type
        highs = [p for p in self.pivots if p.swing_type == SwingType.HIGH]
        lows = [p for p in self.pivots if p.swing_type == SwingType.LOW]

        if highs and lows:
            if highs[-1].price > highs[0].price and lows[-1].price > lows[0].price:
                self.trend = TrendBias.BULLISH
            elif highs[-1].price < highs[0].price and lows[-1].price < lows[0].price:
                self.trend = TrendBias.BEARISH

        # Process candles chronologically to track sweeps and body CHoCH
        self._track_chronological_structure(df)
        return self.trend, self.events

    def _track_chronological_structure(self, df: pd.DataFrame) -> None:
        """
        Simulates bar-by-bar progression to detect sweeps vs valid body-close CHoCH.
        """
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df.index.values

        active_highs: List[SwingPoint] = []
        active_lows: List[SwingPoint] = []

        pivot_map = {p.bar_idx: p for p in self.pivots}

        for i in range(len(df)):
            # If a new pivot formed at i - right_bars, register it
            confirmed_idx = i - self.detector.right_bars
            if confirmed_idx in pivot_map:
                new_pivot = pivot_map[confirmed_idx]
                if new_pivot.swing_type == SwingType.HIGH:
                    active_highs.append(new_pivot)
                    self.recent_high = new_pivot
                    # In an uptrend, update key HL as the lowest low between previous high and this high
                    if self.trend == TrendBias.BULLISH and active_lows:
                        # find lowest low before this high
                        valid_lows = [l for l in active_lows if l.bar_idx < new_pivot.bar_idx]
                        if valid_lows:
                            self.last_key_hl = valid_lows[-1]
                else:
                    active_lows.append(new_pivot)
                    self.recent_low = new_pivot
                    # In a downtrend, update key LH as the highest high before this low
                    if self.trend == TrendBias.BEARISH and active_highs:
                        valid_highs = [h for h in active_highs if h.bar_idx < new_pivot.bar_idx]
                        if valid_highs:
                            self.last_key_lh = valid_highs[-1]

            cur_high = highs[i]
            cur_low = lows[i]
            cur_close = closes[i]
            cur_ts = timestamps[i]

            # 1. Check Bullish Break / CHoCH:
            # Price tests the last active unbroken Swing High
            unbroken_highs = [h for h in active_highs if not h.broken_by_body]
            if unbroken_highs:
                target_high = unbroken_highs[-1]
                target_level = target_high.price
                if cur_high > target_level:
                    if cur_close > target_level:
                        # VALID BODY CLOSE: CHoCH / BOS Bullish
                        ev_name = "CHOCH_BULLISH" if self.trend != TrendBias.BULLISH else "BOS_BULLISH"
                        self.trend = TrendBias.BULLISH
                        target_high.broken_by_body = True
                        target_high.broken_bar_idx = i
                        target_high.broken_timestamp = cur_ts
                        event = StructureShiftEvent(
                            event_type=ev_name,
                            bar_idx=i,
                            timestamp=cur_ts,
                            trigger_price=float(cur_close),
                            level_price=target_level,
                            pivot=target_high,
                            origin_swing=active_lows[-1] if active_lows else None,
                        )
                        self.events.append(event)
                    else:
                        # LIQUIDITY SWEEP (FAKEOUT): Wick pierced, but closed inside
                        if not target_high.swept:
                            target_high.swept = True
                            self.events.append(
                                StructureShiftEvent(
                                    event_type="SWEEP_HIGH",
                                    bar_idx=i,
                                    timestamp=cur_ts,
                                    trigger_price=float(cur_high),
                                    level_price=target_level,
                                    pivot=target_high,
                                    origin_swing=active_lows[-1] if active_lows else None,
                                )
                            )

            # 2. Check Bearish Break / CHoCH:
            # Price tests the last active unbroken Swing Low
            unbroken_lows = [l for l in active_lows if not l.broken_by_body]
            if unbroken_lows:
                target_low = unbroken_lows[-1]
                target_level = target_low.price
                if cur_low < target_level:
                    if cur_close < target_level:
                        # VALID BODY CLOSE: CHoCH / BOS Bearish
                        ev_name = "CHOCH_BEARISH" if self.trend != TrendBias.BEARISH else "BOS_BEARISH"
                        self.trend = TrendBias.BEARISH
                        target_low.broken_by_body = True
                        target_low.broken_bar_idx = i
                        target_low.broken_timestamp = cur_ts
                        event = StructureShiftEvent(
                            event_type=ev_name,
                            bar_idx=i,
                            timestamp=cur_ts,
                            trigger_price=float(cur_close),
                            level_price=target_level,
                            pivot=target_low,
                            origin_swing=active_highs[-1] if active_highs else None,
                        )
                        self.events.append(event)
                    else:
                        # LIQUIDITY SWEEP (FAKEOUT): Wick pierced, but closed inside
                        if not target_low.swept:
                            target_low.swept = True
                            self.events.append(
                                StructureShiftEvent(
                                    event_type="SWEEP_LOW",
                                    bar_idx=i,
                                    timestamp=cur_ts,
                                    trigger_price=float(cur_low),
                                    level_price=target_level,
                                    pivot=target_low,
                                    origin_swing=active_highs[-1] if active_highs else None,
                                )
                            )


def detect_structure_shifts(
    df: pd.DataFrame, left_bars: int = 3, right_bars: int = 3
) -> Tuple[TrendBias, List[SwingPoint], List[StructureShiftEvent]]:
    """Helper function to analyze structure and return current trend, pivots, and shifts."""
    ms = MarketStructure(left_bars, right_bars)
    trend, events = ms.analyze(df)
    return trend, ms.pivots, events
