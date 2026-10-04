"""
SMC Bias Model: Structure & The Box Method (bias_structure.py)
=============================================================
Rule 1 of the 5-Rule SMC Intraday Bias Model:
- Timeframe: 15-Minute (M15) Macro Trend & Context
- The Box Method:
  1. Locate the most recent external structural swing high and swing low that created
     the current trend leg.
  2. Draw an external box from the swing high to the swing low. Any price action
     developing inside this box is classified as *internal structure* and disregarded.
  3. Valid Structural Breaks: A structural shift requires a **candle body close**
     past the external swing point. Wicks do NOT constitute a break; they represent sweeps.
  4. Directional Bias:
     - Bearish Bias: M15 printed a lower external low (body close). Target: next external low.
     - Bullish Bias: M15 printed a higher external high (body close). Target: next external high.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


class BiasDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SwingPointType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass
class ExternalSwingPoint:
    bar_idx: int
    timestamp: Any
    price: float
    swing_type: SwingPointType
    swept: bool = False
    broken_by_body: bool = False
    broken_bar_idx: Optional[int] = None
    broken_timestamp: Optional[Any] = None


@dataclass
class ExternalBoxRange:
    high_point: ExternalSwingPoint
    low_point: ExternalSwingPoint
    bias: BiasDirection
    target_price: float
    invalidation_price: float

    @property
    def upper_bound(self) -> float:
        return self.high_point.price

    @property
    def lower_bound(self) -> float:
        return self.low_point.price

    def is_inside_box(self, price: float) -> bool:
        """Returns True if price is inside the internal noise box."""
        return self.lower_bound <= price <= self.upper_bound


class M15BoxStructureDetector:
    """
    Identifies 15-Minute external swing points, filters internal noise,
    and enforces the Box Rule for directional bias.
    """

    def __init__(self, left_bars: int = 3, right_bars: int = 3):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.external_highs: List[ExternalSwingPoint] = []
        self.external_lows: List[ExternalSwingPoint] = []
        self.bias: BiasDirection = BiasDirection.NEUTRAL
        self.current_box: Optional[ExternalBoxRange] = None

    def find_fractal_pivots(self, df_m15: pd.DataFrame) -> List[ExternalSwingPoint]:
        """
        Extracts fractal swing highs and lows from 15-minute candles.
        """
        if len(df_m15) < (self.left_bars + self.right_bars + 1):
            return []

        highs = df_m15["high"].values
        lows = df_m15["low"].values
        timestamps = df_m15.index.values
        n = len(df_m15)
        pivots: List[ExternalSwingPoint] = []

        for i in range(self.left_bars, n - self.right_bars):
            cur_high = highs[i]
            cur_low = lows[i]

            # Swing High
            is_high = True
            for offset in range(-self.left_bars, self.right_bars + 1):
                if offset != 0 and highs[i + offset] >= cur_high:
                    is_high = False
                    break
            if is_high:
                pivots.append(
                    ExternalSwingPoint(
                        bar_idx=i,
                        timestamp=timestamps[i],
                        price=float(cur_high),
                        swing_type=SwingPointType.HIGH,
                    )
                )

            # Swing Low
            is_low = True
            for offset in range(-self.left_bars, self.right_bars + 1):
                if offset != 0 and lows[i + offset] <= cur_low:
                    is_low = False
                    break
            if is_low:
                pivots.append(
                    ExternalSwingPoint(
                        bar_idx=i,
                        timestamp=timestamps[i],
                        price=float(cur_low),
                        swing_type=SwingPointType.LOW,
                    )
                )

        pivots.sort(key=lambda p: p.bar_idx)
        return pivots

    def analyze_external_structure(self, df_m15: pd.DataFrame) -> Tuple[BiasDirection, Optional[ExternalBoxRange]]:
        """
        Applies The Box Method across the 15-Minute timeframe:
        1. Classifies external structural high and low.
        2. Differentiates body close breaks from wick sweeps.
        3. Sets directional bias and macro external liquidity targets.
        """
        pivots = self.find_fractal_pivots(df_m15)
        if len(pivots) < 2:
            return self.bias, self.current_box

        high_pivots = [p for p in pivots if p.swing_type == SwingPointType.HIGH]
        low_pivots = [p for p in pivots if p.swing_type == SwingPointType.LOW]

        if not high_pivots or not low_pivots:
            return self.bias, self.current_box

        # Step bar-by-bar to detect valid body breaks vs wick sweeps of external extremes
        closes = df_m15["close"].values
        highs = df_m15["high"].values
        lows = df_m15["low"].values
        timestamps = df_m15.index.values

        active_high: Optional[ExternalSwingPoint] = None
        active_low: Optional[ExternalSwingPoint] = None
        current_bias: BiasDirection = BiasDirection.NEUTRAL

        pivot_map = {p.bar_idx: p for p in pivots}

        for i in range(len(df_m15)):
            confirmed_idx = i - self.right_bars
            if confirmed_idx in pivot_map:
                p = pivot_map[confirmed_idx]
                if p.swing_type == SwingPointType.HIGH:
                    # Update active external high if higher or fresh
                    if active_high is None or p.price >= active_high.price or active_high.broken_by_body:
                        active_high = p
                else:
                    if active_low is None or p.price <= active_low.price or active_low.broken_by_body:
                        active_low = p

            c_close = closes[i]
            c_high = highs[i]
            c_low = lows[i]

            # 1. Test External High break / sweep
            if active_high and not active_high.broken_by_body:
                if c_high > active_high.price:
                    if c_close > active_high.price:
                        # VALID BODY CLOSE: Bullish Structural Break
                        active_high.broken_by_body = True
                        active_high.broken_bar_idx = i
                        active_high.broken_timestamp = timestamps[i]
                        current_bias = BiasDirection.BULLISH
                    else:
                        # WICK SWEEP: Liquidity purged, not a structural break
                        active_high.swept = True

            # 2. Test External Low break / sweep
            if active_low and not active_low.broken_by_body:
                if c_low < active_low.price:
                    if c_close < active_low.price:
                        # VALID BODY CLOSE: Bearish Structural Break
                        active_low.broken_by_body = True
                        active_low.broken_bar_idx = i
                        active_low.broken_timestamp = timestamps[i]
                        current_bias = BiasDirection.BEARISH
                    else:
                        # WICK SWEEP
                        active_low.swept = True

        self.bias = current_bias

        # Form The Box Range using the most recent external high and low
        latest_high = high_pivots[-1]
        latest_low = low_pivots[-1]

        if current_bias == BiasDirection.BULLISH:
            target = latest_high.price
            invalidation = latest_low.price
        elif current_bias == BiasDirection.BEARISH:
            target = latest_low.price
            invalidation = latest_high.price
        else:
            # Baseline from pivot relationship
            if latest_high.price > high_pivots[0].price:
                target = latest_high.price
                invalidation = latest_low.price
                current_bias = BiasDirection.BULLISH
            else:
                target = latest_low.price
                invalidation = latest_high.price
                current_bias = BiasDirection.BEARISH

        self.bias = current_bias
        self.current_box = ExternalBoxRange(
            high_point=latest_high,
            low_point=latest_low,
            bias=current_bias,
            target_price=target,
            invalidation_price=invalidation,
        )

        return self.bias, self.current_box
