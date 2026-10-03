"""
Fibonacci Retrace Swing (Daily)
Source: Thomas Bulkowski — adapted to Daily timeframe.

Setup  : Identify a 10-day swing move (highest - lowest ≥ 3 %).
         Price retraces into the 38.2 %–61.8 % "golden pocket".
Entry  : Today's candle closes green while inside the fib zone (bullish reversal).
TP     : Swing high (0 % retracement).
SL     : Entry − 2 × ATR (below swing low acts as structural floor).
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from lumibot.entities import Asset
from .swing_base import SwingStrategyBase


class FibRetraceSwing(SwingStrategyBase):

    parameters = {
        "Ticker":          Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RiskPct":         0.02,
        "MAX_PYRAMIDS":    3,
        "TOTAL_RISK_PCT":  0.06,
        "ATR_Multiplier":  2.0,
        "ATR_Length":      14,
        "Lookback":        40,
        "TrendDays":       10,    # Swing window for measuring the move
        "MinMovePct":      0.03,  # Minimum move size (3 %) for a valid swing
        "FibLow":          0.382,
        "FibHigh":         0.618,
        "Plot": False,
    }

    def get_strategy_name(self) -> str:
        return "FibRetraceSwing"

    def get_setup_signal(
        self, df: pd.DataFrame, atr: float, current_price: float
    ) -> Optional[Dict[str, Any]]:
        n = int(self.parameters.get("TrendDays", 10))
        min_move = float(self.parameters.get("MinMovePct", 0.03))
        fib_lo = float(self.parameters.get("FibLow", 0.382))
        fib_hi = float(self.parameters.get("FibHigh", 0.618))

        if len(df) < n + 2:
            return None

        period = df.iloc[-(n + 1):-1]
        highest = period["high"].max()
        lowest  = period["low"].min()
        move    = highest - lowest

        # Require a meaningful upswing
        if move / lowest < min_move:
            return None

        # Fibonacci retracement levels (from swing high)
        fib_38 = highest - (fib_lo * move)
        fib_62 = highest - (fib_hi * move)

        # Price must be in the golden pocket
        if not (fib_62 <= current_price <= fib_38):
            return None

        # Require today's close > open (bullish reversal candle)
        last = df.iloc[-1]
        if last["close"] <= last["open"]:
            return None

        take_profit = highest
        stop_loss   = current_price - (atr * self._atr_mult)

        if take_profit <= current_price:
            return None

        return {
            "side":        "buy",
            "take_profit": take_profit,
            "stop_loss":   stop_loss,
            "setup_tag":   f"FIB_RETRACE_{round(highest, 2)}_{round(lowest, 2)}",
        }
