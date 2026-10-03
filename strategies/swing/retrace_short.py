"""
Retrace Short (Swing — Daily)
Source: Thomas Bulkowski — adapted to Daily timeframe.

Setup  : 3 consecutive up days (green daily candles).
Entry  : Today's price breaks below yesterday's low (reversal confirmed).
TP     : Entry − 1.5 × ATR.
SL     : Entry + 1.5 × ATR.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from lumibot.entities import Asset
from .swing_base import SwingStrategyBase


class RetraceShortSwing(SwingStrategyBase):

    parameters = {
        "Ticker":          Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RiskPct":         0.02,
        "MAX_PYRAMIDS":    3,
        "TOTAL_RISK_PCT":  0.06,
        "ATR_Multiplier":  1.5,
        "ATR_Length":      14,
        "Lookback":        40,
        "TP_ATR_Multiple": 1.5,
        "Plot": False,
    }

    def get_strategy_name(self) -> str:
        return "RetraceShortSwing"

    def get_setup_signal(
        self, df: pd.DataFrame, atr: float, current_price: float
    ) -> Optional[Dict[str, Any]]:
        if len(df) < 5:
            return None

        tp_mult = float(self.parameters.get("TP_ATR_Multiple", 1.5))

        c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]

        # 3 consecutive green (up) days
        if not (
            c1["close"] > c1["open"]
            and c2["close"] > c2["open"]
            and c3["close"] > c3["open"]
        ):
            return None

        yesterday_low = float(c3["low"])
        if current_price >= yesterday_low:
            return None  # Breakdown not yet confirmed

        stop_loss   = current_price + (atr * self._atr_mult)
        take_profit = current_price - (atr * tp_mult)

        return {
            "side":        "sell",
            "take_profit": take_profit,
            "stop_loss":   stop_loss,
            "setup_tag":   f"RETRACE_SHORT_{self.get_datetime().date()}",
        }
