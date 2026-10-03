"""
HCR Breakout Long (Swing)
Source: Thomas Bulkowski — adapted to Daily timeframe.

Setup  : 15-day Horizontal Consolidation Region (range ≤ 5 %).
Entry  : Price breaks above HCR high by 0.2 % (avoids false breaks).
TP     : Entry + 2 × ATR (or nearest swing high above).
SL     : Entry − 1.5 × ATR (below HCR low acts as structural floor).
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from lumibot.entities import Asset
from .swing_base import SwingStrategyBase


class HCRBreakoutLongSwing(SwingStrategyBase):

    parameters = {
        "Ticker":           Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RiskPct":          0.02,
        "MAX_PYRAMIDS":     3,
        "TOTAL_RISK_PCT":   0.06,
        "ATR_Multiplier":   1.5,
        "ATR_Length":       14,
        "Lookback":         40,
        "ConsolidationDays": 15,
        "MaxRangePct":      0.05,   # 5 % max range over consolidation period
        "BreakoutBuffer":   0.002,  # 0.2 % above HCR high before entry
        "TP_ATR_Multiple":  2.0,    # Take profit = entry + N × ATR
        "Plot": False,
    }

    def get_strategy_name(self) -> str:
        return "HCRBreakoutLongSwing"

    def get_setup_signal(
        self, df: pd.DataFrame, atr: float, current_price: float
    ) -> Optional[Dict[str, Any]]:
        n = int(self.parameters.get("ConsolidationDays", 15))
        max_range = float(self.parameters.get("MaxRangePct", 0.05))
        buffer = float(self.parameters.get("BreakoutBuffer", 0.002))
        tp_mult = float(self.parameters.get("TP_ATR_Multiple", 2.0))

        if len(df) < n + 2:
            return None

        period = df.iloc[-(n + 1):-1]   # exclude current bar
        highest = period["high"].max()
        lowest  = period["low"].min()
        range_pct = (highest - lowest) / lowest

        if range_pct > max_range:
            return None  # Too volatile — not an HCR

        entry_trigger = highest * (1 + buffer)
        if current_price < entry_trigger:
            return None

        stop_loss   = current_price - (atr * self._atr_mult)
        take_profit = current_price + (atr * tp_mult)

        return {
            "side":        "buy",
            "take_profit": take_profit,
            "stop_loss":   stop_loss,
            "setup_tag":   f"HCR_LONG_{round(highest, 2)}",
        }
