"""
RSI Setup (Swing — Daily)
Source: Thomas Bulkowski — adapted to Daily timeframe.

Setup  : RSI(14) drops below 30 (oversold), then crosses back above 30.
Entry  : On the crossover bar.
TP     : Entry + 2 × ATR (or until RSI ≥ 70).
         NOTE: RSI-based TP is not enforceable as a bracket price level,
         so we set a price-based TP using ATR and rely on the broker bracket.
SL     : Entry − 2 × ATR.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from lumibot.entities import Asset
from .swing_base import SwingStrategyBase


class RSISetupSwing(SwingStrategyBase):

    parameters = {
        "Ticker":          Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RiskPct":         0.02,
        "MAX_PYRAMIDS":    3,
        "TOTAL_RISK_PCT":  0.06,
        "ATR_Multiplier":  2.0,
        "ATR_Length":      14,
        "Lookback":        60,
        "RsiLength":       14,
        "RsiOversold":     30,
        "TP_ATR_Multiple": 2.0,
        "Plot": False,
    }

    def get_strategy_name(self) -> str:
        return "RSISetupSwing"

    def get_setup_signal(
        self, df: pd.DataFrame, atr: float, current_price: float
    ) -> Optional[Dict[str, Any]]:
        rsi_len    = int(self.parameters.get("RsiLength", 14))
        oversold   = float(self.parameters.get("RsiOversold", 30))
        tp_mult    = float(self.parameters.get("TP_ATR_Multiple", 2.0))
        rsi_col    = f"RSI_{rsi_len}"

        if rsi_col not in df.columns:
            df.ta.rsi(length=rsi_len, append=True)
        if rsi_col not in df.columns or len(df) < rsi_len + 2:
            return None

        rsi_prev    = float(df[rsi_col].iloc[-2])
        rsi_current = float(df[rsi_col].iloc[-1])

        # Cross above oversold level
        if not (rsi_prev < oversold <= rsi_current):
            return None

        stop_loss   = current_price - (atr * self._atr_mult)
        take_profit = current_price + (atr * tp_mult)

        return {
            "side":        "buy",
            "take_profit": take_profit,
            "stop_loss":   stop_loss,
            "setup_tag":   f"RSI_SETUP_{self.get_datetime().date()}",
        }
