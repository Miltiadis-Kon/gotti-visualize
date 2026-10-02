"""
Gap Setup Long (Swing)
Source: Thomas Bulkowski — adapted to Daily timeframe.

Setup  : Daily gap down ≥ 2 % (today open < yesterday close).
Entry  : Price pushes back above today's open (gap fade confirmed).
TP     : Yesterday's close (gap fill target).
SL     : today_open − 0.5 × ATR (tight intraday stop below gap open).
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from lumibot.entities import Asset
from .swing_base import SwingStrategyBase


class GapSetupLongSwing(SwingStrategyBase):

    parameters = {
        "Ticker":         Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RiskPct":        0.02,
        "MAX_PYRAMIDS":   3,
        "TOTAL_RISK_PCT": 0.06,
        "ATR_Multiplier": 1.5,
        "ATR_Length":     14,
        "Lookback":       40,
        "GapPct":         0.02,    # Minimum gap-down magnitude (2 %)
        "SL_ATR_Fraction": 0.5,   # SL = today_open − fraction × ATR
        "Plot": False,
    }

    def get_strategy_name(self) -> str:
        return "GapSetupLongSwing"

    def get_setup_signal(
        self, df: pd.DataFrame, atr: float, current_price: float
    ) -> Optional[Dict[str, Any]]:
        if len(df) < 3:
            return None

        min_gap = float(self.parameters.get("GapPct", 0.02))
        sl_frac = float(self.parameters.get("SL_ATR_Fraction", 0.5))

        yest_close = float(df["close"].iloc[-2])
        today_open = float(df["open"].iloc[-1])

        gap = (today_open - yest_close) / yest_close  # negative = gap down
        if gap > -min_gap:
            return None  # Not a meaningful gap-down

        # Confirm fade: price has moved back above today's open
        if current_price <= today_open:
            return None

        take_profit = yest_close                         # gap fill
        stop_loss   = today_open - (atr * sl_frac)

        if take_profit <= current_price:
            return None  # Gap already filled

        return {
            "side":        "buy",
            "take_profit": take_profit,
            "stop_loss":   stop_loss,
            "setup_tag":   f"GAP_LONG_{self.get_datetime().date()}",
        }
