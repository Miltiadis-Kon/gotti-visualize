"""
Alpaca SMC Execution Bridge & Order Coordinates Adapter (alpaca_smc_bridge.py)
==============================================================================
Maps Smart Money Concepts (SMC) mechanical trading rules cleanly to Alpaca's API
execution capabilities using the open-source `smartmoneyconcepts` library:

1. Time and Price (Killzones):
   - London Window: 02:00 AM - 05:00 AM EST (07:00 - 10:00 UTC)
   - New York Window: 07:00 AM - 10:00 AM EST (12:00 - 15:00 UTC, extensible to 11:30 EST)
   - Enforces strict execution loop gating.

2. Limit Order Execution:
   - Anchors entry limit orders directly to the proximate boundary of detected
     Fair Value Gaps (FVG) or Order Blocks (OB) from `smartmoneyconcepts`.

3. Strict Invalidation (Stop Loss Placement & Bracket Orders):
   - Places Stop Loss 1-2 ticks/cents beyond the structural extreme that triggered CHoCH.
   - Formulates native Alpaca Bracket Orders (Limit Entry + Stop Loss + Take Profit).

4. Partial Profit Scaling:
   - Manages active positions by fractional share size or exact lot quantities.
   - Triggers partial close orders (e.g. 40% TP1, 40% TP2, 20% runner or 40% / 60%)
     and trails stop loss to Breakeven upon TP1 fill.

Open-source reference: https://pypi.org/project/smartmoneyconcepts/
"""

from __future__ import annotations

import os
import sys
import io
import math
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Any
import pytz
import pandas as pd
import numpy as np

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class AlpacaSMCBridge:
    """
    Translates Smart Money Concepts rules and `smartmoneyconcepts` library signals
    into institutional Alpaca order requests, bracket orders, and tranche executions.
    """

    def __init__(
        self,
        allow_ny_extension: bool = True,
        tick_size: float = 0.01,
        default_tp1_pct: float = 0.40,
        default_tp2_pct: float = 0.40,
        default_tp3_pct: float = 0.20,
    ):
        self.tz_est = pytz.timezone("America/New_York")
        self.allow_ny_extension = allow_ny_extension
        self.tick_size = tick_size
        self.default_tp1_pct = default_tp1_pct
        self.default_tp2_pct = default_tp2_pct
        self.default_tp3_pct = default_tp3_pct

    # =========================================================================
    # 1. TIME AND PRICE (KILLZONE GATING)
    # =========================================================================
    def is_in_killzone(self, current_dt: Optional[Any] = None) -> Tuple[bool, str]:
        """
        Validates if current clock falls strictly within London or New York Killzone.
        - London Window: 02:00 AM - 05:00 AM EST
        - New York Window: 07:00 AM - 10:00 AM EST (or 11:30 AM EST for US Equities)
        """
        if current_dt is None:
            current_dt = datetime.now(pytz.utc)
        elif isinstance(current_dt, pd.Timestamp):
            current_dt = current_dt.to_pydatetime()

        if current_dt.tzinfo is None:
            current_dt = pytz.utc.localize(current_dt)

        est_time = current_dt.astimezone(self.tz_est)
        t = est_time.time()

        # London Window: 02:00 - 05:00 EST
        if time(2, 0) <= t < time(5, 0):
            return True, "LONDON_KILLZONE"

        # New York Window: 07:00 - 10:00 EST (or 11:30 EST)
        ny_end = time(11, 30) if self.allow_ny_extension else time(10, 0)
        if time(7, 0) <= t < ny_end:
            return True, "NEW_YORK_KILLZONE"

        return False, "OUTSIDE_KILLZONE"

    # =========================================================================
    # 2. HISTORICAL PRICE FEED TO SMARTMONEYCONCEPTS LIBRARY
    # =========================================================================
    def analyze_smc_coordinates(
        self,
        df_ohlcv: pd.DataFrame,
        bias: str,
        swing_length: int = 3
    ) -> Dict[str, Any]:
        """
        Feeds historical candlestick data into the open-source `smartmoneyconcepts`
        library and returns the exact active FVG, Order Block, and swing coordinates.
        """
        if df_ohlcv.empty or len(df_ohlcv) < 5:
            return {"status": "INSUFFICIENT_DATA"}

        ohlc = df_ohlcv[["open", "high", "low", "close", "volume"]].copy() if "volume" in df_ohlcv.columns else df_ohlcv[["open", "high", "low", "close"]].copy()
        if "volume" not in ohlc.columns:
            ohlc["volume"] = 100_000

        try:
            from smartmoneyconcepts import smc

            # 1. Detect Swings
            swings = smc.swing_highs_lows(ohlc, swing_length=swing_length)

            # 2. Detect FVGs
            fvgs = smc.fvg(ohlc, join_consecutive=False)

            # 3. Detect Order Blocks
            obs = smc.ob(ohlc, swings)

            # 4. Detect BOS / CHoCH
            bos_choch = smc.bos_choch(ohlc, swings)

            # Filter for unmitigated POIs
            unmitigated_fvgs = []
            for i in range(len(fvgs)):
                f_val = fvgs["FVG"].iloc[i]
                if pd.isna(f_val):
                    continue
                mit_idx = fvgs["MitigatedIndex"].iloc[i] if "MitigatedIndex" in fvgs.columns else 0.0
                if mit_idx == 0.0:  # Unmitigated
                    unmitigated_fvgs.append({
                        "bar_idx": i,
                        "type": "BULLISH" if f_val == 1.0 else "BEARISH",
                        "top": float(fvgs["Top"].iloc[i]),
                        "bottom": float(fvgs["Bottom"].iloc[i]),
                        "timestamp": str(df_ohlcv.index[i])
                    })

            unmitigated_obs = []
            for i in range(len(obs)):
                ob_val = obs["OB"].iloc[i]
                if pd.isna(ob_val):
                    continue
                mit_idx = obs["MitigatedIndex"].iloc[i] if "MitigatedIndex" in obs.columns else 0.0
                if mit_idx == 0.0:  # Unmitigated
                    unmitigated_obs.append({
                        "bar_idx": i,
                        "type": "BULLISH" if ob_val == 1.0 else "BEARISH",
                        "top": float(obs["Top"].iloc[i]),
                        "bottom": float(obs["Bottom"].iloc[i]),
                        "percentage": float(obs["Percentage"].iloc[i]) if "Percentage" in obs.columns else 50.0,
                        "timestamp": str(df_ohlcv.index[i])
                    })

            return {
                "status": "OK",
                "bias": bias,
                "fvgs": unmitigated_fvgs,
                "obs": unmitigated_obs,
                "swings": swings.dropna().to_dict(orient="index"),
                "recent_close": float(df_ohlcv["close"].iloc[-1])
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    # =========================================================================
    # 3. ORDER BUILDERS (LIMIT & BRACKET ORDERS)
    # =========================================================================
    def build_limit_order_payload(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Constructs an Alpaca API Limit Order payload anchored directly
        to the proximate boundary of the SMC POI.
        """
        payload = {
            "symbol": symbol.upper(),
            "qty": int(qty),
            "side": side.lower(),  # "buy" or "sell"
            "type": "limit",
            "time_in_force": "day",
            "limit_price": round(float(limit_price), 2),
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        return payload

    def build_bracket_order_payload(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_limit_price: float,
        structural_extreme: float,
        take_profit_target: float,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Constructs an Alpaca Advanced Bracket Order payload:
        - Limit Entry: Proximate boundary of FVG / OB
        - Structural Stop Loss: 1-2 ticks beyond structural extreme (never arbitrary)
        - Take Profit: External liquidity pool / macro swing target
        """
        entry_p = round(float(entry_limit_price), 2)
        side_clean = side.lower()

        # Structural Invalidation: 1 tick beyond extreme
        if side_clean == "buy":
            stop_p = round(float(structural_extreme - self.tick_size), 2)
            tp_p = round(float(take_profit_target), 2)
            assert stop_p < entry_p < tp_p, f"Invalid BUY bracket coordinates: SL={stop_p}, Entry={entry_p}, TP={tp_p}"
        else:
            stop_p = round(float(structural_extreme + self.tick_size), 2)
            tp_p = round(float(take_profit_target), 2)
            assert stop_p > entry_p > tp_p, f"Invalid SELL bracket coordinates: SL={stop_p}, Entry={entry_p}, TP={tp_p}"

        payload = {
            "symbol": symbol.upper(),
            "qty": int(qty),
            "side": side_clean,
            "type": "limit",
            "time_in_force": "gtc",
            "limit_price": entry_p,
            "order_class": "bracket",
            "take_profit": {
                "limit_price": tp_p
            },
            "stop_loss": {
                "stop_price": stop_p,
                "limit_price": stop_p  # optional stop-limit or market stop
            }
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id

        return payload

    # =========================================================================
    # 4. PARTIAL PROFIT SCALING (TRANCHE EXECUTION)
    # =========================================================================
    def calculate_tranche_quantities(
        self,
        total_quantity: int,
        model: str = "HYBRID_40_60"  # or "THREE_STAGE_40_40_20"
    ) -> Dict[str, int]:
        """
        Calculates tranche share lot sizes based on the strategy scaling model.
        """
        if model == "THREE_STAGE_40_40_20":
            tp1_qty = int(math.floor(total_quantity * self.default_tp1_pct))
            tp2_qty = int(math.floor(total_quantity * self.default_tp2_pct))
            tp3_qty = total_quantity - (tp1_qty + tp2_qty)
            return {"TP1": tp1_qty, "TP2": tp2_qty, "RUNNER": tp3_qty}
        else:
            # Hybrid 40% TP1 / 60% Runner
            tp1_qty = int(math.floor(total_quantity * 0.40))
            runner_qty = total_quantity - tp1_qty
            return {"TP1": tp1_qty, "RUNNER": runner_qty}

    def build_partial_scale_close_payload(
        self,
        symbol: str,
        side: str,  # Position side: "buy" (long) or "sell" (short)
        scale_qty: int,
        reason: str = "TP1_SCALING"
    ) -> Dict[str, Any]:
        """
        Builds the Alpaca API market order to scale out a specific tranche quantity
        upon reaching a predefined liquidity pool threshold.
        """
        close_side = "sell" if side.lower() == "buy" else "buy"
        return {
            "symbol": symbol.upper(),
            "qty": int(scale_qty),
            "side": close_side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"scale_{symbol}_{reason}_{int(datetime.now().timestamp())}"
        }

    def build_breakeven_stop_update_payload(
        self,
        existing_stop_order_id: str,
        entry_price: float,
        remaining_qty: int
    ) -> Dict[str, Any]:
        """
        Builds payload to replace the active Stop Loss with a Breakeven Stop
        at the original Entry Price once TP1 has been secured.
        """
        return {
            "order_id": existing_stop_order_id,
            "qty": int(remaining_qty),
            "stop_price": round(float(entry_price), 2),
            "time_in_force": "gtc"
        }
