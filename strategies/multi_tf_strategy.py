"""
Multi-Timeframe Key Levels Strategy (Modernized)
================================================
Combines multi-resolution Support/Resistance key levels (1D, 4H, 15m) with
Fibonacci retracement setups on 5-minute RTH intraday execution.

Architecture:
  - Subclasses StrategyBaseplate for unified ATR risk framework, order tracking, and plotting.
  - Generates multi-timeframe S/R levels and Fibonacci setups entirely in-memory from
    historical 5-minute candles via pandas resampling (no external network latency).
  - Priority logic: Fibonacci 0.618 retracement setups prioritized over S/R bounces.
  - S/R bounce entries validated for minimum Risk-to-Reward (>= 1.5).
  - ATR-buffered bracket orders manage trade lifecycle automatically.
  - Exports standardized trade records for unified dashboard and interactive Plotly charting.
"""

from __future__ import annotations

import os
import sys
from math import floor
import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lumibot.entities import Asset
from lumibot.backtesting import PandasDataBacktesting

try:
    from strategies.strat_baseplate import StrategyBaseplate
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from strat_baseplate import StrategyBaseplate
from key_levels.key_levels import find_key_levels, merge_key_levels
from key_levels.fibonacci_levels import get_fibonacci_trade_setups, find_fibonacci_levels


class MultiTimeframeKeyLevelsStrategy(StrategyBaseplate):
    """
    Multi-Timeframe Key Levels & Fibonacci Strategy.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "TradingStyle": "day_trading",
        "RiskPct": 0.02,                      # 2% portfolio risk budget per trade
        "MinRiskReward": 1.5,                 # Minimum 1.5:1 Risk/Reward
        "MinImportance": 2,                   # Minimum S/R importance threshold
        "EntryTolerancePct": 0.004,           # 0.4% price tolerance around key levels
        "FibSRProximity": 0.02,               # 2% proximity threshold for noise reduction
        "ATR_Length": 14,                     # ATR lookback period
        "ATR_Stop_Buffer_Multiplier": 0.5,    # ATR buffer added to stop loss
        "TickSize": 0.01,                     # Minimum tick offset
        "Plot": True,
    }

    # Class-level cache for multi-timeframe analysis across backtest iterations
    _analysis_cache: Dict[str, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"  # 5-minute candles

        # State tracking
        self.state: str = "IDLE"  # "IDLE", "WAITING_FOR_FILL", "IN_POSITION"
        self.active_order = None
        self.pending_side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trade_setup_type: Optional[str] = None

        # Level data
        self.merged_levels: pd.DataFrame = pd.DataFrame()
        self.support_levels: pd.DataFrame = pd.DataFrame()
        self.resistance_levels: pd.DataFrame = pd.DataFrame()
        self.fib_trade_setups: pd.DataFrame = pd.DataFrame()
        self.last_analysis_date: Optional[datetime.date] = None

        # Standardized trade records
        self.trade_records: List[Dict[str, Any]] = []
        self.current_trade: Optional[Dict[str, Any]] = None

        ticker_sym = self.parameters["Ticker"].symbol if hasattr(self.parameters["Ticker"], "symbol") else str(self.parameters["Ticker"])
        self.log_message(f"[{ticker_sym}] MultiTimeframeKeyLevelsStrategy initialized. Min R:R={self.parameters['MinRiskReward']}.")

    # ──────────────────────────────────────────────────────────────────────────
    # Indicators & In-Memory Multi-Timeframe Resampling
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_intraday_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Volume MA, ATR(14), and 09:30-anchored Session VWAP."""
        data = df.copy()

        # Volume MA
        data["vol_ma"] = data["volume"].rolling(20).mean()

        # ATR(14)
        atr_len = int(self.parameters.get("ATR_Length", 14))
        data.ta.atr(length=atr_len, append=True)
        atr_cols = [c for c in data.columns if c.startswith("ATRr_")]
        data["atr"] = data[atr_cols[0]] if atr_cols else 1.0

        # Session VWAP anchored to 09:30 EST
        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        data["pv"] = data["typical_price"] * data["volume"]

        if isinstance(data.index, pd.DatetimeIndex):
            idx_ny = data.index.tz_convert("America/New_York") if data.index.tz else data.index.tz_localize("UTC").tz_convert("America/New_York")
            session_dates = idx_ny.date
            times = idx_ny.time
        else:
            session_dates = np.zeros(len(data))
            times = [datetime.time(9, 30)] * len(data)

        data["session_date"] = session_dates
        rth_pv = np.where(times >= datetime.time(9, 30), data["pv"], 0.0)
        rth_vol = np.where(times >= datetime.time(9, 30), data["volume"], 0.0)
        data["cum_pv"] = pd.Series(rth_pv, index=data.index).groupby(data["session_date"]).cumsum()
        data["cum_vol"] = pd.Series(rth_vol, index=data.index).groupby(data["session_date"]).cumsum()
        data["vwap"] = data["cum_pv"] / data["cum_vol"].replace(0, np.nan)
        data["vwap"] = data["vwap"].ffill().bfill()

        return data

    def _refresh_multi_tf_analysis(self, df_5m: pd.DataFrame, current_date: datetime.date, current_price: float):
        """
        Fast in-memory multi-timeframe level generator:
        Resamples historical 5m bars up to current date into 1D, 4H, and 15m,
        then detects merged S/R key levels and Fibonacci setups without network calls.
        """
        ticker_sym = self.parameters["Ticker"].symbol if hasattr(self.parameters["Ticker"], "symbol") else str(self.parameters["Ticker"])
        cache_key = f"{ticker_sym}_{current_date}"

        if cache_key in MultiTimeframeKeyLevelsStrategy._analysis_cache:
            cached = MultiTimeframeKeyLevelsStrategy._analysis_cache[cache_key]
            self.merged_levels = cached["merged"]
            self.support_levels = cached["support"]
            self.resistance_levels = cached["resistance"]
            self.fib_trade_setups = cached["fib"]
            self.last_analysis_date = current_date
            return

        # Use historical bars strictly prior to or including current date
        history = df_5m.copy()
        if history.empty or len(history) < 60:
            return

        # 1. Resample to 1D, 4H, and 15m
        try:
            df_1d = history.resample("1D").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            df_4h = history.resample("4h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            df_15m = history.resample("15min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()

            # 2. Key Levels Detection across resolutions
            levels_list = []
            if len(df_1d) >= 15:
                lvl_1d = find_key_levels(df_1d, resolution="1D")
                if not lvl_1d.empty:
                    levels_list.append(lvl_1d)

            if len(df_4h) >= 20:
                lvl_4h = find_key_levels(df_4h, resolution="4H")
                if not lvl_4h.empty:
                    levels_list.append(lvl_4h)

            if len(df_15m) >= 30:
                lvl_15m = find_key_levels(df_15m, resolution="15m")
                if not lvl_15m.empty:
                    levels_list.append(lvl_15m)

            if levels_list:
                all_raw_levels = pd.concat(levels_list, ignore_index=True)
                merged = merge_key_levels(all_raw_levels, price_threshold=0.5)
            else:
                merged = pd.DataFrame(columns=["level_price", "type", "touch_count", "importance"])

            # Filter by minimum importance
            min_imp = int(self.parameters.get("MinImportance", 2))
            if not merged.empty:
                valid_levels = merged[merged["importance"] >= min_imp]
                self.support_levels = valid_levels[valid_levels["level_price"] < current_price].sort_values("level_price", ascending=False).reset_index(drop=True)
                self.resistance_levels = valid_levels[valid_levels["level_price"] > current_price].sort_values("level_price", ascending=True).reset_index(drop=True)
            else:
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()

            self.merged_levels = merged

            # 3. Fibonacci Setups across resolutions
            fib_list = []
            if len(df_1d) >= 15:
                f_1d = get_fibonacci_trade_setups(df_1d, resolution="1D")
                if not f_1d.empty:
                    fib_list.append(f_1d)

            if len(df_4h) >= 20:
                f_4h = get_fibonacci_trade_setups(df_4h, resolution="4H")
                if not f_4h.empty:
                    fib_list.append(f_4h)

            if len(df_15m) >= 30:
                f_15m = get_fibonacci_trade_setups(df_15m, resolution="15m")
                if not f_15m.empty:
                    fib_list.append(f_15m)

            if fib_list:
                self.fib_trade_setups = pd.concat(fib_list, ignore_index=True)
            else:
                self.fib_trade_setups = pd.DataFrame()

            # Store in cache
            MultiTimeframeKeyLevelsStrategy._analysis_cache[cache_key] = {
                "merged": self.merged_levels,
                "support": self.support_levels,
                "resistance": self.resistance_levels,
                "fib": self.fib_trade_setups,
            }
            self.last_analysis_date = current_date

        except Exception as e:
            self.log_message(f"[{ticker_sym}] Multi-TF level generation error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Trading Iteration
    # ──────────────────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        ticker_str = symbol.symbol if hasattr(symbol, "symbol") else str(symbol)

        # 1. Historical Prices (5-Minute candles)
        bars = self.get_historical_prices(symbol, 1500, "5M")
        if bars is None or len(bars.df) < 50:
            return

        df = self._compute_intraday_indicators(bars.df)
        current_dt = self.get_datetime()
        current_time = current_dt.time()
        current_price = self.get_last_price(symbol)
        if current_price is None:
            return

        last_candle = df.iloc[-1]
        current_atr = float(last_candle["atr"])

        # 2. Time Gate: Only trade during Regular Market Hours (09:45 to 15:45 EST)
        if current_time < datetime.time(9, 45) or current_time > datetime.time(15, 45):
            if self.state == "WAITING_FOR_FILL":
                self._reset_to_idle()
            return

        # 3. Refresh Multi-Timeframe Levels (Once per day or when cache is empty)
        today = current_dt.date()
        if self.last_analysis_date != today or self.merged_levels.empty:
            self._refresh_multi_tf_analysis(df, today, current_price)

        # 4. Check Open Position
        pos = self.get_position(symbol)
        if pos is not None and pos.quantity != 0:
            return

        if self.state != "IDLE":
            return

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL EVALUATION
        # ──────────────────────────────────────────────────────────────────────
        entry_tol = float(self.parameters.get("EntryTolerancePct", 0.004))
        min_rr = float(self.parameters.get("MinRiskReward", 1.5))
        atr_buf = current_atr * float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 0.5))

        chosen_signal = None

        # PRIORITY 1: Fibonacci Trade Setups (0.618 Retracement)
        if not self.fib_trade_setups.empty:
            for _, setup in self.fib_trade_setups.iterrows():
                fib_entry = float(setup["entry_price"])
                tol = fib_entry * entry_tol

                if abs(current_price - fib_entry) <= tol:
                    trend = str(setup["trend"]).lower()
                    side = "buy" if trend == "uptrend" else "sell"
                    raw_tp = float(setup["take_profit"])
                    raw_sl = float(setup["stop_loss"])

                    # Apply ATR buffer to Stop Loss
                    if side == "buy":
                        sl = round(raw_sl - atr_buf, 2)
                        tp = round(raw_tp, 2)
                        risk = current_price - sl
                        reward = tp - current_price
                    else:
                        sl = round(raw_sl + atr_buf, 2)
                        tp = round(raw_tp, 2)
                        risk = sl - current_price
                        reward = current_price - tp

                    if risk > 0 and (reward / risk) >= min_rr:
                        chosen_signal = {
                            "side": side,
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reason": f"FIB_{trend.upper()}_0.618 ({setup.get('resolution', 'TF')})",
                        }
                        break

        # PRIORITY 2: Support & Resistance Bounces (Secondary Signal)
        if chosen_signal is None:
            # Check Support Bounce (Long)
            if not self.support_levels.empty:
                nearest_sup = float(self.support_levels.iloc[0]["level_price"])
                if abs(current_price - nearest_sup) <= (nearest_sup * entry_tol):
                    # Target nearest resistance or 2x ATR
                    if not self.resistance_levels.empty:
                        target_tp = float(self.resistance_levels.iloc[0]["level_price"])
                    else:
                        target_tp = current_price + (2.5 * current_atr)

                    sl = round(nearest_sup - atr_buf, 2)
                    tp = round(target_tp, 2)
                    risk = current_price - sl
                    reward = tp - current_price

                    if risk > 0 and (reward / risk) >= min_rr:
                        # Noise reduction: check if near inactive Fib zone
                        is_near_fib = False
                        if not self.fib_trade_setups.empty:
                            for _, fb in self.fib_trade_setups.iterrows():
                                if abs(current_price - float(fb["entry_price"])) / current_price < float(self.parameters["FibSRProximity"]):
                                    is_near_fib = True
                                    break
                        if not is_near_fib:
                            chosen_signal = {
                                "side": "buy",
                                "entry": current_price,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "reason": f"SR_SUPPORT_BOUNCE (${nearest_sup:.2f})",
                            }

            # Check Resistance Rejection (Short)
            if chosen_signal is None and not self.resistance_levels.empty:
                nearest_res = float(self.resistance_levels.iloc[0]["level_price"])
                if abs(current_price - nearest_res) <= (nearest_res * entry_tol):
                    if not self.support_levels.empty:
                        target_tp = float(self.support_levels.iloc[0]["level_price"])
                    else:
                        target_tp = current_price - (2.5 * current_atr)

                    sl = round(nearest_res + atr_buf, 2)
                    tp = round(target_tp, 2)
                    risk = sl - current_price
                    reward = current_price - tp

                    if risk > 0 and (reward / risk) >= min_rr:
                        chosen_signal = {
                            "side": "sell",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reason": f"SR_RESISTANCE_REJECT (${nearest_res:.2f})",
                        }

        # ──────────────────────────────────────────────────────────────────────
        # ORDER EXECUTION (BRACKET ORDER)
        # ──────────────────────────────────────────────────────────────────────
        if chosen_signal:
            self._arm_trade_execution(
                side=chosen_signal["side"],
                entry=chosen_signal["entry"],
                stop_loss=chosen_signal["stop_loss"],
                take_profit=chosen_signal["take_profit"],
                reason=chosen_signal["reason"],
            )

    def _arm_trade_execution(self, side: str, entry: float, stop_loss: float, take_profit: float, reason: str):
        """Size position based on 2% risk budget and submit bracket limit order."""
        symbol = self.parameters["Ticker"]
        ticker_str = symbol.symbol if hasattr(symbol, "symbol") else str(symbol)

        risk_per_share = abs(entry - stop_loss)
        if risk_per_share <= 0:
            return

        portfolio_val = self.get_portfolio_value()
        risk_budget = portfolio_val * float(self.parameters.get("RiskPct", 0.02))
        qty = floor(risk_budget / risk_per_share)

        max_affordable = floor((portfolio_val * 0.95) / max(0.01, entry))
        qty = min(qty, max_affordable)

        if qty < 1:
            return

        order = self.create_order(
            asset=symbol,
            quantity=qty,
            side=side,
            limit_price=round(entry, 2),
            order_class="bracket",
            secondary_stop_price=round(stop_loss, 2),
            secondary_limit_price=round(take_profit, 2),
            time_in_force="gtc"
        )
        self.submit_order(order)
        self.active_order = order

        self.pending_side = side
        self.entry_price = round(entry, 2)
        self.stop_loss = round(stop_loss, 2)
        self.take_profit = round(take_profit, 2)
        self.trade_setup_type = reason
        self.state = "WAITING_FOR_FILL"

        self.log_message(
            f"[{ticker_str}] SUBMITTED {side.upper()} BRACKET ORDER ({reason}): "
            f"Qty={qty} @ ${entry:.2f} | SL=${stop_loss:.2f} | TP=${take_profit:.2f}"
        )

    def _reset_to_idle(self):
        """Purge pending state and cancel working entry orders."""
        if self.active_order is not None and self.state == "WAITING_FOR_FILL":
            try:
                self.cancel_order(self.active_order)
            except Exception:
                pass
        self.state = "IDLE"
        self.active_order = None
        self.pending_side = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_setup_type = None

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Handle execution fills and maintain trade records."""
        ticker_str = self.parameters["Ticker"].symbol if hasattr(self.parameters["Ticker"], "symbol") else str(self.parameters["Ticker"])

        if self.state == "WAITING_FOR_FILL":
            self.state = "IN_POSITION"
            self.current_trade = {
                "ticker": ticker_str,
                "entry_time": str(self.get_datetime()),
                "side": self.pending_side,
                "entry_price": float(price),
                "quantity": float(quantity),
                "stop_loss": float(self.stop_loss) if self.stop_loss else None,
                "take_profit": float(self.take_profit) if self.take_profit else None,
                "reason": self.trade_setup_type,
            }
            self.log_message(f"[{ticker_str}] TRADE FILLED: In position {quantity} @ ${price:.2f}.")

        elif self.state == "IN_POSITION":
            if position is None or position.quantity == 0:
                if self.current_trade:
                    self.current_trade["exit_time"] = str(self.get_datetime())
                    self.current_trade["exit_price"] = float(price)
                    entry_p = self.current_trade["entry_price"]
                    qty = self.current_trade["quantity"]
                    side = self.current_trade["side"]
                    pnl = (price - entry_p) * qty if side == "buy" else (entry_p - price) * qty
                    self.current_trade["pnl"] = round(pnl, 2)
                    self.current_trade["pnl_pct"] = round(
                        ((price - entry_p) / entry_p) * 100 if side == "buy" else ((entry_p - price) / entry_p) * 100, 2
                    )
                    self.trade_records.append(self.current_trade)
                    self.log_message(f"[{ticker_str}] POSITION CLOSED @ ${price:.2f} | PnL: ${pnl:.2f} ({self.current_trade['pnl_pct']}%)")
                    self.current_trade = None
                self._reset_to_idle()
