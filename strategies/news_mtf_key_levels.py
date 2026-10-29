"""
News + MultiTimeframe Key Levels Strategy (news_mtf_key_levels.py)
==================================================================
Combines Stock Alchemist intelligence news/fundamental signals with
Multi-Timeframe Support/Resistance clustering and 0.618 Fibonacci setups.

Architecture:
1. Signal Gating:
   - Evaluates signals via Dual Hard Floor (Fund Conf >= 70% AND News Conv >= 70%).
   - Establishes directional bias: only allows technical entries in the news-approved direction.
2. Technical Execution:
   - Priority 1: Fibonacci 0.618 Golden Pocket retracement setup in direction of news bias.
   - Priority 2: Multi-Timeframe Key Level bounce (Support for BUY, Resistance for SELL).
3. Risk & Position Management:
   - Sizing: 2% portfolio risk budget ($2,000 per trade on $100k capital).
   - S/R + ATR buffered bracket orders (Stop Loss & Take Profit).
   - Multi-day overnight holding permitted until TP or SL is triggered.
"""

import os
import sys
from math import floor
import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from lumibot.entities import Asset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategies.strat_baseplate import StrategyBaseplate
from strategies.market_reactor import fetch_reactor_signals, ReactorSignal
from strategies.key_levels.key_levels import find_key_levels, merge_key_levels
from strategies.key_levels.fibonacci_levels import get_fibonacci_trade_setups
from strategies.risk_management import optimize_sl_tp, calculate_atr
from strategies.trade_tracker import TradeTracker


class NewsMTFKeyLevelsStrategy(StrategyBaseplate):
    """
    Combines news directional bias with Multi-Timeframe Key Levels & Fibonacci execution.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "TradingStyle": "swing_trading",
        "RiskPct": 0.02,                      # 2% portfolio risk budget
        "MinRiskReward": 1.5,                 # Minimum 1.5:1 Risk/Reward
        "EntryTolerancePct": 0.005,           # 0.5% entry tolerance
        "ATR_Length": 14,
        "ATR_Stop_Buffer_Multiplier": 0.5,
        "AllowOvernight": True,
    }

    _analysis_cache: Dict[str, Dict[str, Any]] = {}

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.risk_pct = float(self.parameters.get("RiskPct", 0.02))
        self.min_rr = float(self.parameters.get("MinRiskReward", 1.5))
        self.entry_tol = float(self.parameters.get("EntryTolerancePct", 0.005))

        # Trade Tracker
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_trade_id = None
        self.matched_signal: Optional[ReactorSignal] = None

        # Level cache
        self.merged_levels = pd.DataFrame()
        self.support_levels = pd.DataFrame()
        self.resistance_levels = pd.DataFrame()
        self.fib_trade_setups = pd.DataFrame()
        self.last_analysis_date = None

        self._load_and_bind_signal()

    def _load_and_bind_signal(self):
        ticker_sym = self.parameters["Ticker"].symbol
        all_signals = fetch_reactor_signals(self.signal_date_str)
        ticker_signals = [s for s in all_signals if s.ticker == ticker_sym]

        if not ticker_signals:
            self.log_message(f"[{ticker_sym}] No signal found for {self.signal_date_str}.")
            return

        sig = ticker_signals[-1]
        self.matched_signal = sig

        status = "ELIGIBLE" if sig.is_eligible else "REJECTED (<70% dual floor)"
        self.log_message(
            f"[{ticker_sym}] NEWS BIAS GATED: {sig.signal_position} | "
            f"Fund: {sig.fundamental_sentiment} ({sig.fundamental_confidence:.1%}) | "
            f"News: {sig.news_sentiment} ({sig.news_conviction:.1%}) -> {status}"
        )

    def _compute_intraday_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        atr_len = int(self.parameters.get("ATR_Length", 14))
        data.ta.atr(length=atr_len, append=True)
        atr_cols = [c for c in data.columns if c.startswith("ATRr_")]
        data["atr"] = data[atr_cols[0]] if atr_cols else data["close"] * 0.015
        return data

    def _refresh_levels(self, df_5m: pd.DataFrame, current_date: datetime.date):
        ticker_sym = self.parameters["Ticker"].symbol
        cache_key = f"{ticker_sym}_{current_date}"

        if cache_key in NewsMTFKeyLevelsStrategy._analysis_cache:
            cached = NewsMTFKeyLevelsStrategy._analysis_cache[cache_key]
            self.merged_levels = cached["merged"]
            self.support_levels = cached["support"]
            self.resistance_levels = cached["resistance"]
            self.fib_trade_setups = cached["fib"]
            self.last_analysis_date = current_date
            return

        if df_5m.empty or len(df_5m) < 40:
            return

        try:
            df_1d = df_5m.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
            df_4h = df_5m.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
            df_15m = df_5m.resample("15min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

            raw_levels = []
            for tf_label, r_df in [("1D", df_1d), ("4H", df_4h), ("15m", df_15m)]:
                if len(r_df) >= 10:
                    found = find_key_levels(r_df, resolution=tf_label)
                    if not found.empty:
                        raw_levels.append(found)

            if raw_levels:
                combined = pd.concat(raw_levels, ignore_index=True)
                self.merged_levels = merge_key_levels(combined)
                self.support_levels = self.merged_levels[self.merged_levels["type"] == "support"].sort_values("level_price", ascending=False)
                self.resistance_levels = self.merged_levels[self.merged_levels["type"] == "resistance"].sort_values("level_price", ascending=True)
            else:
                self.merged_levels = pd.DataFrame()
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()

            # Fibonacci setups
            fib_list = []
            for tf_label, r_df in [("1D", df_1d), ("15m", df_15m)]:
                if len(r_df) >= 15:
                    setups = get_fibonacci_trade_setups(r_df, resolution=tf_label, backcandles=30)
                    if not setups.empty:
                        fib_list.append(setups)
            
            self.fib_trade_setups = pd.concat(fib_list, ignore_index=True) if fib_list else pd.DataFrame()

            NewsMTFKeyLevelsStrategy._analysis_cache[cache_key] = {
                "merged": self.merged_levels,
                "support": self.support_levels,
                "resistance": self.resistance_levels,
                "fib": self.fib_trade_setups,
            }
            self.last_analysis_date = current_date
        except Exception as e:
            self.log_message(f"[{ticker_sym}] Level refresh error: {e}")

    def on_trading_iteration(self):
        ticker = self.parameters["Ticker"]
        ticker_sym = ticker.symbol
        current_price = self.get_last_price(ticker)
        current_dt = self.get_datetime()

        if current_price is None or current_price <= 0:
            return

        # Monitor active trade
        pos = self.get_position(ticker)
        if pos is not None and pos.quantity != 0:
            if self.active_trade_id and self.active_trade_id in self.trade_tracker.trades:
                trade = self.trade_tracker.trades[self.active_trade_id]
                # TP check
                if (trade.trade_type == "BUY" and current_price >= trade.take_profit) or \
                   (trade.trade_type == "SELL" and current_price <= trade.take_profit):
                    self.log_message(f"[{ticker_sym}] MTF KEY LEVELS TP REACHED @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "TP")
                    self._close_position()
                    self.active_trade_id = None
                    return
                # SL check
                if (trade.trade_type == "BUY" and current_price <= trade.stop_loss) or \
                   (trade.trade_type == "SELL" and current_price >= trade.stop_loss):
                    self.log_message(f"[{ticker_sym}] MTF KEY LEVELS SL HIT @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "SL")
                    self._close_position()
                    self.active_trade_id = None
                    return
            return

        # Check news eligibility
        if not self.matched_signal or not self.matched_signal.is_eligible:
            return

        allowed_side = self.matched_signal.signal_position  # "BUY" or "SELL"

        # Time gate: RTH 09:45 - 15:45 EST
        current_time = current_dt.time()
        if current_time < datetime.time(9, 45) or current_time > datetime.time(15, 45):
            return

        # Fetch 5m intraday bars
        bars = self.get_historical_prices(ticker, 300, "5M")
        if bars is None or len(bars.df) < 30:
            return

        df = self._compute_intraday_indicators(bars.df)
        today = current_dt.date()
        if self.last_analysis_date != today or self.merged_levels.empty:
            self._refresh_levels(df, today)

        atr_val = float(df.iloc[-1]["atr"])
        atr_buf = atr_val * float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 0.5))

        chosen_setup = None

        # PRIORITY 1: Fibonacci Retracement in allowed news direction
        if not self.fib_trade_setups.empty:
            for _, setup in self.fib_trade_setups.iterrows():
                trend = str(setup["trend"]).lower()
                side = "BUY" if trend == "uptrend" else "SELL"
                if side != allowed_side:
                    continue  # Filter out opposing setups!

                fib_entry = float(setup["entry_price"])
                if abs(current_price - fib_entry) <= (fib_entry * self.entry_tol):
                    raw_tp = float(setup["take_profit"])
                    raw_sl = float(setup["stop_loss"])
                    sl = raw_sl - atr_buf if side == "BUY" else raw_sl + atr_buf
                    tp = raw_tp
                    risk = abs(current_price - sl)
                    reward = abs(tp - current_price)

                    if risk > 0 and (reward / risk) >= self.min_rr:
                        chosen_setup = {
                            "side": side,
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reason": f"FIB_{trend.upper()}_0.618"
                        }
                        break

        # PRIORITY 2: Key Level Bounce in allowed news direction
        if chosen_setup is None:
            if allowed_side == "BUY" and not self.support_levels.empty:
                nearest_sup = float(self.support_levels.iloc[0]["level_price"])
                if abs(current_price - nearest_sup) <= (nearest_sup * self.entry_tol):
                    tp = float(self.resistance_levels.iloc[0]["level_price"]) if not self.resistance_levels.empty else current_price + (atr_val * 2.5)
                    sl = nearest_sup - atr_buf
                    risk = current_price - sl
                    reward = tp - current_price
                    if risk > 0 and (reward / risk) >= self.min_rr:
                        chosen_setup = {
                            "side": "BUY",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reason": "SUPPORT_BOUNCE"
                        }
            elif allowed_side == "SELL" and not self.resistance_levels.empty:
                nearest_res = float(self.resistance_levels.iloc[0]["level_price"])
                if abs(current_price - nearest_res) <= (nearest_res * self.entry_tol):
                    tp = float(self.support_levels.iloc[0]["level_price"]) if not self.support_levels.empty else current_price - (atr_val * 2.5)
                    sl = nearest_res + atr_buf
                    risk = sl - current_price
                    reward = current_price - tp
                    if risk > 0 and (reward / risk) >= self.min_rr:
                        chosen_setup = {
                            "side": "SELL",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "reason": "RESISTANCE_REJECT"
                        }

        if chosen_setup is None:
            return

        # Sizing (2% risk)
        portfolio_val = self.get_portfolio_value()
        risk_budget = portfolio_val * self.risk_pct
        risk_per_share = abs(current_price - chosen_setup["stop_loss"])
        qty = floor(risk_budget / max(0.01, risk_per_share))
        max_shares = floor((portfolio_val * 0.95) / current_price)
        quantity = max(1, min(qty, max_shares))

        # Submit bracket order
        order_side = "buy" if chosen_setup["side"] == "BUY" else "sell"
        self.log_message(
            f"[{ticker_sym}] NEWS+MTF TRIGGERED: {chosen_setup['reason']} | "
            f"Side: {chosen_setup['side']} | Qty: {quantity} @ ${current_price:.2f} | "
            f"SL: ${chosen_setup['stop_loss']:.2f} | TP: ${chosen_setup['take_profit']:.2f}"
        )

        self.active_trade_id = self.trade_tracker.open_trade(
            date=current_dt,
            entry_price=current_price,
            quantity=quantity,
            take_profit=chosen_setup["take_profit"],
            stop_loss=chosen_setup["stop_loss"],
            support_level=chosen_setup.get("stop_loss", current_price * 0.98),
            resistance_level=chosen_setup.get("take_profit", current_price * 1.02),
            trade_type=chosen_setup["side"]
        )

        order = self.create_order(
            ticker,
            quantity,
            order_side,
            take_profit_price=chosen_setup["take_profit"],
            stop_loss_price=chosen_setup["stop_loss"],
            type="market"
        )
        self.submit_order(order)

    def _close_position(self):
        ticker = self.parameters["Ticker"]
        pos = self.get_position(ticker)
        if pos is not None and pos.quantity != 0:
            side = "sell" if pos.quantity > 0 else "buy"
            order = self.create_order(ticker, abs(pos.quantity), side, type="market")
            self.submit_order(order)
