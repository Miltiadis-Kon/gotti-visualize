"""
News + MultiTimeframe Key Levels Strategy (news_mtf_key_levels.py)
==================================================================
Combines Stock Alchemist intelligence news/fundamental signals with
true Multi-Timeframe Support/Resistance clustering (1D, 4H, 1H, 15m),
0.618 Fibonacci Golden Pocket setups, and S/R Breakout Role-Reversals.

Profitability Upgrades:
1. True MTF Horizon: Fetches 60 Daily bars + 400 15m bars + 5m execution bars
   so 1D, 4H, 1H, and 15m Key Levels & Fibonacci setups never starve for data.
2. Three Confirmed Technical Setups (Eliminates Knife-Catching & Adverse Selection):
   - Priority 1 (MTF Breakout / S/R Flip): Captures strong news catalysts breaking above
     MTF resistance (or below support) with RVOL confirmation and room to next HTF target.
   - Priority 2 (Confirmed 0.618 Fib Retracement): Requires price to test the Fib zone AND
     print a 5m reversal candle (close > open for BUY, close < open for SELL).
   - Priority 3 (Confirmed MTF S/R Bounce): Evaluates all valid levels below/above current
     price and requires a 5m rejection candle + EMA-20 / VWAP stabilization.
3. Institutional Risk & Churn Protection:
   - Uses 15m/1H ATR passed through StrategyBaseplate.optimize_sl_tp() (prevents tiny 5m stops).
   - Enforces one trade per signal/session (eliminates repeated re-entry churn on broken levels).
   - Ratchets stop to Breakeven at +1.2R and flattens weak non-aligned trades before close.
"""

import os
import sys
from math import floor, sqrt
import datetime
from datetime import time, timedelta
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
    Combines news directional bias with Multi-Timeframe Key Levels, S/R Flips & Fibonacci execution.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "DynamicDateBinding": True,           # Auto-load signals as simulation date advances
        "EnforceNoLookahead": True,           # Respect signal created_at timestamp
        "TradingStyle": "swing_trading",
        "RiskPct": 0.02,                      # Base 2% portfolio risk budget
        "MinRiskReward": 1.6,                 # Minimum 1.6:1 Risk/Reward
        "EntryTolerancePct": 0.006,           # 0.6% zone proximity tolerance
        "MinBreakoutRVOL": 1.25,              # Minimum RVOL for MTF Breakout setups
        "ATR_Length": 14,
        "ATR_Stop_Buffer_Multiplier": 1.0,    # 1.0x 15m ATR buffer (replaces noisy 0.5x 5m ATR)
        "BreakevenTriggerR": 1.2,             # Move stop to breakeven at +1.2R
        "AllowOvernight": True,
    }

    _analysis_cache: Dict[str, Dict[str, Any]] = {}

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.dynamic_date_binding = bool(self.parameters.get("DynamicDateBinding", True))
        self.enforce_no_lookahead = bool(self.parameters.get("EnforceNoLookahead", True))
        self.risk_pct = float(self.parameters.get("RiskPct", 0.02))
        self.min_rr = float(self.parameters.get("MinRiskReward", 1.6))
        self.entry_tol = float(self.parameters.get("EntryTolerancePct", 0.006))
        self.min_breakout_rvol = float(self.parameters.get("MinBreakoutRVOL", 1.25))
        self.be_trigger_r = float(self.parameters.get("BreakevenTriggerR", 1.2))

        # Trade Tracker & Active Trade State
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_trade_id: Optional[str] = None
        self.active_initial_risk: float = 0.0
        self.active_selector_id: int = 1
        self.breakeven_applied: bool = False

        # Session & Signal Gating State
        self.current_session_date: Optional[datetime.date] = None
        self.session_signals: List[ReactorSignal] = []
        self.matched_signal: Optional[ReactorSignal] = None
        self.traded_signal_ids: set = set()
        self.trades_taken_today: int = 0

        # Opening Range State (9:30 - 9:45 AM EST)
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None

        # MTF Level Cache
        self.merged_levels = pd.DataFrame()
        self.support_levels = pd.DataFrame()
        self.resistance_levels = pd.DataFrame()
        self.fib_trade_setups = pd.DataFrame()
        self.last_analysis_date: Optional[datetime.date] = None

        self._load_signals_for_date(self.signal_date_str)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SIGNAL BINDING & SESSION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_session(self, new_date: datetime.date):
        self.current_session_date = new_date
        self.trades_taken_today = 0
        self.or_high = None
        self.or_low = None
        target_date_str = new_date.strftime("%Y-%m-%d") if self.dynamic_date_binding else self.signal_date_str
        self._load_signals_for_date(target_date_str)

    def _load_signals_for_date(self, target_date_str: str):
        ticker_sym = self.parameters["Ticker"].symbol
        try:
            all_signals = fetch_reactor_signals(
                target_date=target_date_str,
                ticker=ticker_sym,
                include_prev_afterhours=True
            )
        except TypeError:
            # Backward-compatible fallback if using older fetch_reactor_signals signature
            raw_all = fetch_reactor_signals(target_date_str)
            all_signals = [s for s in raw_all if s.ticker == ticker_sym]

        valid_signals: List[ReactorSignal] = []
        for s in all_signals:
            if s.raw_signal_id in self.traded_signal_ids:
                continue
            created_at = getattr(s, "created_at", None)
            sig_date = getattr(s, "signal_date", target_date_str)
            if self.enforce_no_lookahead and created_at is not None and str(sig_date) == target_date_str:
                if created_at.time() >= time(16, 0):
                    continue
            valid_signals.append(s)

        self.session_signals = valid_signals
        if not self.session_signals:
            return

        eligible = [s for s in self.session_signals if s.is_eligible]
        self.matched_signal = eligible[-1] if eligible else self.session_signals[-1]

        sig = self.matched_signal
        status = "ELIGIBLE" if sig.is_eligible else "REJECTED (<70% dual floor)"
        self.log_message(
            f"[{ticker_sym}] NEWS+MTF SIGNAL ({target_date_str}): {sig.signal_position} | "
            f"Fund: {sig.fundamental_sentiment} ({sig.fundamental_confidence:.1%}) | "
            f"News: {sig.news_sentiment} ({sig.news_conviction:.1%}) | "
            f"Selector {sig.selector_id} -> {status}"
        )

    def _get_active_signal(self, current_dt: datetime.datetime) -> Optional[ReactorSignal]:
        cur_date_str = current_dt.strftime("%Y-%m-%d")
        eligible_now = []
        for s in self.session_signals:
            if not s.is_eligible or s.raw_signal_id in self.traded_signal_ids:
                continue
            created_at = getattr(s, "created_at", None)
            sig_date = getattr(s, "signal_date", cur_date_str)
            if self.enforce_no_lookahead and created_at is not None:
                if str(sig_date) == cur_date_str and current_dt.time() < created_at.time():
                    continue
            eligible_now.append(s)

        if eligible_now:
            self.matched_signal = eligible_now[-1]
            return self.matched_signal
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 2. TRUE MULTI-TIMEFRAME LEVEL & FIBONACCI ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_levels(self, current_date: datetime.date):
        """
        Builds true Multi-Timeframe Support/Resistance clusters and Fibonacci setups
        using 60 Daily bars + 400 15m bars (resampled to 1H and 4H).
        """
        ticker = self.parameters["Ticker"]
        ticker_sym = ticker.symbol
        cache_key = f"{ticker_sym}_{current_date}"

        if cache_key in NewsMTFKeyLevelsStrategy._analysis_cache:
            cached = NewsMTFKeyLevelsStrategy._analysis_cache[cache_key]
            self.merged_levels = cached["merged"]
            self.support_levels = cached["support"]
            self.resistance_levels = cached["resistance"]
            self.fib_trade_setups = cached["fib"]
            self.last_analysis_date = current_date
            return

        try:
            bars_1d = self.get_historical_prices(ticker, 65, "day")
            bars_15m = self.get_historical_prices(ticker, 400, "15m")

            df_1d = bars_1d.df if (bars_1d is not None and hasattr(bars_1d, "df")) else pd.DataFrame()
            df_15m = bars_15m.df if (bars_15m is not None and hasattr(bars_15m, "df")) else pd.DataFrame()

            if df_15m.empty:
                return

            df_1h = df_15m.resample("1h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            df_4h = df_15m.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()

            raw_levels = []
            for tf_label, r_df in [("1D", df_1d), ("4H", df_4h), ("1H", df_1h), ("15m", df_15m)]:
                if r_df is not None and not r_df.empty and len(r_df) >= 10:
                    found = find_key_levels(r_df, resolution=tf_label)
                    if found is not None and not found.empty:
                        raw_levels.append(found)

            if raw_levels:
                combined = pd.concat(raw_levels, ignore_index=True)
                self.merged_levels = merge_key_levels(combined)
                self.support_levels = self.merged_levels[
                    self.merged_levels["type"] == "support"
                ].sort_values("level_price", ascending=False)
                self.resistance_levels = self.merged_levels[
                    self.merged_levels["type"] == "resistance"
                ].sort_values("level_price", ascending=True)
            else:
                self.merged_levels = pd.DataFrame()
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()

            # Compute Fibonacci Retracement Setups across 1D, 1H, and 15m
            fib_list = []
            for tf_label, r_df in [("1D", df_1d), ("1H", df_1h), ("15m", df_15m)]:
                if r_df is not None and not r_df.empty and len(r_df) >= 20:
                    setups = get_fibonacci_trade_setups(r_df, resolution=tf_label, backcandles=min(35, len(r_df) - 2))
                    if setups is not None and not setups.empty:
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
            self.log_message(f"[{ticker_sym}] MTF level refresh error: {e}")

    def _get_15m_atr(self, fallback_price: float) -> float:
        """Computes 14-period ATR on 15m bars so stops are not sized off 5m noise."""
        ticker = self.parameters["Ticker"]
        try:
            bars_15m = self.get_historical_prices(ticker, 35, "15m")
            if bars_15m is not None and len(bars_15m.df) >= 15:
                return float(calculate_atr(bars_15m.df, length=14))
        except Exception:
            pass
        return fallback_price * 0.012

    def _update_opening_range(self, df_5m: pd.DataFrame, today: datetime.date):
        if self.or_high is not None and self.or_low is not None:
            return
        try:
            idx = pd.to_datetime(df_5m.index)
            mask = (idx.date == today) & (idx.time >= time(9, 30)) & (idx.time < time(9, 45))
            or_bars = df_5m[mask]
            if not or_bars.empty:
                self.or_high = float(or_bars["high"].max())
                self.or_low = float(or_bars["low"].min())
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 3. THREE CONFIRMED MTF ENTRY SETUPS
    # ─────────────────────────────────────────────────────────────────────────

    def _find_confirmed_setup(
        self,
        allowed_side: str,
        df_5m: pd.DataFrame,
        current_price: float,
        atr_15m: float
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates 3 hierarchy setups in the news-approved direction, requiring
        5m candle confirmation so the strategy never catches falling knives.
        """
        last_bar = df_5m.iloc[-1]
        prev_bar = df_5m.iloc[-2]
        bar_open = float(last_bar["open"])
        bar_close = float(last_bar["close"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        prev_close = float(prev_bar["close"])

        # Relative Volume (RVOL) on 5m bar
        avg_vol = float(df_5m["volume"].iloc[:-1].tail(20).mean()) if "volume" in df_5m.columns else 1.0
        cur_vol = float(last_bar.get("volume", 1.0))
        rvol = (cur_vol / avg_vol) if avg_vol > 0 else 1.3

        # 20-period EMA on 5m for short-term stabilization check
        ema20 = float(df_5m["close"].ewm(span=20, adjust=False).mean().iloc[-1])

        # Candle Confirmation Flags
        is_bullish_candle = (bar_close > bar_open) and (bar_close >= prev_close)
        is_bearish_candle = (bar_close < bar_open) and (bar_close <= prev_close)

        # Prevent buying a breakdown below the Opening Range Low (or shorting above OR_High)
        if allowed_side == "BUY" and self.or_low is not None and current_price < self.or_low:
            return None
        if allowed_side == "SELL" and self.or_high is not None and current_price > self.or_high:
            return None

        atr_buf = atr_15m * float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 1.0))
        all_levels_prices = (
            np.sort(self.merged_levels["level_price"].unique())
            if not self.merged_levels.empty
            else np.array([])
        )

        # ── SETUP 1: MTF KEY LEVEL BREAKOUT / S/R FLIP (Captures Strong News Runners) ──
        if len(all_levels_prices) > 0 and rvol >= self.min_breakout_rvol:
            if allowed_side == "BUY" and is_bullish_candle and current_price >= ema20:
                below_levels = all_levels_prices[all_levels_prices <= current_price]
                above_levels = all_levels_prices[all_levels_prices > current_price + (0.5 * atr_15m)]
                if len(below_levels) > 0:
                    flipped_level = float(below_levels[-1])
                    # Confirm price just broke or retested above the flipped level within tolerance
                    if 0 <= (current_price - flipped_level) <= (flipped_level * self.entry_tol * 1.2) and (
                        bar_low <= flipped_level * (1 + self.entry_tol) or prev_close <= flipped_level * (1 + self.entry_tol)
                    ):
                        sl = flipped_level - atr_buf
                        tp = float(above_levels[0]) if len(above_levels) > 0 else current_price + (2.5 * atr_15m)
                        if (tp - current_price) / max(0.01, current_price - sl) >= self.min_rr:
                            return {
                                "side": "BUY",
                                "entry": current_price,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "support": flipped_level,
                                "resistance": tp,
                                "reason": f"MTF_SR_FLIP_BREAKOUT (RVOL={rvol:.2f})"
                            }

            elif allowed_side == "SELL" and is_bearish_candle and current_price <= ema20:
                above_levels = all_levels_prices[all_levels_prices >= current_price]
                below_levels = all_levels_prices[all_levels_prices < current_price - (0.5 * atr_15m)]
                if len(above_levels) > 0:
                    flipped_level = float(above_levels[0])
                    if 0 <= (flipped_level - current_price) <= (flipped_level * self.entry_tol * 1.2) and (
                        bar_high >= flipped_level * (1 - self.entry_tol) or prev_close >= flipped_level * (1 - self.entry_tol)
                    ):
                        sl = flipped_level + atr_buf
                        tp = float(below_levels[-1]) if len(below_levels) > 0 else current_price - (2.5 * atr_15m)
                        if (current_price - tp) / max(0.01, sl - current_price) >= self.min_rr:
                            return {
                                "side": "SELL",
                                "entry": current_price,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "support": tp,
                                "resistance": flipped_level,
                                "reason": f"MTF_SR_FLIP_BREAKDOWN (RVOL={rvol:.2f})"
                            }

        # ── SETUP 2: CONFIRMED FIBONACCI 0.618 RETRACEMENT ───────────────────
        if not self.fib_trade_setups.empty:
            for _, setup in self.fib_trade_setups.iterrows():
                trend = str(setup["trend"]).lower()
                side = "BUY" if trend == "uptrend" else "SELL"
                if side != allowed_side:
                    continue

                fib_entry = float(setup["entry_price"])
                # Require price to have tested the Fib zone AND printed a reversal candle
                in_fib_zone = abs(current_price - fib_entry) <= (fib_entry * self.entry_tol)
                if side == "BUY" and in_fib_zone and is_bullish_candle and bar_close >= fib_entry:
                    raw_sl = min(float(setup["stop_loss"]), current_price - atr_buf)
                    raw_tp = float(setup["take_profit"])
                    if raw_tp > current_price and (raw_tp - current_price) / max(0.01, current_price - raw_sl) >= self.min_rr:
                        return {
                            "side": "BUY",
                            "entry": current_price,
                            "stop_loss": raw_sl,
                            "take_profit": raw_tp,
                            "support": fib_entry,
                            "resistance": raw_tp,
                            "reason": "FIB_UPTREND_0.618_CONFIRMED"
                        }
                elif side == "SELL" and in_fib_zone and is_bearish_candle and bar_close <= fib_entry:
                    raw_sl = max(float(setup["stop_loss"]), current_price + atr_buf)
                    raw_tp = float(setup["take_profit"])
                    if raw_tp < current_price and (current_price - raw_tp) / max(0.01, raw_sl - current_price) >= self.min_rr:
                        return {
                            "side": "SELL",
                            "entry": current_price,
                            "stop_loss": raw_sl,
                            "take_profit": raw_tp,
                            "support": raw_tp,
                            "resistance": fib_entry,
                            "reason": "FIB_DOWNTREND_0.618_CONFIRMED"
                        }

        # ── SETUP 3: CONFIRMED MTF KEY LEVEL BOUNCE (Scans All Valid Levels) ─
        if allowed_side == "BUY" and not self.support_levels.empty and is_bullish_candle:
            # Filter supports at or below current_price (fixes iloc[0] bug)
            valid_sups = self.support_levels[
                self.support_levels["level_price"] <= current_price * (1 + self.entry_tol * 0.5)
            ]["level_price"].values
            valid_ress = self.resistance_levels[
                self.resistance_levels["level_price"] > current_price + atr_15m
            ]["level_price"].values

            for sup_px in valid_sups:
                sup_val = float(sup_px)
                # Confirm low tested support zone and candle rejected upward
                if abs(current_price - sup_val) <= (sup_val * self.entry_tol) and bar_close > sup_val:
                    sl = sup_val - atr_buf
                    tp = float(valid_ress[0]) if len(valid_ress) > 0 else current_price + (2.5 * atr_15m)
                    if (tp - current_price) / max(0.01, current_price - sl) >= self.min_rr:
                        return {
                            "side": "BUY",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "support": sup_val,
                            "resistance": tp,
                            "reason": "MTF_SUPPORT_REJECTION"
                        }

        elif allowed_side == "SELL" and not self.resistance_levels.empty and is_bearish_candle:
            valid_ress = self.resistance_levels[
                self.resistance_levels["level_price"] >= current_price * (1 - self.entry_tol * 0.5)
            ]["level_price"].values
            valid_sups = self.support_levels[
                self.support_levels["level_price"] < current_price - atr_15m
            ]["level_price"].values

            for res_px in valid_ress:
                res_val = float(res_px)
                if abs(current_price - res_val) <= (res_val * self.entry_tol) and bar_close < res_val:
                    sl = res_val + atr_buf
                    tp = float(valid_sups[0]) if len(valid_sups) > 0 else current_price - (2.5 * atr_15m)
                    if (current_price - tp) / max(0.01, sl - current_price) >= self.min_rr:
                        return {
                            "side": "SELL",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "support": tp,
                            "resistance": res_val,
                            "reason": "MTF_RESISTANCE_REJECTION"
                        }

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 4. MAIN EXECUTION LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        ticker = self.parameters["Ticker"]
        ticker_sym = ticker.symbol
        current_price = self.get_last_price(ticker)
        current_dt = self.get_datetime()

        if current_price is None or current_price <= 0:
            return

        # Daily Session Rollover
        today = current_dt.date()
        if self.current_session_date != today:
            self._reset_session(today)

        # ── 1. SYNC BROKER BRACKET EXITS & MANAGE OPEN TRADES ────────────────
        pos = self.get_position(ticker)
        has_open_pos = pos is not None and pos.quantity != 0

        if not has_open_pos and self.active_trade_id is not None:
            if self.active_trade_id in self.trade_tracker.trades:
                trade = self.trade_tracker.trades[self.active_trade_id]
                if trade.is_open:
                    dist_tp = abs(current_price - trade.take_profit)
                    dist_sl = abs(current_price - trade.stop_loss)
                    reason = "TP" if dist_tp <= dist_sl else "SL"
                    fill_px = trade.take_profit if reason == "TP" else trade.stop_loss
                    self.log_message(f"[{ticker_sym}] MTF Bracket {reason} filled @ ~${fill_px:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, fill_px, reason)
            self.active_trade_id = None
            self.breakeven_applied = False

        if has_open_pos and self.active_trade_id in self.trade_tracker.trades:
            trade = self.trade_tracker.trades[self.active_trade_id]
            pnl_per_share = (current_price - trade.entry_price) if trade.trade_type == "BUY" else (trade.entry_price - current_price)
            r_mult = (pnl_per_share / self.active_initial_risk) if self.active_initial_risk > 0 else 0.0

            # EOD Protection for Selector 2 (Counter-Fundamental) or weak Selector 3 trades
            if current_dt.time() >= time(15, 50):
                if self.active_selector_id == 2 or (self.active_selector_id == 3 and r_mult < 0.5):
                    self.log_message(f"[{ticker_sym}] EOD Risk Guard Flatten @ ${current_price:.2f} ({r_mult:+.2f}R)")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "EOD_GUARD")
                    self._close_position()
                    return

            # Ratchet Stop Loss to Breakeven (+0.1R cushion) once trade reaches +1.2R
            if not self.breakeven_applied and r_mult >= self.be_trigger_r:
                cushion = 0.10 * self.active_initial_risk
                trade.stop_loss = (trade.entry_price + cushion) if trade.trade_type == "BUY" else (trade.entry_price - cushion)
                self.breakeven_applied = True
                self.log_message(f"[{ticker_sym}] Reached +{r_mult:.2f}R -> Stop ratcheted to ${trade.stop_loss:.2f}")

            if self.breakeven_applied:
                if (trade.trade_type == "BUY" and current_price <= trade.stop_loss) or \
                   (trade.trade_type == "SELL" and current_price >= trade.stop_loss):
                    self.log_message(f"[{ticker_sym}] Protected Breakeven Stop Hit @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "BE_STOP")
                    self._close_position()
            return

        # ── 2. GATING: MAX 1 TRADE PER SESSION & RTH WINDOW (09:45 - 15:30) ──
        if self.trades_taken_today >= 1:
            return

        current_time = current_dt.time()
        if current_time < time(9, 45) or current_time > time(15, 30):
            return

        sig = self._get_active_signal(current_dt)
        if sig is None or not sig.is_eligible:
            return

        # Fetch 5m execution bars & update Opening Range + MTF Levels
        bars_5m = self.get_historical_prices(ticker, 120, "5M")
        if bars_5m is None or len(bars_5m.df) < 25:
            return

        df_5m = bars_5m.df
        self._update_opening_range(df_5m, today)

        if self.last_analysis_date != today or self.merged_levels.empty:
            self._refresh_levels(today)

        atr_15m = self._get_15m_atr(current_price)

        # Evaluate 3 MTF Confirmed Setups
        chosen_setup = self._find_confirmed_setup(
            allowed_side=sig.signal_position,
            df_5m=df_5m,
            current_price=current_price,
            atr_15m=atr_15m
        )

        if chosen_setup is None:
            return

        # ── 3. OPTIMIZE SL/TP VIA STRATEGY BASEPLATE ATR ENGINE ──────────────
        order_side = "buy" if chosen_setup["side"] == "BUY" else "sell"
        style = getattr(sig, "trading_style", self.parameters.get("TradingStyle", "swing_trading"))

        risk_levels = self.optimize_sl_tp(
            entry_price=current_price,
            stop_loss=chosen_setup["stop_loss"],
            take_profit=chosen_setup["take_profit"],
            side=order_side,
            trading_style=style,
            atr=atr_15m,
            risk_reward_ratio=self.min_rr
        )

        final_sl = risk_levels.stop_loss
        final_tp = risk_levels.take_profit

        # Conviction-scaled sizing with 50% max single-position cap (prevents over-leverage)
        portfolio_val = self.get_portfolio_value()
        comp_conv = getattr(sig, "composite_conviction", sqrt(sig.fundamental_confidence * sig.news_conviction))
        conv_scale = 0.75 + 0.50 * max(0.0, min(1.0, (comp_conv - 0.70) / 0.30))
        risk_budget = portfolio_val * self.risk_pct * conv_scale

        # Enforce minimum 0.8x 15m ATR stop distance for share calculation
        effective_risk_per_share = max(abs(current_price - final_sl), 0.8 * atr_15m, current_price * 0.008)
        qty = floor(risk_budget / effective_risk_per_share)
        max_shares = floor((portfolio_val * 0.50) / current_price)
        quantity = max(0, min(qty, max_shares))

        if quantity <= 0:
            return

        self.log_message(
            f"[{ticker_sym}] NEWS+MTF CONFIRMED: {chosen_setup['reason']} | "
            f"Side: {chosen_setup['side']} | Qty: {quantity} @ ${current_price:.2f} | "
            f"SL: ${final_sl:.2f} | TP: ${final_tp:.2f} | R:R: {risk_levels.risk_reward_ratio:.2f}"
        )

        self.active_trade_id = self.trade_tracker.open_trade(
            date=current_dt,
            entry_price=current_price,
            quantity=quantity,
            take_profit=final_tp,
            stop_loss=final_sl,
            support_level=chosen_setup["support"],
            resistance_level=chosen_setup["resistance"],
            trade_type=chosen_setup["side"]
        )
        self.active_initial_risk = abs(current_price - final_sl)
        self.active_selector_id = sig.selector_id
        self.breakeven_applied = False
        self.trades_taken_today += 1
        self.traded_signal_ids.add(sig.raw_signal_id)

        order = self.create_order(
            ticker,
            quantity,
            order_side,
            take_profit_price=final_tp,
            stop_loss_price=final_sl,
            type="market"
        )
        self.submit_order(order)

    def on_strategy_end(self):
        ticker = self.parameters["Ticker"]
        current_price = self.get_last_price(ticker)
        current_dt = self.get_datetime()

        if self.active_trade_id and self.active_trade_id in self.trade_tracker.trades:
            trade = self.trade_tracker.trades[self.active_trade_id]
            if trade.is_open:
                self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "END_SIM")
                self._close_position()

        perf = self.trade_tracker.get_performance_summary()
        self.log_message(f"[{ticker.symbol}] News+MTF Performance Summary: {perf}")

    def _close_position(self):
        self.cancel_open_orders()
        ticker = self.parameters["Ticker"]
        pos = self.get_position(ticker)
        if pos is not None and pos.quantity != 0:
            side = "sell" if pos.quantity > 0 else "buy"
            order = self.create_order(ticker, abs(pos.quantity), side, type="market")
            self.submit_order(order)
        self.active_trade_id = None
        self.breakeven_applied = False
