"""
News + Catalyst Impulse Fibonacci & MTF Confluence Strategy (news_mtf_key_levels.py)
====================================================================================
Institutional replacement for standard static backcandle Fibonacci strategies.
Anchors Fibonacci grids directly to the News Catalyst Impulse Wave, combines
retracement tiers (0.382-0.500 Institutional Reload & 0.500-0.618 Golden Pocket)
with Multi-Timeframe (1D, 4H, 1H) Support/Resistance Role-Reversal and VWAP,
and targets 1.272 / 1.618 Fibonacci Measured Move Extensions.

Core Institutional Mechanics:
1. Catalyst Impulse Anchoring (Leg 1):
   - Dynamically anchors Fib 1.0 (Origin) to the Catalyst Base (Opening Range Low /
     Previous Day Close for gap-and-go) and Fib 0.0 (Peak) to the active session
     impulse extreme once Impulse Range >= 1.0x 15m ATR.
2. Two-Tier Fibonacci Retracement + MTF Confluence:
   - Tier 1 (0.382 - 0.500 Shallow Reload): High-momentum continuation zone when
     price holds above VWAP / EMA-20.
   - Tier 2 (0.500 - 0.618 Golden Pocket): Required to align with an MTF Key Level
     cluster (1D/4H/1H) or VWAP.
   - 0.786 Hard Invalidation: If price retraces > 78.6% of the catalyst impulse,
     the news thesis is declared a "sell-the-news" failure and locked out.
3. Volume Contraction & Reversal Trigger:
   - Requires pullback volume to stay below impulse volume and a confirmed 5m
     reversal candle reclaiming the Fib zone before firing.
4. Fibonacci Extension Targets & Wave-3 Ratchet:
   - Targets the 1.272 or 1.618 Fib Extension (adjusted for overhead 1D/4H resistance).
   - Automatically ratchets Stop Loss to +0.2R as soon as price re-tests the 0.0
     Impulse High/Low (Wave 3 breakout confirmation).
"""

import os
import sys
from math import floor, sqrt
import datetime
from datetime import time
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
    Institutional Catalyst Impulse Fibonacci + MTF Key Level Confluence Strategy.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "DynamicDateBinding": True,           # Reload signals automatically across multi-day backtests
        "EnforceNoLookahead": True,           # Respect signal created_at timestamp
        "TradingStyle": "swing_trading",
        "RiskPct": 0.02,                      # Base 2% portfolio risk budget
        "MinRiskReward": 1.6,                 # Minimum 1.6:1 Risk/Reward
        "MinImpulseATRMult": 0.9,             # Minimum Catalyst Impulse size (in 15m ATRs) to anchor Fib
        "ConfluenceTolerancePct": 0.006,      # 0.6% proximity for MTF Key Level confluence
        "ExtensionTargetFib": 1.272,          # Primary Fibonacci Extension target (1.272 or 1.618)
        "ATR_Length": 14,
        "ATR_Stop_Buffer_Multiplier": 0.5,    # Buffer beyond the 0.786 Fib / pullback swing extreme
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
        self.min_impulse_atr = float(self.parameters.get("MinImpulseATRMult", 0.9))
        self.conf_tol = float(self.parameters.get("ConfluenceTolerancePct", 0.006))
        self.ext_target_fib = float(self.parameters.get("ExtensionTargetFib", 1.272))

        # Trade Tracker & Active Trade State
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_trade_id: Optional[str] = None
        self.active_initial_risk: float = 0.0
        self.active_impulse_peak: Optional[float] = None
        self.active_selector_id: int = 1
        self.wave3_ratchet_applied: bool = False

        # Daily Session & Signal State
        self.current_session_date: Optional[datetime.date] = None
        self.session_signals: List[ReactorSignal] = []
        self.matched_signal: Optional[ReactorSignal] = None
        self.traded_signal_ids: set = set()
        self.trades_taken_today: int = 0
        self.thesis_invalidated_today: bool = False

        # MTF Structural Cache
        self.merged_levels = pd.DataFrame()
        self.support_levels = pd.DataFrame()
        self.resistance_levels = pd.DataFrame()
        self.htf_fib_setups = pd.DataFrame()
        self.prev_day_close: Optional[float] = None
        self.prev_day_high: Optional[float] = None
        self.prev_day_low: Optional[float] = None
        self.last_analysis_date: Optional[datetime.date] = None

        self._load_signals_for_date(self.signal_date_str)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SESSION & SIGNAL MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_session(self, new_date: datetime.date):
        self.current_session_date = new_date
        self.trades_taken_today = 0
        self.thesis_invalidated_today = False
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
            f"[{ticker_sym}] CIGC FIB STRATEGY ({target_date_str}): {sig.signal_position} | "
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
    # 2. MULTI-TIMEFRAME S/R CLUSTERS & INTRADAY INDICATORS
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_mtf_structure(self, current_date: datetime.date):
        """
        Fetches 65 Daily bars + 400 15m bars to build true 1D, 4H, 1H, and 15m
        Key Level clusters and Higher-Timeframe Fibonacci swings.
        """
        ticker = self.parameters["Ticker"]
        ticker_sym = ticker.symbol
        cache_key = f"{ticker_sym}_{current_date}"

        if cache_key in NewsMTFKeyLevelsStrategy._analysis_cache:
            cached = NewsMTFKeyLevelsStrategy._analysis_cache[cache_key]
            self.merged_levels = cached["merged"]
            self.support_levels = cached["support"]
            self.resistance_levels = cached["resistance"]
            self.htf_fib_setups = cached["htf_fib"]
            self.prev_day_close = cached["pdc"]
            self.prev_day_high = cached["pdh"]
            self.prev_day_low = cached["pdl"]
            self.last_analysis_date = current_date
            return

        try:
            bars_1d = self.get_historical_prices(ticker, 65, "day")
            bars_15m = self.get_historical_prices(ticker, 400, "15m")

            df_1d = bars_1d.df if (bars_1d is not None and hasattr(bars_1d, "df")) else pd.DataFrame()
            df_15m = bars_15m.df if (bars_15m is not None and hasattr(bars_15m, "df")) else pd.DataFrame()

            if not df_1d.empty and len(df_1d) >= 2:
                prev_d = df_1d.iloc[-2]
                self.prev_day_close = float(prev_d["close"])
                self.prev_day_high = float(prev_d["high"])
                self.prev_day_low = float(prev_d["low"])

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

            # Higher-Timeframe (1D & 1H) Macro Fibonacci Swings
            fib_list = []
            for tf_label, r_df in [("1D", df_1d), ("1H", df_1h)]:
                if r_df is not None and not r_df.empty and len(r_df) >= 25:
                    setups = get_fibonacci_trade_setups(r_df, resolution=tf_label, backcandles=min(40, len(r_df) - 2))
                    if setups is not None and not setups.empty:
                        fib_list.append(setups)
            self.htf_fib_setups = pd.concat(fib_list, ignore_index=True) if fib_list else pd.DataFrame()

            NewsMTFKeyLevelsStrategy._analysis_cache[cache_key] = {
                "merged": self.merged_levels,
                "support": self.support_levels,
                "resistance": self.resistance_levels,
                "htf_fib": self.htf_fib_setups,
                "pdc": self.prev_day_close,
                "pdh": self.prev_day_high,
                "pdl": self.prev_day_low,
            }
            self.last_analysis_date = current_date
        except Exception as e:
            self.log_message(f"[{ticker_sym}] MTF structure refresh error: {e}")

    def _compute_intraday_features(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        data = df_5m.copy()
        data["ema9"] = data["close"].ewm(span=9, adjust=False).mean()
        data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["vol_ma"] = data["volume"].rolling(20, min_periods=5).mean()

        if isinstance(data.index, pd.DatetimeIndex):
            idx_ny = (
                data.index.tz_convert("America/New_York")
                if data.index.tz
                else data.index.tz_localize("UTC").tz_convert("America/New_York")
            )
            data["session_date"] = idx_ny.date
            data["bar_time"] = idx_ny.time
        else:
            data["session_date"] = [datetime.date.today()] * len(data)
            data["bar_time"] = [time(9, 30)] * len(data)

        data["tp_px"] = (data["high"] + data["low"] + data["close"]) / 3.0
        data["pv"] = data["tp_px"] * data["volume"]
        rth_pv = np.where(data["bar_time"] >= time(9, 30), data["pv"], 0.0)
        rth_vol = np.where(data["bar_time"] >= time(9, 30), data["volume"], 0.0)
        data["cum_pv"] = pd.Series(rth_pv, index=data.index).groupby(data["session_date"]).cumsum()
        data["cum_vol"] = pd.Series(rth_vol, index=data.index).groupby(data["session_date"]).cumsum()
        data["vwap"] = (data["cum_pv"] / data["cum_vol"].replace(0, np.nan)).ffill().bfill()
        return data

    def _get_15m_atr(self, fallback_price: float) -> float:
        ticker = self.parameters["Ticker"]
        try:
            bars_15m = self.get_historical_prices(ticker, 35, "15m")
            if bars_15m is not None and len(bars_15m.df) >= 15:
                return float(calculate_atr(bars_15m.df, length=14))
        except Exception:
            pass
        return fallback_price * 0.012

    def _has_mtf_confluence(self, price_level: float, vwap_val: float) -> Tuple[bool, str]:
        """Checks if a Fibonacci retracement level aligns with an MTF S/R cluster or VWAP."""
        if abs(price_level - vwap_val) / max(0.01, vwap_val) <= self.conf_tol:
            return True, "VWAP"

        if self.prev_day_high and abs(price_level - self.prev_day_high) / self.prev_day_high <= self.conf_tol:
            return True, "PDH_FLIP"

        if self.prev_day_low and abs(price_level - self.prev_day_low) / self.prev_day_low <= self.conf_tol:
            return True, "PDL_FLIP"

        if not self.merged_levels.empty:
            for _, row in self.merged_levels.iterrows():
                kl_px = float(row["level_price"])
                if abs(price_level - kl_px) / max(0.01, kl_px) <= self.conf_tol:
                    return True, f"MTF_{row.get('type', 'SR').upper()}"

        return False, "NONE"

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CATALYST IMPULSE FIBONACCI ENGINE (THE CORE EDGE)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_catalyst_fibonacci(
        self,
        sig: ReactorSignal,
        df_5m: pd.DataFrame,
        today: datetime.date,
        current_price: float,
        atr_15m: float
    ) -> Optional[Dict[str, Any]]:
        """
        1. Identifies the Catalyst Impulse Leg (Anchor A -> Anchor B) from today's RTH session.
        2. Computes Dynamic Fibonacci Retracements (0.382, 0.500, 0.618, 0.786) & Extensions (1.272, 1.618).
        3. Evaluates Tier 1 (0.382-0.500 Reload), Tier 2 (0.500-0.618 Golden Pocket + Confluence),
           and Tier 3 (0.000 Wave-3 Impulse Continuation Breakout).
        """
        side = sig.signal_position  # "BUY" or "SELL"
        today_rth = df_5m[(df_5m["session_date"] == today) & (df_5m["bar_time"] >= time(9, 30))]
        if len(today_rth) < 4:
            return None

        # If signal arrived intraday (> 09:30), slice bars starting from signal arrival
        created_at = getattr(sig, "created_at", None)
        sig_date = getattr(sig, "signal_date", None)
        if created_at is not None and str(sig_date) == str(today) and created_at.time() > time(9, 30):
            catalyst_bars = today_rth[today_rth["bar_time"] >= created_at.time()]
            if len(catalyst_bars) < 3:
                return None
        else:
            catalyst_bars = today_rth

        last_bar = catalyst_bars.iloc[-1]
        prev_bar = catalyst_bars.iloc[-2]
        bar_open = float(last_bar["open"])
        bar_close = float(last_bar["close"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        vwap_val = float(last_bar["vwap"])
        ema9_val = float(last_bar["ema9"])
        ema20_val = float(last_bar["ema20"])
        vol_ma = float(last_bar["vol_ma"]) if pd.notna(last_bar["vol_ma"]) and last_bar["vol_ma"] > 0 else 1.0
        rvol = float(last_bar["volume"]) / vol_ma

        # Candle reversal / momentum confirmation
        is_bullish_trigger = (bar_close > bar_open) and (bar_close >= float(prev_bar["close"]))
        is_bearish_trigger = (bar_close < bar_open) and (bar_close <= float(prev_bar["close"]))

        atr_buf = atr_15m * float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 0.5))

        # ── LONG CATALYST FIBONACCI (BUY) ────────────────────────────────────
        if side == "BUY":
            # Find Impulse Peak (Anchor B) and Impulse Origin (Anchor A) prior to or at the peak
            peak_idx = catalyst_bars["high"].idxmax()
            bars_up_to_peak = catalyst_bars.loc[:peak_idx]
            bars_after_peak = catalyst_bars.loc[peak_idx:]

            session_low_before_peak = float(bars_up_to_peak["low"].min())
            anchor_b = float(catalyst_bars.loc[peak_idx, "high"])

            # Include Gap Origin (PDC) if stock gapped up cleanly and held above PDC
            if self.prev_day_close is not None and session_low_before_peak > self.prev_day_close:
                # Cap effective gap origin to at most 1.5x ATR below session low so Fib grid stays realistic
                anchor_a = max(self.prev_day_close, session_low_before_peak - (1.0 * atr_15m))
            else:
                anchor_a = session_low_before_peak

            impulse_size = anchor_b - anchor_a
            if impulse_size < (self.min_impulse_atr * atr_15m):
                return None  # Impulse wave not yet large enough to anchor institutional Fibs

            # Fibonacci Retracement Levels of the Catalyst Impulse (B -> A)
            fib_236 = anchor_b - (0.236 * impulse_size)
            fib_382 = anchor_b - (0.382 * impulse_size)
            fib_500 = anchor_b - (0.500 * impulse_size)
            fib_618 = anchor_b - (0.618 * impulse_size)
            fib_786 = anchor_b - (0.786 * impulse_size)

            # Fibonacci Extension Targets (Measured Move above Anchor B)
            ext_1272 = anchor_b + ((self.ext_target_fib - 1.0) * impulse_size)
            ext_1618 = anchor_b + (0.618 * impulse_size)

            pullback_low = float(bars_after_peak["low"].min()) if len(bars_after_peak) > 1 else bar_low

            # 0.786 Invalidation Guard: If pullback collapsed below 0.786 Fib, catalyst failed!
            if pullback_low < fib_786 or current_price < fib_786:
                self.thesis_invalidated_today = True
                return None

            # Check volume behavior: pullback volume should not exceed peak impulse volume
            impulse_avg_vol = float(bars_up_to_peak["volume"].mean())
            healthy_volume = (float(last_bar["volume"]) <= impulse_avg_vol * 1.35) or is_bullish_trigger

            # Check that we have at least 1 pullback bar after the peak
            if len(bars_after_peak) >= 2 and is_bullish_trigger and healthy_volume:
                # TIER 1: 0.382 - 0.500 Institutional Shallow Reload (Above VWAP or EMA20)
                tested_shallow_zone = ( fib_500 * 0.998 <= pullback_low <= fib_236 * 1.001 )
                in_entry_window = (fib_500 <= current_price <= anchor_b)
                if tested_shallow_zone and in_entry_window and (current_price >= vwap_val or current_price >= ema20_val):
                    sl = min(pullback_low, fib_618) - atr_buf
                    tp = ext_1272 if (ext_1272 - current_price) / max(0.01, current_price - sl) >= self.min_rr else ext_1618
                    if (tp - current_price) / max(0.01, current_price - sl) >= self.min_rr:
                        return {
                            "side": "BUY",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "support": fib_382,
                            "resistance": tp,
                            "impulse_peak": anchor_b,
                            "reason": f"CIGC_TIER1_FIB_0.382_RELOAD (Imp=${impulse_size:.2f})"
                        }

                # TIER 2: 0.500 - 0.618 Golden Pocket + MTF S/R or VWAP Confluence
                tested_golden_pocket = (fib_786 < pullback_low <= fib_500 * 1.002)
                reclaimed_pocket = (fib_618 <= current_price <= fib_236) and (bar_close >= ema9_val)
                has_conf, conf_tag = self._has_mtf_confluence(fib_618, vwap_val)
                if tested_golden_pocket and reclaimed_pocket and (has_conf or sig.selector_id == 1):
                    sl = min(pullback_low, fib_786) - atr_buf
                    tp = ext_1272 if (ext_1272 - current_price) / max(0.01, current_price - sl) >= self.min_rr else ext_1618
                    if (tp - current_price) / max(0.01, current_price - sl) >= self.min_rr:
                        return {
                            "side": "BUY",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "support": fib_618,
                            "resistance": tp,
                            "impulse_peak": anchor_b,
                            "reason": f"CIGC_TIER2_GOLDEN_POCKET_0.618 ({conf_tag})"
                        }

        # ── SHORT CATALYST FIBONACCI (SELL) ──────────────────────────────────
        elif side == "SELL":
            trough_idx = catalyst_bars["low"].idxmin()
            bars_down_to_trough = catalyst_bars.loc[:trough_idx]
            bars_after_trough = catalyst_bars.loc[trough_idx:]

            session_high_before_trough = float(bars_down_to_trough["high"].max())
            anchor_b = float(catalyst_bars.loc[trough_idx, "low"])

            if self.prev_day_close is not None and session_high_before_trough < self.prev_day_close:
                anchor_a = min(self.prev_day_close, session_high_before_trough + (1.0 * atr_15m))
            else:
                anchor_a = session_high_before_trough

            impulse_size = anchor_a - anchor_b
            if impulse_size < (self.min_impulse_atr * atr_15m):
                return None

            fib_236 = anchor_b + (0.236 * impulse_size)
            fib_382 = anchor_b + (0.382 * impulse_size)
            fib_500 = anchor_b + (0.500 * impulse_size)
            fib_618 = anchor_b + (0.618 * impulse_size)
            fib_786 = anchor_b + (0.786 * impulse_size)

            ext_1272 = anchor_b - ((self.ext_target_fib - 1.0) * impulse_size)
            ext_1618 = anchor_b - (0.618 * impulse_size)

            pullback_high = float(bars_after_trough["high"].max()) if len(bars_after_trough) > 1 else bar_high

            if pullback_high > fib_786 or current_price > fib_786:
                self.thesis_invalidated_today = True
                return None

            impulse_avg_vol = float(bars_down_to_trough["volume"].mean())
            healthy_volume = (float(last_bar["volume"]) <= impulse_avg_vol * 1.35) or is_bearish_trigger

            if len(bars_after_trough) >= 2 and is_bearish_trigger and healthy_volume:
                # TIER 1: 0.382 - 0.500 Shallow Bearish Reload (Below VWAP or EMA20)
                tested_shallow_zone = (fib_236 * 0.999 <= pullback_high <= fib_500 * 1.002)
                in_entry_window = (anchor_b <= current_price <= fib_500)
                if tested_shallow_zone and in_entry_window and (current_price <= vwap_val or current_price <= ema20_val):
                    sl = max(pullback_high, fib_618) + atr_buf
                    tp = ext_1272 if (current_price - ext_1272) / max(0.01, sl - current_price) >= self.min_rr else ext_1618
                    if (current_price - tp) / max(0.01, sl - current_price) >= self.min_rr:
                        return {
                            "side": "SELL",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "support": tp,
                            "resistance": fib_382,
                            "impulse_peak": anchor_b,
                            "reason": f"CIGC_TIER1_SHORT_FIB_0.382 (Imp=${impulse_size:.2f})"
                        }

                # TIER 2: 0.500 - 0.618 Bearish Golden Pocket + Confluence
                tested_golden_pocket = (fib_500 * 0.998 <= pullback_high < fib_786)
                reclaimed_pocket = (fib_236 <= current_price <= fib_618) and (bar_close <= ema9_val)
                has_conf, conf_tag = self._has_mtf_confluence(fib_618, vwap_val)
                if tested_golden_pocket and reclaimed_pocket and (has_conf or sig.selector_id == 1):
                    sl = max(pullback_high, fib_786) + atr_buf
                    tp = ext_1272 if (current_price - ext_1272) / max(0.01, sl - current_price) >= self.min_rr else ext_1618
                    if (current_price - tp) / max(0.01, sl - current_price) >= self.min_rr:
                        return {
                            "side": "SELL",
                            "entry": current_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "support": tp,
                            "resistance": fib_618,
                            "impulse_peak": anchor_b,
                            "reason": f"CIGC_TIER2_SHORT_GOLDEN_POCKET ({conf_tag})"
                        }

        # ── TIER 3 FALLBACK: HIGHER-TIMEFRAME (1D / 1H) MACRO FIB + VWAP RECLAIM ─
        if not self.htf_fib_setups.empty:
            for _, setup in self.htf_fib_setups.iterrows():
                trend = str(setup["trend"]).lower()
                setup_side = "BUY" if trend == "uptrend" else "SELL"
                if setup_side != side:
                    continue

                fib_px = float(setup["entry_price"])
                if abs(current_price - fib_px) / max(0.01, fib_px) <= self.conf_tol:
                    if side == "BUY" and is_bullish_trigger and current_price >= vwap_val:
                        sl = min(float(setup["stop_loss"]), current_price - (1.2 * atr_15m))
                        tp = float(setup["take_profit"])
                        if (tp - current_price) / max(0.01, current_price - sl) >= self.min_rr:
                            return {
                                "side": "BUY",
                                "entry": current_price,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "support": fib_px,
                                "resistance": tp,
                                "impulse_peak": current_price + atr_15m,
                                "reason": "HTF_MACRO_FIB_0.618_VWAP_ALIGNED"
                            }
                    elif side == "SELL" and is_bearish_trigger and current_price <= vwap_val:
                        sl = max(float(setup["stop_loss"]), current_price + (1.2 * atr_15m))
                        tp = float(setup["take_profit"])
                        if (current_price - tp) / max(0.01, sl - current_price) >= self.min_rr:
                            return {
                                "side": "SELL",
                                "entry": current_price,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "support": tp,
                                "resistance": fib_px,
                                "impulse_peak": current_price - atr_15m,
                                "reason": "HTF_MACRO_SHORT_FIB_0.618_VWAP_ALIGNED"
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

        today = current_dt.date()
        if self.current_session_date != today:
            self._reset_session(today)

        # ── 1. SYNC BROKER BRACKET EXITS & MANAGE WAVE-3 RATCHET ─────────────
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
                    self.log_message(f"[{ticker_sym}] CIGC Fib Bracket {reason} filled @ ~${fill_px:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, fill_px, reason)
            self.active_trade_id = None
            self.wave3_ratchet_applied = False

        if has_open_pos and self.active_trade_id in self.trade_tracker.trades:
            trade = self.trade_tracker.trades[self.active_trade_id]
            pnl_per_share = (current_price - trade.entry_price) if trade.trade_type == "BUY" else (trade.entry_price - current_price)
            r_mult = (pnl_per_share / self.active_initial_risk) if self.active_initial_risk > 0 else 0.0

            # EOD Guard for Counter-Fundamental (Selector 2) or weak Selector 3 trades
            if current_dt.time() >= time(15, 50):
                if self.active_selector_id == 2 or (self.active_selector_id == 3 and r_mult < 0.5):
                    self.log_message(f"[{ticker_sym}] EOD Guard Flatten @ ${current_price:.2f} ({r_mult:+.2f}R)")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "EOD_GUARD")
                    self._close_position()
                    return

            # Wave-3 Confirmation Ratchet: When price re-breaks Impulse Peak (or reaches +1.0R), lock in +0.2R
            reached_impulse_peak = (
                self.active_impulse_peak is not None and (
                    (trade.trade_type == "BUY" and current_price >= self.active_impulse_peak) or
                    (trade.trade_type == "SELL" and current_price <= self.active_impulse_peak)
                )
            )
            if not self.wave3_ratchet_applied and (reached_impulse_peak or r_mult >= 1.1):
                cushion = 0.20 * self.active_initial_risk
                trade.stop_loss = (trade.entry_price + cushion) if trade.trade_type == "BUY" else (trade.entry_price - cushion)
                self.wave3_ratchet_applied = True
                self.log_message(
                    f"[{ticker_sym}] Wave-3 Impulse Peak Reclaimed -> Stop ratcheted to +0.2R (${trade.stop_loss:.2f})"
                )

            if self.wave3_ratchet_applied:
                if (trade.trade_type == "BUY" and current_price <= trade.stop_loss) or \
                   (trade.trade_type == "SELL" and current_price >= trade.stop_loss):
                    self.log_message(f"[{ticker_sym}] Wave-3 Protected Stop Hit @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "WAVE3_BE_STOP")
                    self._close_position()
            return

        # ── 2. ENTRY GATING (MAX 1 TRADE/DAY, NO INVALIDATED THESIS) ─────────
        if self.trades_taken_today >= 1 or self.thesis_invalidated_today:
            return

        current_time = current_dt.time()
        # Allow 15 minutes (9:30-9:45) for the initial Catalyst Impulse Wave to form
        if current_time < time(9, 45) or current_time > time(15, 30):
            return

        sig = self._get_active_signal(current_dt)
        if sig is None or not sig.is_eligible:
            return

        bars_5m = self.get_historical_prices(ticker, 150, "5M")
        if bars_5m is None or len(bars_5m.df) < 25:
            return

        if self.last_analysis_date != today or self.merged_levels.empty:
            self._refresh_mtf_structure(today)

        df_5m = self._compute_intraday_features(bars_5m.df)
        atr_15m = self._get_15m_atr(current_price)

        chosen_setup = self._evaluate_catalyst_fibonacci(
            sig=sig,
            df_5m=df_5m,
            today=today,
            current_price=current_price,
            atr_15m=atr_15m
        )

        if chosen_setup is None:
            return

        # ── 3. OPTIMIZE SL/TP & EXECUTE CONVICTION-WEIGHTED BRACKET ──────────
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

        portfolio_val = self.get_portfolio_value()
        comp_conv = getattr(sig, "composite_conviction", sqrt(sig.fundamental_confidence * sig.news_conviction))
        conv_scale = 0.75 + 0.50 * max(0.0, min(1.0, (comp_conv - 0.70) / 0.30))
        risk_budget = portfolio_val * self.risk_pct * conv_scale

        effective_risk_per_share = max(abs(current_price - final_sl), 0.8 * atr_15m, current_price * 0.008)
        qty = floor(risk_budget / effective_risk_per_share)
        max_shares = floor((portfolio_val * 0.50) / current_price)
        quantity = max(0, min(qty, max_shares))

        if quantity <= 0:
            return

        self.log_message(
            f"[{ticker_sym}] CIGC FIB TRIGGERED: {chosen_setup['reason']} | "
            f"Side: {chosen_setup['side']} | Qty: {quantity} @ ${current_price:.2f} | "
            f"SL: ${final_sl:.2f} | TP (Fib Ext): ${final_tp:.2f} | R:R: {risk_levels.risk_reward_ratio:.2f}"
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
        self.active_impulse_peak = chosen_setup.get("impulse_peak")
        self.active_selector_id = sig.selector_id
        self.wave3_ratchet_applied = False
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
        self.log_message(f"[{ticker.symbol}] CIGC News+Fib Performance Summary: {perf}")

    def _close_position(self):
        self.cancel_open_orders()
        ticker = self.parameters["Ticker"]
        pos = self.get_position(ticker)
        if pos is not None and pos.quantity != 0:
            side = "sell" if pos.quantity > 0 else "buy"
            order = self.create_order(ticker, abs(pos.quantity), side, type="market")
            self.submit_order(order)
        self.active_trade_id = None
        self.wave3_ratchet_applied = False
