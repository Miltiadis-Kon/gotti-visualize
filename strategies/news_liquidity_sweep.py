"""
News + Liquidity Sweep Strategy (news_liquidity_sweep.py)
=========================================================
Combines Stock Alchemist intelligence news/fundamental signals with
institutional liquidity sweeps across Opening Range (OR_Low/OR_High),
Previous Day High/Low/Close (PDH/PDL/PDC), and VWAP structure reclaim.

Profitability Upgrades:
1. Multi-Pool Institutional Liquidity Engine:
   - Monitors 3 key liquidity pools:
     * Pool 1: 9:30-9:45 AM Opening Range Extremes (OR_Low for BUY, OR_High for SELL).
     * Pool 2: Previous Day Close (PDC / Gap-Fill level) & Breakout Retests.
     * Pool 3: True Daily PDH / PDL (fetched from 1D bars rather than truncated 5m bars).
2. True Sweep & Reclaim State Machine:
   - Verifies Pre-Sweep Structure (price was trading above support before sweeping it)
     to prevent false triggers during sustained breakdowns.
   - Captures BOTH 1-Bar Hammer/Pinbar Sweeps (wick sweeps level + closes back inside on
     the same 5m bar) AND Multi-Bar Sweeps (reclaims within <= MaxReclaimCandles).
   - Enforces VWAP / EMA-9 momentum confirmation on the reclaim bar.
3. Institutional Risk & Churn Controls:
   - Uses 15m ATR + StrategyBaseplate.optimize_sl_tp() to prevent tiny 5m stops & over-leverage.
   - Enforces 1 trade per session/signal (eliminates re-entry churn on broken levels).
   - Ratchets stop to Breakeven at +1.2R and flattens counter-fundamental trades before EOD.
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
from strategies.risk_management import optimize_sl_tp, calculate_atr
from strategies.trade_tracker import TradeTracker


class NewsLiquiditySweepStrategy(StrategyBaseplate):
    """
    Combines news directional bias with institutional multi-pool liquidity sweeps & VWAP reclaim.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "DynamicDateBinding": True,           # Auto-load signals as simulation date rolls over
        "EnforceNoLookahead": True,           # Respect signal created_at timestamp
        "TradingStyle": "swing_trading",
        "RiskPct": 0.02,                      # 2% portfolio risk budget
        "MinRiskReward": 1.6,                 # Minimum 1.6:1 Risk/Reward
        "MaxReclaimCandles": 3,               # Max 5m candles allowed outside level before invalidation
        "MinSweepRVOL": 1.15,                 # Minimum relative volume on sweep or reclaim bar
        "VolumeMAPeriod": 20,                 # Period for volume moving average
        "ATR_Length": 14,
        "ATR_Stop_Buffer_Multiplier": 0.8,    # Applied to 15m ATR beyond the sweep extreme wick
        "BreakevenTriggerR": 1.2,             # Ratchet stop to breakeven at +1.2R
        "AllowOvernight": True,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.dynamic_date_binding = bool(self.parameters.get("DynamicDateBinding", True))
        self.enforce_no_lookahead = bool(self.parameters.get("EnforceNoLookahead", True))
        self.risk_pct = float(self.parameters.get("RiskPct", 0.02))
        self.min_rr = float(self.parameters.get("MinRiskReward", 1.6))
        self.max_reclaim = int(self.parameters.get("MaxReclaimCandles", 3))
        self.min_sweep_rvol = float(self.parameters.get("MinSweepRVOL", 1.15))
        self.be_trigger_r = float(self.parameters.get("BreakevenTriggerR", 1.2))

        # Trade Tracker & Active Position State
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

        # Daily & Intraday Liquidity Pools
        self.pdh: Optional[float] = None
        self.pdl: Optional[float] = None
        self.pdc: Optional[float] = None
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None

        # Sweep State Machine
        self._reset_sweep_state()
        self.invalidated_levels: set = set()

        self._load_signals_for_date(self.signal_date_str)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. SESSION & SIGNAL MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_sweep_state(self):
        self.sweep_state: str = "IDLE"        # "IDLE", "SWEPT", "IN_POSITION"
        self.sweep_level: Optional[float] = None
        self.sweep_pool_name: str = ""
        self.sweep_extreme: Optional[float] = None
        self.sweep_side: Optional[str] = None
        self.sweep_bar_count: int = 0

    def _reset_daily_session(self, new_date: datetime.date):
        self.current_session_date = new_date
        self.trades_taken_today = 0
        self.pdh = None
        self.pdl = None
        self.pdc = None
        self.or_high = None
        self.or_low = None
        self.invalidated_levels.clear()
        self._reset_sweep_state()

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
            f"[{ticker_sym}] NEWS SWEEP BIAS ({target_date_str}): {sig.signal_position} | "
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
    # 2. INDICATORS & MULTI-POOL LIQUIDITY LEVELS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        vol_period = int(self.parameters.get("VolumeMAPeriod", 20))
        data["vol_ma"] = data["volume"].rolling(vol_period, min_periods=5).mean()
        data["ema9"] = data["close"].ewm(span=9, adjust=False).mean()

        # Session dates & Intraday RTH VWAP
        if isinstance(data.index, pd.DatetimeIndex):
            idx_ny = (
                data.index.tz_convert("America/New_York")
                if data.index.tz
                else data.index.tz_localize("UTC").tz_convert("America/New_York")
            )
            data["session_date"] = idx_ny.date
            times = idx_ny.time
        else:
            data["session_date"] = [datetime.date.today()] * len(data)
            times = [datetime.time(9, 30)] * len(data)

        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        data["pv"] = data["typical_price"] * data["volume"]
        rth_pv = np.where(times >= datetime.time(9, 30), data["pv"], 0.0)
        rth_vol = np.where(times >= datetime.time(9, 30), data["volume"], 0.0)
        data["cum_pv"] = pd.Series(rth_pv, index=data.index).groupby(data["session_date"]).cumsum()
        data["cum_vol"] = pd.Series(rth_vol, index=data.index).groupby(data["session_date"]).cumsum()
        data["vwap"] = (data["cum_pv"] / data["cum_vol"].replace(0, np.nan)).ffill().bfill()

        return data

    def _get_15m_atr(self, fallback_price: float) -> float:
        """Computes 14-period ATR on 15m bars so stops are sized for institutional structure."""
        ticker = self.parameters["Ticker"]
        try:
            bars_15m = self.get_historical_prices(ticker, 35, "15m")
            if bars_15m is not None and len(bars_15m.df) >= 15:
                return float(calculate_atr(bars_15m.df, length=14))
        except Exception:
            pass
        return fallback_price * 0.012

    def _update_liquidity_pools(self, df_5m: pd.DataFrame, today: datetime.date):
        """
        Populates True Previous Day High/Low/Close (from Daily bars)
        and the 9:30-9:45 AM Opening Range High/Low (from 5m bars).
        """
        ticker = self.parameters["Ticker"]

        # 1. Fetch true PDH, PDL, PDC from Daily bars once per session
        if self.pdh is None or self.pdl is None:
            try:
                daily_bars = self.get_historical_prices(ticker, 10, "day")
                if daily_bars is not None and len(daily_bars.df) >= 2:
                    d_df = daily_bars.df
                    prev_row = d_df.iloc[-2]
                    self.pdh = float(prev_row["high"])
                    self.pdl = float(prev_row["low"])
                    self.pdc = float(prev_row["close"])
            except Exception:
                pass

            # Fallback to prior session in 5m df if daily bars unavailable
            if self.pdh is None and "session_date" in df_5m.columns:
                prior = df_5m[df_5m["session_date"] < today]
                if not prior.empty:
                    last_d = prior["session_date"].unique()[-1]
                    last_bars = prior[prior["session_date"] == last_d]
                    self.pdh = float(last_bars["high"].max())
                    self.pdl = float(last_bars["low"].min())
                    self.pdc = float(last_bars["close"].iloc[-1])

        # 2. Compute 9:30 - 9:45 AM Opening Range High/Low
        if self.or_high is None or self.or_low is None:
            try:
                idx = pd.to_datetime(df_5m.index)
                mask = (idx.date == today) & (idx.time >= time(9, 30)) & (idx.time < time(9, 45))
                or_bars = df_5m[mask]
                if not or_bars.empty:
                    self.or_high = float(or_bars["high"].max())
                    self.or_low = float(or_bars["low"].min())
            except Exception:
                pass

    def _get_candidate_pools(self, side: str) -> List[Tuple[str, float]]:
        """
        Returns ordered list of (pool_name, level_price) to monitor for liquidity sweeps.
        - For BUY: Downside stop pools (OR_Low, PDL, PDC, or PDH breakout retest).
        - For SELL: Upside stop pools (OR_High, PDH, PDC, or PDL breakdown retest).
        """
        pools: List[Tuple[str, float]] = []
        if side == "BUY":
            if self.or_low is not None:
                pools.append(("OR_LOW", self.or_low))
            if self.pdl is not None:
                pools.append(("PDL", self.pdl))
            if self.pdc is not None:
                pools.append(("PDC_GAP", self.pdc))
            if self.pdh is not None:
                pools.append(("PDH_FLIP", self.pdh))
        else:
            if self.or_high is not None:
                pools.append(("OR_HIGH", self.or_high))
            if self.pdh is not None:
                pools.append(("PDH", self.pdh))
            if self.pdc is not None:
                pools.append(("PDC_GAP", self.pdc))
            if self.pdl is not None:
                pools.append(("PDL_FLIP", self.pdl))

        # Deduplicate levels that are within 0.15% of each other
        unique_pools: List[Tuple[str, float]] = []
        for name, lvl in pools:
            if round(lvl, 2) in self.invalidated_levels:
                continue
            if not any(abs(lvl - existing_lvl) / max(0.01, existing_lvl) < 0.0015 for _, existing_lvl in unique_pools):
                unique_pools.append((name, lvl))
        return unique_pools

    def _get_opposing_target(self, side: str, entry_price: float, atr_15m: float, df_5m: pd.DataFrame, today: datetime.date) -> float:
        """
        Selects a realistic opposing liquidity pool for Take Profit.
        If price is already beyond PDH/PDL, projects a dynamic 2.2x 15m ATR target.
        """
        today_bars = df_5m[df_5m["session_date"] == today] if "session_date" in df_5m.columns else df_5m
        session_high = float(today_bars["high"].max()) if not today_bars.empty else entry_price + (2.0 * atr_15m)
        session_low = float(today_bars["low"].min()) if not today_bars.empty else entry_price - (2.0 * atr_15m)

        if side == "BUY":
            candidates = [
                lvl for lvl in [self.or_high, session_high, self.pdh, self.pdc]
                if lvl is not None and lvl > entry_price + (1.2 * atr_15m)
            ]
            return min(candidates) if candidates else entry_price + (2.2 * atr_15m)
        else:
            candidates = [
                lvl for lvl in [self.or_low, session_low, self.pdl, self.pdc]
                if lvl is not None and lvl < entry_price - (1.2 * atr_15m)
            ]
            return max(candidates) if candidates else entry_price - (2.2 * atr_15m)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. MAIN EXECUTION LOOP
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
            self._reset_daily_session(today)

        # ── 1. SYNC BROKER BRACKET EXITS & MANAGE OPEN POSITION ──────────────
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
                    self.log_message(f"[{ticker_sym}] Liquidity Sweep Bracket {reason} filled @ ~${fill_px:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, fill_px, reason)
            self.active_trade_id = None
            self.breakeven_applied = False
            self._reset_sweep_state()

        if has_open_pos and self.active_trade_id in self.trade_tracker.trades:
            trade = self.trade_tracker.trades[self.active_trade_id]
            pnl_per_share = (current_price - trade.entry_price) if trade.trade_type == "BUY" else (trade.entry_price - current_price)
            r_mult = (pnl_per_share / self.active_initial_risk) if self.active_initial_risk > 0 else 0.0

            # EOD Protection for Selector 2 (Counter-Fundamental) or weak Selector 3 trades
            if current_dt.time() >= time(15, 50):
                if self.active_selector_id == 2 or (self.active_selector_id == 3 and r_mult < 0.5):
                    self.log_message(f"[{ticker_sym}] EOD Sweep Guard Flatten @ ${current_price:.2f} ({r_mult:+.2f}R)")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "EOD_GUARD")
                    self._close_position()
                    return

            # Ratchet Stop Loss to Breakeven (+0.1R cushion) once trade reaches +1.2R
            if not self.breakeven_applied and r_mult >= self.be_trigger_r:
                cushion = 0.10 * self.active_initial_risk
                trade.stop_loss = (trade.entry_price + cushion) if trade.trade_type == "BUY" else (trade.entry_price - cushion)
                self.breakeven_applied = True
                self.log_message(f"[{ticker_sym}] Sweep reached +{r_mult:.2f}R -> Stop ratcheted to ${trade.stop_loss:.2f}")

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

        allowed_side = sig.signal_position  # "BUY" or "SELL"

        # Fetch 5m bars & compute VWAP, EMA9, Volume MA, and Liquidity Pools
        bars = self.get_historical_prices(ticker, 150, "5M")
        if bars is None or len(bars.df) < 25:
            return

        df = self._compute_indicators(bars.df)
        self._update_liquidity_pools(df, today)
        atr_15m = self._get_15m_atr(current_price)
        atr_buf = atr_15m * float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 0.8))

        last_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        prev2_bar = df.iloc[-3]

        bar_open = float(last_bar["open"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        bar_close = float(last_bar["close"])
        bar_vwap = float(last_bar["vwap"])
        bar_ema9 = float(last_bar["ema9"])
        vol_ma = float(last_bar["vol_ma"]) if pd.notna(last_bar["vol_ma"]) and last_bar["vol_ma"] > 0 else 1.0
        rvol = float(last_bar["volume"]) / vol_ma
        vol_spike = rvol >= self.min_sweep_rvol

        # Candle structure checks
        candle_range = max(0.0001, bar_high - bar_low)
        lower_wick_ratio = (bar_close - bar_low) / candle_range
        upper_wick_ratio = (bar_high - bar_close) / candle_range
        is_bullish_reclaim_bar = (bar_close > bar_open) or (lower_wick_ratio >= 0.55)
        is_bearish_reclaim_bar = (bar_close < bar_open) or (upper_wick_ratio >= 0.55)

        # VWAP / Short-Term Structure Alignment
        bullish_structure_ok = (bar_close >= bar_vwap) or (bar_close >= bar_ema9 and bar_close > float(prev_bar["high"]))
        bearish_structure_ok = (bar_close <= bar_vwap) or (bar_close <= bar_ema9 and bar_close < float(prev_bar["low"]))

        reclaim_triggered = False

        # ── 3. SWEEP & RECLAIM STATE MACHINE (1-BAR PINBAR + MULTI-BAR) ──────
        if allowed_side == "BUY":
            if self.sweep_state == "IDLE":
                for pool_name, lvl in self._get_candidate_pools("BUY"):
                    # Verify Pre-Sweep Structure: at least one of the prior 2 bars closed >= lvl
                    was_above_level = (float(prev_bar["close"]) >= lvl) or (float(prev2_bar["close"]) >= lvl)
                    # Ensure wick did not crash > 1.5x 15m ATR below level (true breakdown, not a sweep)
                    valid_sweep_depth = (lvl - bar_low) <= (1.5 * atr_15m)

                    if was_above_level and bar_low < lvl and valid_sweep_depth and vol_spike:
                        self.sweep_level = lvl
                        self.sweep_pool_name = pool_name
                        self.sweep_extreme = bar_low
                        self.sweep_side = "BUY"
                        self.sweep_bar_count = 1

                        # Case A: Immediate 1-Bar Pinbar Sweep & Reclaim!
                        if bar_close > lvl and is_bullish_reclaim_bar and bullish_structure_ok:
                            reclaim_triggered = True
                        else:
                            self.sweep_state = "SWEPT"
                        break

            elif self.sweep_state == "SWEPT" and self.sweep_side == "BUY":
                self.sweep_bar_count += 1
                self.sweep_extreme = min(float(self.sweep_extreme), bar_low)

                # Invalidate if price cascades > 1.5x 15m ATR below the level
                if (self.sweep_level - self.sweep_extreme) > (1.5 * atr_15m):
                    self.invalidated_levels.add(round(self.sweep_level, 2))
                    self._reset_sweep_state()
                    return

                # Case B: Multi-Bar Reclaim back above swept level
                if bar_close > self.sweep_level and is_bullish_reclaim_bar and bullish_structure_ok:
                    reclaim_triggered = True
                elif self.sweep_bar_count > self.max_reclaim:
                    self.invalidated_levels.add(round(self.sweep_level, 2))
                    self._reset_sweep_state()
                    return

        elif allowed_side == "SELL":
            if self.sweep_state == "IDLE":
                for pool_name, lvl in self._get_candidate_pools("SELL"):
                    was_below_level = (float(prev_bar["close"]) <= lvl) or (float(prev2_bar["close"]) <= lvl)
                    valid_sweep_depth = (bar_high - lvl) <= (1.5 * atr_15m)

                    if was_below_level and bar_high > lvl and valid_sweep_depth and vol_spike:
                        self.sweep_level = lvl
                        self.sweep_pool_name = pool_name
                        self.sweep_extreme = bar_high
                        self.sweep_side = "SELL"
                        self.sweep_bar_count = 1

                        # Case A: Immediate 1-Bar Shooting Star Sweep & Reclaim!
                        if bar_close < lvl and is_bearish_reclaim_bar and bearish_structure_ok:
                            reclaim_triggered = True
                        else:
                            self.sweep_state = "SWEPT"
                        break

            elif self.sweep_state == "SWEPT" and self.sweep_side == "SELL":
                self.sweep_bar_count += 1
                self.sweep_extreme = max(float(self.sweep_extreme), bar_high)

                if (self.sweep_extreme - self.sweep_level) > (1.5 * atr_15m):
                    self.invalidated_levels.add(round(self.sweep_level, 2))
                    self._reset_sweep_state()
                    return

                # Case B: Multi-Bar Reclaim back below swept level
                if bar_close < self.sweep_level and is_bearish_reclaim_bar and bearish_structure_ok:
                    reclaim_triggered = True
                elif self.sweep_bar_count > self.max_reclaim:
                    self.invalidated_levels.add(round(self.sweep_level, 2))
                    self._reset_sweep_state()
                    return

        if not reclaim_triggered or self.sweep_extreme is None or self.sweep_level is None:
            return

        # ── 4. OPTIMIZE SL/TP & EXECUTE BRACKET ORDER ────────────────────────
        raw_sl = (self.sweep_extreme - atr_buf) if allowed_side == "BUY" else (self.sweep_extreme + atr_buf)
        raw_tp = self._get_opposing_target(allowed_side, current_price, atr_15m, df, today)
        order_side = "buy" if allowed_side == "BUY" else "sell"
        style = getattr(sig, "trading_style", self.parameters.get("TradingStyle", "swing_trading"))

        risk_levels = self.optimize_sl_tp(
            entry_price=current_price,
            stop_loss=raw_sl,
            take_profit=raw_tp,
            side=order_side,
            trading_style=style,
            atr=atr_15m,
            risk_reward_ratio=self.min_rr
        )

        final_sl = risk_levels.stop_loss
        final_tp = risk_levels.take_profit

        # Conviction-scaled sizing with 50% max single-position cap
        portfolio_val = self.get_portfolio_value()
        comp_conv = getattr(sig, "composite_conviction", sqrt(sig.fundamental_confidence * sig.news_conviction))
        conv_scale = 0.75 + 0.50 * max(0.0, min(1.0, (comp_conv - 0.70) / 0.30))
        risk_budget = portfolio_val * self.risk_pct * conv_scale

        effective_risk_per_share = max(abs(current_price - final_sl), 0.8 * atr_15m, current_price * 0.008)
        qty = floor(risk_budget / effective_risk_per_share)
        max_shares = floor((portfolio_val * 0.50) / current_price)
        quantity = max(0, min(qty, max_shares))

        if quantity <= 0:
            self._reset_sweep_state()
            return

        self.log_message(
            f"[{ticker_sym}] NEWS+SWEEP {allowed_side} CONFIRMED ({self.sweep_pool_name} @ ${self.sweep_level:.2f}) | "
            f"Extreme: ${self.sweep_extreme:.2f} | Entry: ${current_price:.2f} | "
            f"SL: ${final_sl:.2f} | TP: ${final_tp:.2f} | R:R: {risk_levels.risk_reward_ratio:.2f} | Qty: {quantity}"
        )

        self.active_trade_id = self.trade_tracker.open_trade(
            date=current_dt,
            entry_price=current_price,
            quantity=quantity,
            take_profit=final_tp,
            stop_loss=final_sl,
            support_level=self.sweep_extreme if allowed_side == "BUY" else final_tp,
            resistance_level=final_tp if allowed_side == "BUY" else self.sweep_extreme,
            trade_type=allowed_side
        )
        self.active_initial_risk = abs(current_price - final_sl)
        self.active_selector_id = sig.selector_id
        self.breakeven_applied = False
        self.trades_taken_today += 1
        self.traded_signal_ids.add(sig.raw_signal_id)
        self.sweep_state = "IN_POSITION"

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
        self.log_message(f"[{ticker.symbol}] News+LiquiditySweep Performance Summary: {perf}")

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
        self._reset_sweep_state()
