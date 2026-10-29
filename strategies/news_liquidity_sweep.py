"""
News + Liquidity Sweep Strategy (news_liquidity_sweep.py)
=========================================================
Combines Stock Alchemist intelligence news/fundamental signals with
institutional liquidity sweeps (PDH/PDL, session extremes, and structure reclaim).

Architecture:
1. Signal Gating:
   - Evaluates signals via Dual Hard Floor (Fund Conf >= 70% AND News Conv >= 70%).
   - Establishes directional bias: only allows liquidity sweeps aligned with the institutional narrative.
2. Technical Execution:
   - Long Setup: Price sweeps liquidity below PDL / Session Low with a volume spike,
     reclaims back inside within <= 3 bars, confirms via structure/VWAP, and enters on pullback.
   - Short Setup: Price sweeps liquidity above PDH / Session High with volume,
     reclaims back below within <= 3 bars, confirms via structure/VWAP, and enters on pullback.
3. Risk & Position Management:
   - Stop Loss placed at the sweep extreme wick + 0.5x ATR buffer.
   - Take Profit set to the opposing liquidity pool (minimum 1.5:1 Risk/Reward).
   - Position sizing: 2% portfolio risk ($2,000 per trade on $100k capital).
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
from strategies.risk_management import optimize_sl_tp, calculate_atr
from strategies.trade_tracker import TradeTracker


class NewsLiquiditySweepStrategy(StrategyBaseplate):
    """
    Combines news directional bias with institutional liquidity sweep & structure reclaim execution.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "TradingStyle": "swing_trading",
        "RiskPct": 0.02,                      # 2% portfolio risk budget
        "MinRiskReward": 1.5,                 # Minimum 1.5:1 Risk/Reward
        "MaxReclaimCandles": 3,               # Max candles outside level
        "VolumeMAPeriod": 20,                 # Period for volume spike
        "ATR_Length": 14,
        "ATR_Stop_Buffer_Multiplier": 0.5,
        "AllowOvernight": True,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.risk_pct = float(self.parameters.get("RiskPct", 0.02))
        self.min_rr = float(self.parameters.get("MinRiskReward", 1.5))
        self.max_reclaim = int(self.parameters.get("MaxReclaimCandles", 3))

        # Trade Tracker
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_trade_id = None
        self.matched_signal: Optional[ReactorSignal] = None

        # Sweep state tracking
        self.sweep_state: str = "IDLE"  # "IDLE", "SWEPT", "IN_POSITION"
        self.sweep_level: Optional[float] = None
        self.sweep_extreme: Optional[float] = None
        self.sweep_side: Optional[str] = None
        self.sweep_bar_count: int = 0

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
            f"[{ticker_sym}] NEWS SWEEP BIAS: {sig.signal_position} | "
            f"Fund: {sig.fundamental_sentiment} ({sig.fundamental_confidence:.1%}) | "
            f"News: {sig.news_sentiment} ({sig.news_conviction:.1%}) -> {status}"
        )

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        # Volume MA
        data["vol_ma"] = data["volume"].rolling(20).mean()

        # ATR(14)
        atr_len = int(self.parameters.get("ATR_Length", 14))
        data.ta.atr(length=atr_len, append=True)
        atr_cols = [c for c in data.columns if c.startswith("ATRr_")]
        data["atr"] = data[atr_cols[0]] if atr_cols else data["close"] * 0.015

        # Session dates & VWAP
        if isinstance(data.index, pd.DatetimeIndex):
            idx_ny = data.index.tz_convert("America/New_York") if data.index.tz else data.index.tz_localize("UTC").tz_convert("America/New_York")
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

    def _get_liquidity_levels(self, df: pd.DataFrame, today: datetime.date) -> Tuple[float, float]:
        """Calculates Previous Day High and Previous Day Low from historical bars."""
        prior = df[df["session_date"] < today]
        if not prior.empty:
            prior_dates = prior["session_date"].unique()
            if len(prior_dates) > 0:
                last_d = prior_dates[-1]
                last_bars = prior[prior["session_date"] == last_d]
                return float(last_bars["high"].max()), float(last_bars["low"].min())
        return float(df["high"].max()), float(df["low"].min())

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
                    self.log_message(f"[{ticker_sym}] LIQUIDITY SWEEP TP HIT @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "TP")
                    self._close_position()
                    self.active_trade_id = None
                    self.sweep_state = "IDLE"
                    return
                # SL check
                if (trade.trade_type == "BUY" and current_price <= trade.stop_loss) or \
                   (trade.trade_type == "SELL" and current_price >= trade.stop_loss):
                    self.log_message(f"[{ticker_sym}] LIQUIDITY SWEEP SL HIT @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "SL")
                    self._close_position()
                    self.active_trade_id = None
                    self.sweep_state = "IDLE"
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

        df = self._compute_indicators(bars.df)
        today = current_dt.date()
        pdh, pdl = self._get_liquidity_levels(df, today)

        last_bar = df.iloc[-1]
        atr_val = float(last_bar["atr"])
        atr_buf = atr_val * float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 0.5))
        vol_spike = last_bar["volume"] >= last_bar["vol_ma"]

        # ── STATE MACHINE: DETECT SWEEP & RECLAIM ─────────────────────────
        # FOR BUY: Sweep of Lows (PDL) followed by Reclaim
        if allowed_side == "BUY":
            if self.sweep_state == "IDLE":
                # Check if current bar wick swept below PDL
                if last_bar["low"] < pdl and vol_spike:
                    self.sweep_state = "SWEPT"
                    self.sweep_level = pdl
                    self.sweep_extreme = float(last_bar["low"])
                    self.sweep_side = "BUY"
                    self.sweep_bar_count = 1
                    return
            elif self.sweep_state == "SWEPT" and self.sweep_side == "BUY":
                self.sweep_bar_count += 1
                self.sweep_extreme = min(self.sweep_extreme, float(last_bar["low"]))

                # Check if price reclaimed back ABOVE PDL within max_reclaim bars
                if last_bar["close"] > pdl:
                    # RECLAIM CONFIRMED! Enter Long
                    entry_price = current_price
                    sl = self.sweep_extreme - atr_buf
                    tp = pdh  # Target opposing high
                    risk = entry_price - sl
                    reward = tp - entry_price

                    if risk > 0 and (reward / risk) >= self.min_rr:
                        # Position Sizing
                        portfolio_val = self.get_portfolio_value()
                        risk_budget = portfolio_val * self.risk_pct
                        qty = floor(risk_budget / max(0.01, risk))
                        max_shares = floor((portfolio_val * 0.95) / entry_price)
                        quantity = max(1, min(qty, max_shares))

                        self.log_message(
                            f"[{ticker_sym}] NEWS+SWEEP LONG TRIGGERED @ ${entry_price:.2f} | "
                            f"Swept PDL: ${pdl:.2f} (Extreme: ${self.sweep_extreme:.2f}) | "
                            f"SL: ${sl:.2f} | TP: ${tp:.2f} | Qty: {quantity}"
                        )

                        self.active_trade_id = self.trade_tracker.open_trade(
                            date=current_dt,
                            entry_price=entry_price,
                            quantity=quantity,
                            take_profit=tp,
                            stop_loss=sl,
                            support_level=self.sweep_extreme,
                            resistance_level=tp,
                            trade_type="BUY"
                        )

                        order = self.create_order(ticker, quantity, "buy", take_profit_price=tp, stop_loss_price=sl, type="market")
                        self.submit_order(order)
                        self.sweep_state = "IN_POSITION"
                        return

                if self.sweep_bar_count > self.max_reclaim:
                    self.sweep_state = "IDLE"  # Reclaim timed out

        # FOR SELL: Sweep of Highs (PDH) followed by Reclaim back below
        elif allowed_side == "SELL":
            if self.sweep_state == "IDLE":
                if last_bar["high"] > pdh and vol_spike:
                    self.sweep_state = "SWEPT"
                    self.sweep_level = pdh
                    self.sweep_extreme = float(last_bar["high"])
                    self.sweep_side = "SELL"
                    self.sweep_bar_count = 1
                    return
            elif self.sweep_state == "SWEPT" and self.sweep_side == "SELL":
                self.sweep_bar_count += 1
                self.sweep_extreme = max(self.sweep_extreme, float(last_bar["high"]))

                if last_bar["close"] < pdh:
                    # RECLAIM CONFIRMED! Enter Short
                    entry_price = current_price
                    sl = self.sweep_extreme + atr_buf
                    tp = pdl  # Target opposing low
                    risk = sl - entry_price
                    reward = entry_price - tp

                    if risk > 0 and (reward / risk) >= self.min_rr:
                        portfolio_val = self.get_portfolio_value()
                        risk_budget = portfolio_val * self.risk_pct
                        qty = floor(risk_budget / max(0.01, risk))
                        max_shares = floor((portfolio_val * 0.95) / entry_price)
                        quantity = max(1, min(qty, max_shares))

                        self.log_message(
                            f"[{ticker_sym}] NEWS+SWEEP SHORT TRIGGERED @ ${entry_price:.2f} | "
                            f"Swept PDH: ${pdh:.2f} (Extreme: ${self.sweep_extreme:.2f}) | "
                            f"SL: ${sl:.2f} | TP: ${tp:.2f} | Qty: {quantity}"
                        )

                        self.active_trade_id = self.trade_tracker.open_trade(
                            date=current_dt,
                            entry_price=entry_price,
                            quantity=quantity,
                            take_profit=tp,
                            stop_loss=sl,
                            support_level=tp,
                            resistance_level=self.sweep_extreme,
                            trade_type="SELL"
                        )

                        order = self.create_order(ticker, quantity, "sell", take_profit_price=tp, stop_loss_price=sl, type="market")
                        self.submit_order(order)
                        self.sweep_state = "IN_POSITION"
                        return

                if self.sweep_bar_count > self.max_reclaim:
                    self.sweep_state = "IDLE"

    def _close_position(self):
        ticker = self.parameters["Ticker"]
        pos = self.get_position(ticker)
        if pos is not None and pos.quantity != 0:
            side = "sell" if pos.quantity > 0 else "buy"
            order = self.create_order(ticker, abs(pos.quantity), side, type="market")
            self.submit_order(order)
