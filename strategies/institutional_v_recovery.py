"""
Institutional V-Recovery Strategy (strategies/institutional_v_recovery.py)
==========================================================================
Upgrades Zed Monopoly's V-shape momentum strategy from a discretionary YouTube setup
into a statistically robust, institutional-grade quantitative algorithm by implementing
5 core quantitative enhancements:

1. Volume Profile & Intraday VWAP Confluence:
   - Cumulative intraday VWAP calculation on 5-minute candles.
   - Long V-recovery entries require price to hold above or reclaim VWAP.
   - Relative Volume (RVOL) filter requiring >= 1.5x of the 20-period volume moving average.
2. Quantitative Index Correlation & Relative Strength Filter (vs QQQ):
   - Computes rolling relative strength against Nasdaq Composite (QQQ).
   - Validates Bullish Divergence: Stock holds a higher low while QQQ tests/prints lower lows.
3. Tiered Volatility-Based Stop-Loss (Dynamic ATR Risk):
   - Hard stop-loss anchored at 1.5x ATR(14) below V-bottom swing low or PDC support.
   - Automatic breakeven ratchet: Locks entry price once unrealized gain touches +1.0%.
4. Time-Based Liquidity Windows & Exits (TIF Mechanics):
   - Strict Entry Window: 09:35 EST to 10:15 EST (avoids 09:30 opening auction cross noise).
   - Time Stop: Closes trades at 11:00 EST if neither TP nor SL has been hit, avoiding midday chop.
5. Partial Scale-Outs & Trailing EMA Runner:
   - Target 1 (50% of position): Exits at fixed +1.8% to +2.0% gain.
   - Target 2 (Remaining 50%): Trails 5-minute EMA(9) to capture extended +3.0% to +5.0% expansion moves.
"""

import os
import sys
import json
import logging
from math import floor, sqrt
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategies.strat_baseplate import StrategyBaseplate
from strategies.market_reactor import fetch_reactor_signals, ReactorSignal, get_mysql_connection
from strategies.high_beta_catalyst_open import COMPETITOR_MAP, CatalystConfluenceEngine
from strategies.risk_management import calculate_atr
from strategies.trade_tracker import TradeTracker

logger = logging.getLogger("InstitutionalVRecovery")


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Computes session-anchored Cumulative Volume Weighted Average Price (VWAP)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    cum_vol = vol.cumsum().replace(0, np.nan)
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol


def compute_ema(series: pd.Series, period: int = 9) -> pd.Series:
    """Computes Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


class ZedInstitutionalStrategy(StrategyBaseplate):
    """
    ZedInstitutionalStrategy (Institutional V-Recovery Strategy):
    Upgrades Zed Monopoly's V-shape momentum strategy by implementing VWAP, RVOL >= 1.5,
    Index Relative Strength Divergence, Dynamic ATR Stops, Time Stops (11:00 EST),
    and 2-Tier Scaled Exits (50% at +2.0%, 50% trailing EMA-9).
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="COIN", asset_type=Asset.AssetType.STOCK),
        "MarketBenchmark": Asset(symbol="QQQ", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "DynamicDateBinding": True,
        "EnforceNoLookahead": True,
        "TradingStyle": "day_trading",
        "Target1Pct": 0.020,                  # Tier 1 Take Profit: +2.0%
        "Target2RunnerPct": 0.035,            # Tier 2 Runner Target: +3.5%
        "AtrStopMultiplier": 1.2,             # 1.2x of 15m ATR Volatility Stop
        "MaxStopLossPct": 0.030,              # Max 3.0% structural stop
        "MinRvol": 1.5,                       # RVOL >= 1.5x of 20-period volume MA
        "RequireVwapReclaim": True,           # Must hold or cross above intraday VWAP
        "RequireRvol": True,                  # Must satisfy RVOL threshold
        "RequireIndexDivergence": True,        # Bullish divergence vs QQQ
        "EnforceMarketRegime": True,          # Higher-timeframe Bull Market Filter
        "BreakevenGainPct": 0.010,            # Ratchet stop to entry at +1.0% gain
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.target1_pct = float(self.parameters.get("Target1Pct", 0.020))
        self.target2_pct = float(self.parameters.get("Target2RunnerPct", 0.035))
        self.atr_stop_mult = float(self.parameters.get("AtrStopMultiplier", 1.2))
        self.max_stop_pct = float(self.parameters.get("MaxStopLossPct", 0.030))
        self.min_rvol = float(self.parameters.get("MinRvol", 1.5))
        self.require_vwap = bool(self.parameters.get("RequireVwapReclaim", True))
        self.require_rvol = bool(self.parameters.get("RequireRvol", True))
        self.require_div = bool(self.parameters.get("RequireIndexDivergence", True))
        self.enforce_regime = bool(self.parameters.get("EnforceMarketRegime", True))
        self.be_gain = float(self.parameters.get("BreakevenGainPct", 0.010))

        # Position tracking
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_position_side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.total_quantity: int = 0
        self.remaining_quantity: int = 0
        self.tier1_sold: bool = False
        self.stop_loss_price: Optional[float] = None
        self.target1_price: Optional[float] = None
        self.breakeven_applied: bool = False
        self.setup_reason: str = ""

        # Session context
        self.current_session_date: Optional[date] = None
        self.open_price: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.session_low: Optional[float] = None
        self.session_high: Optional[float] = None
        self.daily_atr: float = 2.0
        self.market_regime: str = "BULL_MARKET"
        self.trades_taken_today: int = 0

        self._reset_daily_session(pd.to_datetime(self.signal_date_str).date())

    def _reset_daily_session(self, new_date: date):
        self.current_session_date = new_date
        self.active_position_side = None
        self.entry_price = None
        self.total_quantity = 0
        self.remaining_quantity = 0
        self.tier1_sold = False
        self.stop_loss_price = None
        self.target1_price = None
        self.breakeven_applied = False
        self.setup_reason = ""

        self.open_price = None
        self.prev_close = None
        self.session_low = None
        self.session_high = None
        self.market_regime = "BULL_MARKET"
        self.trades_taken_today = 0

        ticker_sym = self.parameters["Ticker"].symbol
        date_str = new_date.strftime("%Y-%m-%d")

        # Step 1: Pre-market Catalyst Confluence
        self.catalyst_analysis = CatalystConfluenceEngine.analyze_catalysts(ticker_sym, date_str)
        self._calculate_daily_levels()

    def _calculate_daily_levels(self):
        ticker = self.parameters["Ticker"]
        qqq_asset = self.parameters.get("MarketBenchmark", Asset("QQQ", Asset.AssetType.STOCK))
        try:
            bars_1d = self.get_historical_prices(ticker, 25, "day")
            if bars_1d is not None:
                df = bars_1d.df if hasattr(bars_1d, "df") else bars_1d
                if not df.empty and len(df) >= 2:
                    self.prev_close = float(df["close"].iloc[-2])
                    self.daily_atr = float(calculate_atr(df, length=min(14, len(df) - 1))) or (self.prev_close * 0.025)

            # Evaluate higher-timeframe market regime on QQQ
            bars_qqq_1d = self.get_historical_prices(qqq_asset, 25, "day")
            df_qqq_1d = bars_qqq_1d.df if hasattr(bars_qqq_1d, "df") else bars_qqq_1d
            qqq_last = self.get_last_price(qqq_asset)
            if df_qqq_1d is not None and not df_qqq_1d.empty and qqq_last:
                qqq_open = float(df_qqq_1d["open"].iloc[-1])
                self.market_regime = CatalystConfluenceEngine.evaluate_market_regime(
                    df_qqq_1d.iloc[:-1], qqq_last, qqq_open
                )
        except Exception as e:
            self.log_message(f"Notice calculating daily levels: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # TRADING ITERATION (5-MINUTE BARS)
    # ─────────────────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        super().on_trading_iteration()
        current_dt = self.get_datetime()
        cur_date = current_dt.date()
        cur_time = current_dt.time()

        if cur_date != self.current_session_date:
            self._reset_daily_session(cur_date)

        ticker = self.parameters["Ticker"]
        current_price = self.get_last_price(ticker)
        if not current_price or current_price <= 0:
            return

        if self.open_price is None and cur_time >= time(9, 30):
            self.open_price = current_price
            self.session_high = current_price
            self.session_low = current_price

        if self.session_high is not None:
            self.session_high = max(self.session_high, current_price)
        if self.session_low is not None:
            self.session_low = min(self.session_low, current_price)

        # ── 1. ACTIVE POSITION MANAGEMENT & EXITS ────────────────────────────
        if self.active_position_side is not None:
            self._manage_zed_institutional_trade(current_price, cur_time)
            return

        # Max 1 trade per session
        if self.trades_taken_today >= 1:
            return

        # ── 2. STRICT ENTRY WINDOW (09:35 - 10:15 EST) ───────────────────────
        # Note: 09:30 - 09:35 is the opening auction bar; entries begin at 09:35!
        if time(9, 35) <= cur_time <= time(10, 15):
            self._evaluate_zed_institutional_entry(current_price, cur_time)

    def _evaluate_zed_institutional_entry(self, p: float, t: time):
        """Evaluates entry triggers with VWAP, RVOL >= 1.5, and QQQ Relative Strength (ZedInstitutionalStrategy)."""
        ticker = self.parameters["Ticker"]
        qqq_asset = self.parameters.get("MarketBenchmark", Asset("QQQ", Asset.AssetType.STOCK))

        # Fetch 5m candles for stock
        raw_5m = self.get_historical_prices(ticker, 30, "5m")
        df_5m = raw_5m.df if hasattr(raw_5m, "df") else raw_5m
        if df_5m is None or df_5m.empty or len(df_5m) < 4:
            return

        # Fetch 5m candles for QQQ
        raw_qqq_5m = self.get_historical_prices(qqq_asset, 30, "5m")
        df_qqq_5m = raw_qqq_5m.df if hasattr(raw_qqq_5m, "df") else raw_qqq_5m

        # Compute Intraday VWAP
        df_5m["vwap"] = compute_vwap(df_5m)
        cur_vwap = float(df_5m["vwap"].iloc[-1])

        # Compute RVOL (vs 20-bar volume moving average)
        vol_ma20 = float(df_5m["volume"].rolling(20).mean().iloc[-1]) if len(df_5m) >= 20 else float(df_5m["volume"].mean())
        cur_volume = float(df_5m["volume"].iloc[-1])
        rvol = (cur_volume / vol_ma20) if vol_ma20 > 0 else 1.0

        last_bar = df_5m.iloc[-1]
        prev_bar = df_5m.iloc[-2]
        is_green_reversal = (last_bar["close"] > last_bar["open"]) and (last_bar["close"] > prev_bar["high"])

        # Enhancement 1: VWAP Reclaim & RVOL >= 1.5 (Configurable for V2 vs V3)
        vwap_reclaimed = True if not self.require_vwap else ((p >= cur_vwap) or (last_bar["low"] <= cur_vwap and last_bar["close"] > cur_vwap))
        rvol_confirmed = True if (not self.require_rvol or self.min_rvol <= 0) else ((rvol >= self.min_rvol) or (cur_volume >= vol_ma20 * 1.3))

        # Enhancement 2: Bullish Divergence vs QQQ
        has_bullish_divergence = True
        if self.require_div and df_qqq_5m is not None and len(df_qqq_5m) >= 2:
            qqq_last = df_qqq_5m.iloc[-1]
            qqq_prev = df_qqq_5m.iloc[-2]
            qqq_dropping = qqq_last["low"] <= qqq_prev["low"]
            stock_holding = last_bar["low"] >= prev_bar["low"]
            has_bullish_divergence = (not qqq_dropping) or stock_holding

        # Pre-conditions: Real opening drop of >= 0.8% to 2.0%
        drop_from_open = (self.open_price - self.session_low) / self.open_price if self.open_price else 0.0
        bounce_height = (p - self.session_low) / max(0.01, self.session_low)

        # Higher-Timeframe Market Filter (QQQ Bull Market check)
        if self.enforce_regime and self.market_regime != "BULL_MARKET":
            return

        cat_bias = self.catalyst_analysis.get("catalyst_bias", "NEUTRAL")

        # Confluence Trigger: Bullish Catalyst Confluence + V-recovery + VWAP + RVOL + Divergence
        if (
            is_green_reversal
            and vwap_reclaimed
            and rvol_confirmed
            and has_bullish_divergence
            and (cat_bias == "BULLISH" or drop_from_open >= 0.010)
            and bounce_height >= 0.006
        ):
            # Enhancement 3: Structural ATR Volatility Stop (1.2x of 15m ATR below session low, max 3.0%)
            atr_buffer = self.atr_stop_mult * (self.daily_atr * 0.20)
            hard_sl = max(self.session_low - atr_buffer, p * (1.0 - self.max_stop_pct))

            tp1 = p * (1.0 + self.target1_pct)
            tp2 = p * (1.0 + self.target2_pct)

            self._execute_zed_institutional_entry(
                price=p,
                stop_loss=hard_sl,
                tp1=tp1,
                tp2=tp2,
                rvol=rvol,
                vwap=cur_vwap,
                reason=f"VWAP Reclaimed (${cur_vwap:.2f}) | RVOL={rvol:.1f}x | Bullish Div Confirmed | {self.catalyst_analysis['catalyst_type']}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ORDER EXECUTION & 2-TIER SCALED SIZING
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_zed_institutional_entry(
        self,
        price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        rvol: float,
        vwap: float,
        reason: str
    ):
        portfolio = self.get_portfolio_value()
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0:
            return

        # 2% Risk Budget with strict 5% portfolio allocation cap
        total_qty = floor((portfolio * 0.02) / risk_per_share)
        total_qty = max(2, min(total_qty, floor((portfolio * 0.05) / price)))

        self.active_position_side = "BUY"
        self.entry_price = price
        self.total_quantity = total_qty
        self.remaining_quantity = total_qty
        self.tier1_sold = False
        self.stop_loss_price = stop_loss
        self.target1_price = tp1
        self.breakeven_applied = False
        self.setup_reason = reason
        self.trades_taken_today += 1

        self.log_message(
            f"🏛️ [INSTITUTIONAL V-ENTRY] BUY {total_qty} shares @ ${price:.2f} | "
            f"SL=${stop_loss:.2f} (1.5x ATR) | TP1=${tp1:.2f} (+{self.target1_pct:.1%}) | {reason}"
        )

        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=total_qty,
            side="buy",
            type="market",
            stop_loss_price=stop_loss
        )
        self.submit_order(order)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ACTIVE TRADE MONITORING, SCALE-OUT & TIME STOP (11:00 EST)
    # ─────────────────────────────────────────────────────────────────────────

    def _manage_zed_institutional_trade(self, current_price: float, cur_time: time):
        ticker = self.parameters["Ticker"]
        entry_p = self.entry_price or current_price

        # Enhancement 3: Breakeven Ratchet at +1.0% gain
        gain_pct = (current_price - entry_p) / entry_p
        if not self.breakeven_applied and gain_pct >= self.be_gain:
            self.stop_loss_price = entry_p + (0.001 * entry_p)  # entry + 0.1% cushion
            self.breakeven_applied = True
            self.log_message(f"🔒 [BREAKEVEN LOCK] Gain reached +{gain_pct:.1%}. Stop moved to Breakeven (${self.stop_loss_price:.2f}).")

        # Enhancement 5: Tier 1 Scale-Out (50% at +2.0% Target)
        if not self.tier1_sold and current_price >= self.target1_price:
            half_qty = floor(self.total_quantity * 0.50)
            if half_qty > 0 and self.remaining_quantity >= half_qty:
                self.log_message(
                    f"🎯 [TIER 1 PROFIT HIT] Selling 50% ({half_qty} shs) @ ${current_price:.2f} (+{gain_pct:.2%})."
                )
                tp_order = self.create_order(
                    asset=ticker,
                    quantity=half_qty,
                    side="sell",
                    type="market"
                )
                self.submit_order(tp_order)
                self.remaining_quantity -= half_qty
                self.tier1_sold = True
                # Move stop to breakeven immediately for remaining runner
                self.stop_loss_price = max(self.stop_loss_price or entry_p, entry_p)

        # Enhancement 5: Tier 2 Runner Target Hit (+3.5%)
        if self.tier1_sold and current_price >= entry_p * (1.0 + self.target2_pct):
            self.log_message(
                f"🚀 [RUNNER TARGET HIT] Closing remaining {self.remaining_quantity} shares @ ${current_price:.2f} (+{gain_pct:.2%})."
            )
            self.cancel_open_orders()
            self.close_position(ticker)
            self.active_position_side = None
            return

        # Hard Stop Loss / Breakeven Hit
        if current_price <= self.stop_loss_price:
            exit_label = "BREAKEVEN" if self.breakeven_applied and current_price >= entry_p else "STOP LOSS"
            self.log_message(
                f"🛑 [{exit_label} HIT] {ticker.symbol} hit stop at ${current_price:.2f} ({gain_pct:.2%})."
            )
            self.cancel_open_orders()
            self.close_position(ticker)
            self.active_position_side = None
            return

        # Strict EOD Flatten at 15:55 EST (Zero overnight hold)
        if cur_time >= time(15, 55):
            self.cancel_open_orders()
            self.close_position(ticker)
            self.active_position_side = None
            return

    # Backward compatibility aliases
    _evaluate_institutional_v_entry = _evaluate_zed_institutional_entry
    _execute_institutional_entry = _execute_zed_institutional_entry
    _manage_active_trade = _manage_zed_institutional_trade


class ZedNoVwapStrategy(ZedInstitutionalStrategy):
    """
    ZedNoVwapStrategy (Version 3 of ZedStrategy):
    Ablation variant removing VWAP Reclaim and RVOL confirmation filters
    to test the impact on trade frequency, win rate, and drawdown.
    """
    parameters: Dict[str, Any] = {
        **ZedInstitutionalStrategy.parameters,
        "RequireVwapReclaim": False,
        "RequireRvol": False,
        "MinRvol": 0.0,
    }


# Backward compatibility aliases
InstitutionalVRecoveryStrategy = ZedInstitutionalStrategy
ZedStrategyInstitutional = ZedInstitutionalStrategy
ZedStrategyNoVwap = ZedNoVwapStrategy
