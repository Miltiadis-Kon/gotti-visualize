"""
High-Beta Catalyst Opening Strategy (strategies/high_beta_catalyst_open.py)
==========================================================================
Codifies the 5-step rulebook, the 3 classic opening patterns, the forensic
COIN V-recovery higher-low scale-in mechanics, and the Catalyst Confluence Engine:

Core Rulebook Steps:
1. Stock Selection: High-Beta, high-volatility US equities (COIN, NVDA, TSLA, AMD, PLTR, MARA, MSTR)
   with 20-day ATR% >= 2.5% that move 2%+ daily.
2. Higher-Timeframe Market Filter: Nasdaq Composite / QQQ on Daily/1H.
   - Bull Market (Inflow): Money inflow dominates -> long-only bias.
   - Bear Market (Outflow): Money outflow dominates -> short-only bias.
3. 30-Minute Market Open (MO) Window (09:30 - 10:00 EST):
   - Pattern 1 (Golden V-Shape): Bull market, NO catalyst -> 1%-2% drop to Previous Day Close (PDC)
     or support, rebound on first 5m green reversal candle for +1.5% to +2.0% recovery.
   - Pattern 2 (Straight Rally): Bull market + Bullish Catalyst (> 4) -> Aggressive opening breakout.
   - Pattern 3 (Bear Market False Recovery): Bear market + Bearish Catalyst (< 2) or red pre-market ->
     drops, weak bounce fails below open/VWAP, shorts the lower high breakdown to new lows.
   - Pattern 4 (Forensic COIN V-Recovery + Bull Flag): Initial surge, pullback to establish base,
     second bounce forms a Higher Low (HL), primary entry + consolidation scale-in dip.

Extra Step Catalyst Engine:
- Evaluates Gemini sentiment rating (1 to 5) for:
  a) The given stock itself.
  b) Its primary industry competitors (e.g. COIN <-> MARA/MSTR, NVDA <-> AMD, TSLA <-> RIVN).
- Confluence Matrix combines Catalyst Direction with Nasdaq Market Regime to select the exact pattern.
- Take Profit: +1.5% to +2.0% quick scalp limit.
- Stop Loss: Mechanical -3.0% to -4.0%, or structural below the session V-bottom extreme.
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
from strategies.risk_management import calculate_atr
from strategies.trade_tracker import TradeTracker

logger = logging.getLogger("HighBetaCatalystOpen")

# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPETITOR & PEER NETWORK MAPPING & 7-DAY VOLATILITY SCANNER
# ─────────────────────────────────────────────────────────────────────────────

COMPETITOR_MAP: Dict[str, List[str]] = {
    "COIN": ["MARA", "MSTR", "RIOT", "CLSK"],
    "MARA": ["RIOT", "CLSK", "MSTR", "COIN"],
    "MSTR": ["COIN", "MARA", "RIOT", "CLSK"],
    "NVDA": ["AMD", "AVGO", "ARM", "SMCI", "INTC"],
    "AMD": ["NVDA", "INTC", "AVGO", "QCOM"],
    "TSLA": ["RIVN", "LCID", "NIO", "F", "GM"],
    "PLTR": ["AI", "PATH", "SNOW", "DDOG"],
    "AAPL": ["MSFT", "GOOGL", "META"],
    "MSFT": ["GOOGL", "AAPL", "AMZN", "META"],
    "GOOGL": ["MSFT", "META", "AMZN", "AAPL"],
    "AMZN": ["MSFT", "GOOGL", "BABA", "SHOP"],
}


def check_7day_consecutive_volatility(
    ticker: str,
    target_date: date,
    df_1d: Optional[pd.DataFrame],
    min_daily_range_pct: float = 0.020,
    lookback_days: int = 7,
) -> Tuple[bool, float, List[float]]:
    """
    Step 1 Rulebook Filter (ZedStrategy):
    Verifies that a stock moved >= 2.0% daily ((high - low) / low >= 0.020)
    for 7 consecutive trading days in a row prior to target_date.
    Returns:
        - is_qualified: bool (True only if all 7 consecutive days meet or exceed threshold)
        - avg_range: float (mean daily range over the 7 days)
        - daily_ranges: List[float] (daily range percentage for each day in lookback)
    """
    if df_1d is None or df_1d.empty:
        return False, 0.0, []

    # Filter for completed daily bars strictly prior to target_date (no lookahead)
    if isinstance(df_1d.index, pd.DatetimeIndex):
        prior_days = df_1d[df_1d.index.date < target_date]
    else:
        prior_days = df_1d

    if len(prior_days) < lookback_days:
        return False, 0.0, []

    recent_7 = prior_days.iloc[-lookback_days:]
    daily_ranges = []

    for idx in range(len(recent_7)):
        row = recent_7.iloc[idx]
        h = float(row["high"])
        l = float(row["low"])
        rng = (h - l) / l if l > 0 else 0.0
        daily_ranges.append(rng)
        if rng < min_daily_range_pct:
            return False, float(np.mean(daily_ranges)), daily_ranges

    avg_rng = float(np.mean(daily_ranges))
    return True, avg_rng, daily_ranges


def scan_premarket_7day_volatility_universe(
    target_date: date,
    candidate_tickers: List[str],
    ticker_data_1d: Dict[str, pd.DataFrame],
    min_daily_range_pct: float = 0.020,
    lookback_days: int = 7,
) -> List[str]:
    """
    Dynamic Pre-Market Universe Expansion:
    Scans candidate tickers and returns only those that exhibited >= 2.0% daily range
    for 7 consecutive trading days prior to the session.
    """
    qualified_universe = []
    for sym in candidate_tickers:
        df_1d = ticker_data_1d.get(sym)
        if df_1d is not None and not df_1d.empty:
            ok, avg_r, _ = check_7day_consecutive_volatility(
                sym, target_date, df_1d, min_daily_range_pct, lookback_days
            )
            if ok:
                qualified_universe.append(sym)
    return qualified_universe


class CatalystConfluenceEngine:
    """
    Evaluates news catalysts across the target stock and its primary competitors,
    then combines with the Nasdaq Composite (QQQ) regime to determine setup bias.
    """

    @classmethod
    def analyze_catalysts(
        cls,
        target_ticker: str,
        target_date_str: str,
        signals_cache: Optional[List[ReactorSignal]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes Gemini sentiment (> 4 or < 2) for the target stock and its competitor cluster.
        Returns:
            - target_sentiment: float (1.0 to 5.0) or None
            - competitor_sentiment: float (average of high-conviction peers)
            - catalyst_bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'
            - catalyst_type: 'DIRECT', 'COMPETITOR_SYMPATHY', 'MUTUAL', or 'NONE'
        """
        target_sym = target_ticker.upper().strip()
        peers = COMPETITOR_MAP.get(target_sym, [])

        if signals_cache is None:
            try:
                signals_cache = fetch_reactor_signals(target_date_str, include_prev_afterhours=True)
            except Exception:
                signals_cache = []

        direct_sigs = [s for s in signals_cache if s.ticker == target_sym]
        peer_sigs = [s for s in signals_cache if s.ticker in peers]

        target_rating: Optional[float] = None
        target_side: Optional[str] = None
        if direct_sigs:
            latest = direct_sigs[-1]
            target_rating = float(getattr(latest, "gemini_rating", 3.0))
            target_side = latest.signal_position

        # Evaluate competitor sentiment
        peer_ratings: List[float] = []
        peer_bullish_count = 0
        peer_bearish_count = 0
        for p in peer_sigs:
            r = float(getattr(p, "gemini_rating", 3.0))
            peer_ratings.append(r)
            if r > 4.0 or p.news_sentiment == "POSITIVE":
                peer_bullish_count += 1
            elif r < 2.0 or p.news_sentiment == "NEGATIVE":
                peer_bearish_count += 1

        avg_peer_rating = float(np.mean(peer_ratings)) if peer_ratings else 3.0

        # Classify direct catalyst (> 4 or < 2)
        has_direct_bull = (target_rating is not None and target_rating > 4.0) or target_side == "BUY"
        has_direct_bear = (target_rating is not None and target_rating < 2.0) or target_side == "SELL"

        # Classify competitor sympathy catalyst
        has_peer_bull = peer_bullish_count > peer_bearish_count and peer_bullish_count >= 1
        has_peer_bear = peer_bearish_count > peer_bullish_count and peer_bearish_count >= 1

        catalyst_bias = "NEUTRAL"
        catalyst_type = "NONE"

        if has_direct_bull and has_peer_bull:
            catalyst_bias = "BULLISH"
            catalyst_type = "MUTUAL_CLUSTER_CATALYST"
        elif has_direct_bull:
            catalyst_bias = "BULLISH"
            catalyst_type = "DIRECT_STOCK_CATALYST"
        elif has_peer_bull and not has_direct_bear:
            catalyst_bias = "BULLISH"
            catalyst_type = "COMPETITOR_SYMPATHY_CATALYST"
        elif has_direct_bear and has_peer_bear:
            catalyst_bias = "BEARISH"
            catalyst_type = "MUTUAL_CLUSTER_CATALYST"
        elif has_direct_bear:
            catalyst_bias = "BEARISH"
            catalyst_type = "DIRECT_STOCK_CATALYST"
        elif has_peer_bear and not has_direct_bull:
            catalyst_bias = "BEARISH"
            catalyst_type = "COMPETITOR_SYMPATHY_CATALYST"

        return {
            "target_ticker": target_sym,
            "target_rating": target_rating,
            "target_side": target_side,
            "peers": peers,
            "peer_ratings": peer_ratings,
            "avg_peer_rating": avg_peer_rating,
            "catalyst_bias": catalyst_bias,
            "catalyst_type": catalyst_type,
            "has_high_catalyst": catalyst_bias in ("BULLISH", "BEARISH"),
        }

    @classmethod
    def evaluate_market_regime(
        cls,
        qqq_bars_1d: Optional[pd.DataFrame],
        qqq_current_price: float,
        qqq_open_price: float
    ) -> str:
        """
        Determines Higher-Timeframe Market Filter on Nasdaq Composite (QQQ):
        - Uses 20-day SMA and daily momentum to determine institutional money inflow vs outflow:
        - 'BULL_MARKET' (Money Inflow): QQQ > 20D SMA and holding above open/PDC -> Buy Only.
        - 'BEAR_MARKET' (Money Outflow): QQQ < 20D SMA or down from open/PDC -> Short Only.
        """
        if qqq_bars_1d is not None and not qqq_bars_1d.empty and len(qqq_bars_1d) >= 10:
            c_series = qqq_bars_1d["close"]
            sma_len = min(20, len(c_series))
            sma20 = float(c_series.rolling(sma_len).mean().iloc[-1])
            prev_close = float(c_series.iloc[-1])

            net_change = (qqq_current_price - prev_close) / prev_close
            intraday_change = (qqq_current_price - qqq_open_price) / max(0.01, qqq_open_price)

            # Bearish Macro: below 20-day SMA or dropping > 0.5%
            if qqq_current_price < sma20 or (intraday_change < -0.003 and net_change < 0):
                return "BEAR_MARKET"
            elif qqq_current_price >= sma20 and intraday_change >= 0:
                return "BULL_MARKET"
            elif net_change >= 0.002:
                return "BULL_MARKET"
            else:
                return "BEAR_MARKET"

        if qqq_current_price >= qqq_open_price:
            return "BULL_MARKET"
        else:
            return "BEAR_MARKET"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ZED STRATEGY (HIGH-BETA CATALYST OPEN)
# ─────────────────────────────────────────────────────────────────────────────

class ZedStrategy(StrategyBaseplate):
    """
    ZedStrategy (High-Beta Catalyst Opening Strategy):
    Codifies Zed Monopoly's 5-step rulebook, 7-day consecutive volatility screen,
    and catalyst confluence matrix across 4 market-open patterns:
    - Pattern 1: Golden V-Shape PDC Rebound
    - Pattern 2: Straight Rally Breakout
    - Pattern 3: Bear False Recovery Breakdown
    - Pattern 4: Forensic COIN Higher-Low Bull Flag
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="COIN", asset_type=Asset.AssetType.STOCK),
        "MarketBenchmark": Asset(symbol="QQQ", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",
        "DynamicDateBinding": True,
        "EnforceNoLookahead": True,
        "TradingStyle": "day_trading",
        "TakeProfitPct": 0.020,               # Mechanical +2.0% Take-Profit limit
        "StopLossPct": 0.035,                 # Mechanical -3.5% Stop-Loss (PDF rule: 3% to 4%)
        "MinBetaATR": 0.025,                  # 20-day ATR% >= 2.5% high-beta screen
        "MinDailyRangePct": 0.020,            # 2.0%+ daily volatility
        "Require7DayVolatility": True,        # 7 consecutive days >= 2.0% daily range
        "EnableScaleIn": True,                # Forensic COIN 2-tranche scale-in
        "Tranche1Pct": 0.60,                  # Lot 1: 60% position size (e.g. 220 shares)
        "Tranche2Pct": 0.40,                  # Lot 2: 40% position size on flag dip (e.g. 130 shares)
        "BreakevenRatchetGain": 0.010,        # Move stop to breakeven at +1.0% gain
        "AllowOvernight": False,              # Strict 15:55 EST EOD flatten
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.tp_pct = float(self.parameters.get("TakeProfitPct", 0.020))
        self.sl_pct = float(self.parameters.get("StopLossPct", 0.035))
        self.min_beta_atr = float(self.parameters.get("MinBetaATR", 0.025))
        self.min_daily_range_pct = float(self.parameters.get("MinDailyRangePct", 0.020))
        self.require_7day_vol = bool(self.parameters.get("Require7DayVolatility", True))
        self.enable_scale_in = bool(self.parameters.get("EnableScaleIn", True))
        self.tranche1_ratio = float(self.parameters.get("Tranche1Pct", 0.60))
        self.tranche2_ratio = float(self.parameters.get("Tranche2Pct", 0.40))
        self.be_gain = float(self.parameters.get("BreakevenRatchetGain", 0.010))

        # Position and Trade State
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_position_side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.total_quantity: int = 0
        self.tranche1_shares: int = 0
        self.tranche2_shares: int = 0
        self.tranche2_filled: bool = False
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        self.breakeven_applied: bool = False
        self.chosen_pattern: Optional[str] = None

        # Session Context & Levels
        self.current_session_date: Optional[date] = None
        self.open_price: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.session_low: Optional[float] = None
        self.session_high: Optional[float] = None
        self.or_30m_high: Optional[float] = None
        self.or_30m_low: Optional[float] = None
        self.v_bottom_price: Optional[float] = None
        self.higher_low_price: Optional[float] = None

        # Market & Catalyst Context
        self.market_regime: str = "BULL_MARKET"
        self.catalyst_analysis: Dict[str, Any] = {}
        self.is_high_beta: bool = True
        self.daily_atr: float = 2.0
        self.trades_taken_today: int = 0

        self._reset_daily_session(pd.to_datetime(self.signal_date_str).date())

    def _reset_daily_session(self, new_date: date):
        """Resets session variables and runs pre-market catalyst & regime analysis."""
        self.current_session_date = new_date
        self.active_position_side = None
        self.entry_price = None
        self.total_quantity = 0
        self.tranche1_shares = 0
        self.tranche2_shares = 0
        self.tranche2_filled = False
        self.stop_loss_price = None
        self.take_profit_price = None
        self.breakeven_applied = False
        self.chosen_pattern = None

        self.open_price = None
        self.prev_close = None
        self.session_low = None
        self.session_high = None
        self.or_30m_high = None
        self.or_30m_low = None
        self.v_bottom_price = None
        self.higher_low_price = None
        self.trades_taken_today = 0

        ticker_sym = self.parameters["Ticker"].symbol
        date_str = new_date.strftime("%Y-%m-%d")

        # Step 1: Pre-market Catalyst Analysis (Target Stock + Competitors)
        self.catalyst_analysis = CatalystConfluenceEngine.analyze_catalysts(ticker_sym, date_str)
        self.log_message(
            f"[{ticker_sym}] Session {date_str} Catalyst Context: Bias={self.catalyst_analysis['catalyst_bias']} | "
            f"Type={self.catalyst_analysis['catalyst_type']} | "
            f"Direct Rating={self.catalyst_analysis['target_rating']} | "
            f"Peer Avg={self.catalyst_analysis['avg_peer_rating']:.1f}"
        )

        # Step 2: Fetch Daily Bars & ATR% High-Beta Screening
        self._calculate_daily_levels_and_beta()

    def _calculate_daily_levels_and_beta(self):
        """Validates 20-day ATR% >= 2.5%, 7-day consecutive volatility >= 2.0%, and extracts Previous Day Close (PDC)."""
        ticker = self.parameters["Ticker"]
        try:
            bars_1d = self.get_historical_prices(ticker, 25, "day")
            if bars_1d is not None:
                df = bars_1d.df if hasattr(bars_1d, "df") else bars_1d
                if not df.empty and len(df) >= 2:
                    self.prev_close = float(df["close"].iloc[-2])
                    self.daily_atr = float(calculate_atr(df, length=min(20, len(df) - 1)))
                    atr_pct = self.daily_atr / self.prev_close
                    self.is_high_beta = (atr_pct >= self.min_beta_atr)

                    # 7-Day Consecutive Volatility Check
                    is_7d_vol, avg_7d_rng, _ = check_7day_consecutive_volatility(
                        ticker.symbol,
                        self.current_session_date,
                        df,
                        min_daily_range_pct=self.min_daily_range_pct,
                        lookback_days=7
                    )
                    if self.require_7day_vol and not is_7d_vol:
                        self.is_high_beta = False

                    self.log_message(
                        f"[{ticker.symbol}] PDC=${self.prev_close:.2f} | 20D ATR=${self.daily_atr:.2f} "
                        f"({atr_pct:.1%}) | 7D Volatility ({avg_7d_rng:.1%}): {is_7d_vol} -> Qualified: {self.is_high_beta}"
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

        # Track Session High & Low
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
            self._manage_zed_strategy_trade(current_price, cur_time)
            return

        # Max 1 trade per session
        if self.trades_taken_today >= 1:
            return

        # Require high beta & 7-day volatility qualification
        if not self.is_high_beta:
            return

        # ── 2. MARKET REGIME FILTER (NASDAQ COMPOSITE / QQQ) ─────────────────
        qqq_asset = self.parameters.get("MarketBenchmark", Asset("QQQ", Asset.AssetType.STOCK))
        qqq_price = self.get_last_price(qqq_asset) or 480.0
        qqq_bars = None
        try:
            raw_qqq = self.get_historical_prices(qqq_asset, 5, "day")
            qqq_bars = raw_qqq.df if hasattr(raw_qqq, "df") else raw_qqq
        except Exception:
            pass

        qqq_open = float(qqq_bars["open"].iloc[-1]) if qqq_bars is not None and not qqq_bars.empty else qqq_price
        self.market_regime = CatalystConfluenceEngine.evaluate_market_regime(qqq_bars, qqq_price, qqq_open)

        # ── 3. PATTERN DETECTION & CONFLUENCE TRIGGER (09:30 - 10:15 EST) ─────
        if time(9, 30) <= cur_time < time(10, 15):
            self._evaluate_zed_strategy_entry(current_price, cur_time)

    def _evaluate_zed_strategy_entry(self, p: float, t: time):
        """
        Evaluates the 4 Opening Patterns based on Catalyst Confluence & Nasdaq Regime (ZedStrategy).
        """
        if self.open_price is None or self.prev_close is None:
            return

        cat_bias = self.catalyst_analysis.get("catalyst_bias", "NEUTRAL")
        pdc = self.prev_close
        drop_from_open = (self.open_price - p) / self.open_price
        gain_from_open = (p - self.open_price) / self.open_price

        # Fetch recent 5m bars for candlestick confirmation
        df_5m = None
        try:
            raw_5m = self.get_historical_prices(self.parameters["Ticker"], 12, "5m")
            df_5m = raw_5m.df if hasattr(raw_5m, "df") else raw_5m
        except Exception:
            pass

        if df_5m is None or df_5m.empty or len(df_5m) < 2:
            return

        last_bar = df_5m.iloc[-1]
        prev_bar = df_5m.iloc[-2]
        is_green_reversal = (last_bar["close"] > last_bar["open"]) and (last_bar["close"] > prev_bar["high"])
        is_red_breakdown = (last_bar["close"] < last_bar["open"]) and (last_bar["close"] < prev_bar["low"])

        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 2: STRAIGHT RALLY (BULL MARKET + STRONG BULL CATALYST > 4)
        # ─────────────────────────────────────────────────────────────────────
        if self.market_regime == "BULL_MARKET" and cat_bias == "BULLISH":
            # Stock surges right off the bell without waiting for a deep V-dip
            if gain_from_open >= 0.010 and is_green_reversal and t <= time(9, 45):
                self._execute_zed_strategy_entry(
                    side="BUY",
                    price=p,
                    pattern="PATTERN_2_STRAIGHT_RALLY_BREAKOUT",
                    sl_price=min(p * (1.0 - self.sl_pct), float(df_5m["low"].min())),
                    tp_price=p * (1.0 + self.tp_pct),
                    reason=f"Bull Market Inflow + {self.catalyst_analysis['catalyst_type']} (>4)"
                )
                return

        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 1: GOLDEN V-SHAPE (BULL MARKET + NO CATALYST / NEUTRAL)
        # ─────────────────────────────────────────────────────────────────────
        if self.market_regime == "BULL_MARKET" and cat_bias == "NEUTRAL":
            # 1% to 2% indecision drop to PDC or Key Support within first 30 minutes
            tested_pdc = (drop_from_open >= 0.008 and abs(p - pdc) / pdc <= 0.007) or (p <= pdc and drop_from_open >= 0.010)
            if tested_pdc and is_green_reversal:
                self.v_bottom_price = float(df_5m["low"].min())
                self._execute_zed_strategy_entry(
                    side="BUY",
                    price=p,
                    pattern="PATTERN_1_GOLDEN_V_SHAPE_PDC_REBOUND",
                    sl_price=min(p * (1.0 - self.sl_pct), self.v_bottom_price - (0.5 * self.daily_atr * 0.15)),
                    tp_price=p * (1.0 + self.tp_pct),
                    reason="Bull Market Inflow + Non-Catalyst Reversion to PDC Support"
                )
                return

        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 4: FORENSIC COIN V-RECOVERY + HIGHER-LOW BULL FLAG
        # (Bull Market + Bullish Catalyst Confluence > 4 on stock or peers)
        # ─────────────────────────────────────────────────────────────────────
        if self.market_regime == "BULL_MARKET" and cat_bias == "BULLISH":
            lowest_low = float(df_5m["low"].min())
            highest_high = float(df_5m["high"].max())
            # Must have experienced a real opening dip of >= 0.8% and bounced
            dip_depth = (self.open_price - lowest_low) / self.open_price
            bounce_height = (highest_high - lowest_low) / lowest_low

            if dip_depth >= 0.008 and bounce_height >= 0.008:
                # Forming a genuine Higher Low (at least 0.6% above lowest low, holding above support)
                is_higher_low = (last_bar["low"] >= lowest_low * 1.006) and (p < highest_high)
                if is_higher_low and is_green_reversal:
                    self.higher_low_price = float(last_bar["low"])
                    self._execute_zed_strategy_entry(
                        side="BUY",
                        price=p,
                        pattern="PATTERN_4_FORENSIC_COIN_V_RECOVERY_BULL_FLAG",
                        sl_price=lowest_low - (0.2 * self.daily_atr * 0.15),
                        tp_price=p * (1.0 + self.tp_pct),
                        reason=f"Forensic COIN V-Recovery Rebound off Higher Low (+{bounce_height:.1%}) | {self.catalyst_analysis['catalyst_type']} (>4)"
                    )
                    return

        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 3: BEAR MARKET FALSE RECOVERY BREAKDOWN
        # (Bear Market Regime + Bearish Catalyst Confluence < 2 on stock or peers)
        # ─────────────────────────────────────────────────────────────────────
        if self.market_regime == "BEAR_MARKET" and (cat_bias == "BEARISH" or self.open_price < pdc):
            # Stock drops, attempts a recovery that fails below open, and breaks down below morning low
            if self.session_low is not None and self.session_high is not None:
                attempted_recovery = (self.session_high < self.open_price * 1.003) and (self.session_high > self.session_low * 1.004)
                breaks_morning_low = (p < self.session_low) and is_red_breakdown
                if attempted_recovery and breaks_morning_low:
                    self._execute_zed_strategy_entry(
                        side="SELL",
                        price=p,
                        pattern="PATTERN_3_BEAR_FALSE_RECOVERY_BREAKDOWN",
                        sl_price=min(p * (1.0 + self.sl_pct), self.session_high + (0.2 * self.daily_atr * 0.15)),
                        tp_price=p * (1.0 - self.tp_pct),
                        reason=f"Bear Market Outflow + Failed Morning Recovery Breakdown | {self.catalyst_analysis['catalyst_type']} (<2)"
                    )
                    return

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ORDER SIZING & SCALE-IN EXECUTION
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_zed_strategy_entry(
        self,
        side: str,
        price: float,
        pattern: str,
        sl_price: float,
        tp_price: float,
        reason: str
    ):
        """Calculates lot sizing, submits primary tranche, and arms scale-in tranche (ZedStrategy)."""
        portfolio = self.get_portfolio_value()
        risk_per_share = abs(price - sl_price)
        if risk_per_share <= 0:
            return

        # Sizing based on 2% risk budget with strict 5% portfolio allocation cap
        total_qty = floor((portfolio * 0.02) / risk_per_share)
        total_qty = max(2, min(total_qty, floor((portfolio * 0.05) / price)))

        if self.enable_scale_in and pattern == "PATTERN_4_FORENSIC_COIN_V_RECOVERY_BULL_FLAG":
            # 2 Tranches: e.g. 60% lot 1 (220 shs) + 40% lot 2 (130 shs)
            self.tranche1_shares = floor(total_qty * self.tranche1_ratio)
            self.tranche2_shares = total_qty - self.tranche1_shares
            exec_qty = self.tranche1_shares
        else:
            exec_qty = total_qty
            self.tranche1_shares = total_qty
            self.tranche2_shares = 0

        self.active_position_side = side
        self.entry_price = price
        self.total_quantity = exec_qty
        self.stop_loss_price = sl_price
        self.take_profit_price = tp_price
        self.chosen_pattern = pattern
        self.breakeven_applied = False
        self.trades_taken_today += 1

        self.log_message(
            f"🚀 [ORDER ENTRY] {side} {exec_qty} shares @ ${price:.2f} | Pattern: {pattern} | "
            f"TP=${tp_price:.2f} (+{self.tp_pct:.1%}) | SL=${sl_price:.2f} | {reason}"
        )

        # Broker Execution via Lumibot
        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=exec_qty,
            side=side.lower(),
            type="market",
            take_profit_price=tp_price,
            stop_loss_price=sl_price
        )
        self.submit_order(order)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ACTIVE TRADE MONITORING, SCALE-IN & EOD FLATTEN
    # ─────────────────────────────────────────────────────────────────────────

    def _manage_zed_strategy_trade(self, current_price: float, cur_time: time):
        """Monitors scale-in dips, breakeven stop ratchets, and 15:55 EST EOD flattening (ZedStrategy)."""
        ticker = self.parameters["Ticker"]
        entry_p = self.entry_price or current_price

        # Check Tranche 2 Scale-In (Forensic COIN Bull Flag dip)
        if (
            self.enable_scale_in
            and not self.tranche2_filled
            and self.tranche2_shares > 0
            and self.active_position_side == "BUY"
            and cur_time < time(10, 15)
        ):
            # Scale-in if price dips shallowly into flag consolidation (holding above SL)
            is_consolidation_dip = (current_price <= entry_p * 1.002) and (current_price > self.stop_loss_price)
            if is_consolidation_dip:
                self.log_message(
                    f"➕ [SCALE-IN TRANCHE 2] Buying {self.tranche2_shares} shares @ ${current_price:.2f} "
                    f"(Forensic COIN Bull Flag Dip)"
                )
                scale_order = self.create_order(
                    asset=ticker,
                    quantity=self.tranche2_shares,
                    side="buy",
                    type="market",
                    take_profit_price=self.take_profit_price,
                    stop_loss_price=self.stop_loss_price
                )
                self.submit_order(scale_order)
                # Re-average entry price
                self.entry_price = ((entry_p * self.tranche1_shares) + (current_price * self.tranche2_shares)) / (self.tranche1_shares + self.tranche2_shares)
                self.total_quantity += self.tranche2_shares
                self.tranche2_filled = True

        # Breakeven Ratchet at +1.0%
        gain_pct = (current_price - self.entry_price) / self.entry_price if self.active_position_side == "BUY" else (self.entry_price - current_price) / self.entry_price
        if not self.breakeven_applied and gain_pct >= self.be_gain:
            self.stop_loss_price = self.entry_price
            self.breakeven_applied = True
            self.log_message(f"🔒 [BREAKEVEN RATCHET] Gain reached +{gain_pct:.1%}. Stop Loss moved to Entry (${self.entry_price:.2f}).")

        # Strict EOD Flatten at 15:55 EST
        if cur_time >= time(15, 55):
            self.log_message(f"⏰ [EOD FLATTEN] Closing {ticker.symbol} position at {cur_time.strftime('%H:%M:%S')} EST.")
            self.cancel_open_orders()
            self.close_position(ticker)
            self.active_position_side = None
            return

        # TP & SL Hits (for manual checking and backtesting sync)
        if self.active_position_side == "BUY":
            if current_price >= self.take_profit_price:
                self.log_message(f"🎯 [TAKE PROFIT HIT] {ticker.symbol} reached ${current_price:.2f} (+{gain_pct:.2%}).")
                self.close_position(ticker)
                self.active_position_side = None
            elif current_price <= self.stop_loss_price:
                self.log_message(f"🛑 [STOP LOSS HIT] {ticker.symbol} hit stop at ${current_price:.2f} ({gain_pct:.2%}).")
                self.close_position(ticker)
                self.active_position_side = None
        else:  # SELL
            if current_price <= self.take_profit_price:
                self.log_message(f"🎯 [TAKE PROFIT HIT] {ticker.symbol} short covered at ${current_price:.2f} (+{gain_pct:.2%}).")
                self.close_position(ticker)
                self.active_position_side = None
            elif current_price >= self.stop_loss_price:
                self.log_message(f"🛑 [STOP LOSS HIT] {ticker.symbol} short stopped at ${current_price:.2f} ({gain_pct:.2%}).")
                self.close_position(ticker)
                self.active_position_side = None

    # Backward compatibility aliases
    _evaluate_opening_patterns = _evaluate_zed_strategy_entry
    _execute_entry = _execute_zed_strategy_entry
    _manage_active_trade = _manage_zed_strategy_trade


# Backward compatibility class alias
HighBetaCatalystOpenStrategy = ZedStrategy
