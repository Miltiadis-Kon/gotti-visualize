"""
Market Reactor Strategy (market_reactor.py)
============================================
Connects Stock Alchemist intelligence signals (fundamental & news sentiment)
with automated market execution using Lumibot, institutional Key Levels clustering,
and the unified StrategyBaseplate ATR volatility risk framework.

Core Mechanics:
1. Signal Ingestion:
   - Ingests signals from MySQL `gotti.signals` joined with `gotti.fundamental_analysis`.
   - Dual Hard Floor Confidence Filter:
     * Fundamental Confidence >= 70% (from financial statements & valuation completeness)
     * News Conviction >= 70% (from Gemini sentiment rating & catalyst impact)
2. Three Selectors:
   - Selector 1 (Macro Alignment): News & Fundamentals in same direction -> 1-Hour (1h) S/R + ATR
   - Selector 2 (Catalyst Momentum): News opposing Fundamentals (e.g. Bullish on Value Trap) -> 15-Minute (15m) S/R + ATR
   - Selector 3 (Directional Flow): News with Neutral/HOLD Fundamentals -> 30-Minute (30m) S/R + ATR
3. Execution:
   - Market open (9:30 AM EST) bracket order execution (Entry, Stop Loss, Take Profit).
   - Position sizing: 2% portfolio risk per trade.
   - Overnight holding allowed until either TP or SL is triggered.
"""

import os
import sys
import json
import logging
from math import floor
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import mysql.connector
from dotenv import load_dotenv

from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy

# Ensure root paths are in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from strategies.strat_baseplate import StrategyBaseplate
from strategies.risk_management import optimize_sl_tp, calculate_atr, ATRRiskLevels
from strategies.key_levels import analyze
from strategies.trade_tracker import TradeTracker

load_dotenv()
logger = logging.getLogger("MarketReactor")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SIGNAL CLASSIFIER & CONFIDENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReactorSignal:
    ticker: str
    signal_position: str         # "BUY" or "SELL"
    fundamental_sentiment: str   # e.g. "BUY", "SELL [VALUE TRAP]", "HOLD"
    news_sentiment: str          # "POSITIVE" or "NEGATIVE"
    headline: str
    fundamental_confidence: float # 0.0 to 1.0 (from fundamental_analysis.data)
    news_conviction: float        # 0.0 to 1.0 (from gemini rating & impact)
    is_eligible: bool            # True if BOTH >= 0.70
    selector_id: int             # 1, 2, or 3
    selector_name: str           # "Macro Alignment", "Catalyst Momentum", "Directional Flow"
    sr_timeframe: str            # "1H", "15m", or "30m"
    resolution_code: str         # "1H" or "15m" for key_levels analyzer
    gemini_rating: int           # 1 to 5
    gemini_impact: str           # "High", "Medium", "Low"
    raw_signal_id: str


class MarketReactorClassifier:
    """
    Evaluates raw signal rows from MySQL, applies the Dual Hard Floor confidence
    filter (both >= 70%), and maps qualifying setups into the 3 Selectors.
    """

    MIN_CONFIDENCE_THRESHOLD = 0.70

    @classmethod
    def calculate_news_conviction(cls, gemini_rating: int, impact_level: str) -> float:
        """
        Calculates News Conviction (0.0 to 1.0) from Gemini rating (1-5) and impact type.
        
        Rating Distance from Neutral:
        - Rating 5 or 1 (Strong Buy / Strong Sell) -> 1.00 base conviction
        - Rating 4 or 2 (Bullish / Bearish)        -> 0.80 base conviction
        - Rating 3 (Neutral / Noise)               -> 0.50 base conviction (always rejected)
        
        Impact Multiplier:
        - High   -> 1.00
        - Medium -> 0.95
        - Low    -> 0.75
        """
        if gemini_rating in (1, 5):
            base_conviction = 1.00
        elif gemini_rating in (2, 4):
            base_conviction = 0.80
        else:
            base_conviction = 0.50

        clean_impact = str(impact_level or "Medium").strip().capitalize()
        if "High" in clean_impact:
            impact_multiplier = 1.00
        elif "Low" in clean_impact:
            impact_multiplier = 0.75
        else:
            impact_multiplier = 0.95

        return round(base_conviction * impact_multiplier, 4)

    @classmethod
    def classify_signal(cls, raw_row: Dict[str, Any], fa_data: Optional[Dict[str, Any]] = None) -> ReactorSignal:
        ticker = str(raw_row.get("ticker", "")).strip().upper()
        signal_pos = str(raw_row.get("signal_position", "BUY")).strip().upper()
        
        # Parse sentiment JSON
        sentiment = raw_row.get("sentiment") or {}
        if isinstance(sentiment, str):
            try:
                sentiment = json.loads(sentiment)
            except Exception:
                sentiment = {}

        news_sentiment = str(sentiment.get("news_sentiment", "NEUTRAL")).strip().upper()
        fund_sentiment = str(sentiment.get("fundamental_sentiment", "HOLD")).strip().upper()
        headline = str(sentiment.get("headline", "")).strip()

        # Extract Fundamental Confidence (Primary: fa_data.confidence, fallback: sentiment.confidence or 0.85)
        fund_conf = 0.85
        if fa_data and isinstance(fa_data, dict):
            fc = fa_data.get("confidence")
            if fc is not None:
                try:
                    fund_conf = float(fc)
                except (ValueError, TypeError):
                    pass
        elif sentiment.get("confidence") is not None:
            try:
                fund_conf = float(sentiment["confidence"])
            except (ValueError, TypeError):
                pass

        # Extract News Conviction
        try:
            gemini_rating = int(sentiment.get("gemini_sentiment", 3))
        except (ValueError, TypeError):
            gemini_rating = 3
        gemini_impact = str(sentiment.get("gemini_impact_type", "Medium"))
        news_conv = cls.calculate_news_conviction(gemini_rating, gemini_impact)

        # Dual Hard Floor check: BOTH must be >= 70%
        is_eligible = (fund_conf >= cls.MIN_CONFIDENCE_THRESHOLD) and (news_conv >= cls.MIN_CONFIDENCE_THRESHOLD)

        # Routing between the 3 Selectors
        is_news_bullish = news_sentiment in ("POSITIVE", "BULLISH")
        is_news_bearish = news_sentiment in ("NEGATIVE", "BEARISH")
        
        is_fund_bullish = ("BUY" in fund_sentiment or "POSITIVE" in fund_sentiment or "UNDERVALUED" in fund_sentiment) and ("SELL" not in fund_sentiment)
        is_fund_bearish = ("SELL" in fund_sentiment or "NEGATIVE" in fund_sentiment or "VALUE TRAP" in fund_sentiment or "VETO" in fund_sentiment or "DISTRESS" in fund_sentiment)
        is_fund_neutral = ("HOLD" in fund_sentiment or "NEUTRAL" in fund_sentiment) and not is_fund_bearish and not is_fund_bullish

        # Determine Selector
        if (is_news_bullish and is_fund_bullish) or (is_news_bearish and is_fund_bearish):
            # Selector 1: Macro Alignment (1-Hour S/R)
            selector_id = 1
            selector_name = "Macro Alignment"
            sr_timeframe = "1H"
            resolution_code = "1H"
        elif (is_news_bullish and is_fund_bearish) or (is_news_bearish and is_fund_bullish):
            # Selector 2: Catalyst Momentum Opposing Fundamentals (15-Minute S/R)
            selector_id = 2
            selector_name = "Catalyst Momentum"
            sr_timeframe = "15m"
            resolution_code = "15m"
        else:
            # Selector 3: Directional Flow on Neutral / Hold Fundamentals (30-Minute S/R)
            selector_id = 3
            selector_name = "Directional Flow"
            sr_timeframe = "30m"
            resolution_code = "15m"  # Computed using 15m/1H key levels

        return ReactorSignal(
            ticker=ticker,
            signal_position=signal_pos,
            fundamental_sentiment=fund_sentiment,
            news_sentiment=news_sentiment,
            headline=headline,
            fundamental_confidence=round(fund_conf, 4),
            news_conviction=round(news_conv, 4),
            is_eligible=is_eligible,
            selector_id=selector_id,
            selector_name=selector_name,
            sr_timeframe=sr_timeframe,
            resolution_code=resolution_code,
            gemini_rating=gemini_rating,
            gemini_impact=gemini_impact,
            raw_signal_id=str(raw_row.get("signal_id", ""))
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. MYSQL DATA ACCESS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_mysql_connection():
    """Returns an autocommit MySQL connection to the gotti database."""
    host = os.getenv("DB_HOST", os.getenv("MYSQL_HOST", "127.0.0.1"))
    port = int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", 3306)))
    user = os.getenv("DB_USER", os.getenv("DB_USERNAME", "root"))
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", os.getenv("DB_DATABASE", "gotti"))

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True
    )


def fetch_reactor_signals(target_date: str) -> List[ReactorSignal]:
    """
    Fetches signals for a given target date from MySQL `signals` table,
    joins linked `fundamental_analysis` record for confidence scores,
    and returns classified ReactorSignal objects.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT 
        s.signal_id,
        s.ticker,
        s.signal_date,
        s.signal_position,
        s.fundamental_analysis_key,
        s.sentiment,
        s.created_at,
        fa.data AS fa_data
    FROM signals s
    LEFT JOIN fundamental_analysis fa 
        ON s.fundamental_analysis_key = fa.entry_key
    WHERE s.signal_date = %s
    ORDER BY s.updated_at ASC;
    """
    
    cursor.execute(query, (target_date,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    signals: List[ReactorSignal] = []
    for row in rows:
        fa_data = None
        if row.get("fa_data"):
            raw_fa = row["fa_data"]
            if isinstance(raw_fa, str):
                try:
                    fa_data = json.loads(raw_fa)
                except Exception:
                    fa_data = None
            elif isinstance(raw_fa, dict):
                fa_data = raw_fa

        reactor_sig = MarketReactorClassifier.classify_signal(row, fa_data)
        signals.append(reactor_sig)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET REACTOR STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

class MarketReactorStrategy(StrategyBaseplate):
    """
    Lumibot Market Reactor Strategy.
    Executes trades on market open according to Stock Alchemist signals,
    applying Dual Hard Floor confidence filtering and 3-selector S/R + ATR bracket orders.
    """

    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",         # Date of news signals to trade
        "MinConfidence": 0.70,              # Dual floor threshold
        "RiskPct": 0.02,                   # 2% portfolio risk per trade
        "TradingStyle": "swing_trading",   # "swing_trading" or "day_trading"
        "MinRR": 1.5,                      # Minimum Risk-to-Reward ratio
        "AllowOvernight": True,            # Hold positions overnight until TP/SL hits
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"  # Evaluate every 5 minutes
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self.risk_pct = float(self.parameters.get("RiskPct", 0.02))
        self.min_rr = float(self.parameters.get("MinRR", 1.5))
        self.allow_overnight = bool(self.parameters.get("AllowOvernight", True))

        # Trade tracking
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_trade_id = None
        self.has_entered_for_signal = False
        
        # Load and resolve matching signal for this ticker
        self.matched_signal: Optional[ReactorSignal] = None
        self._load_and_bind_signal()

    def _load_and_bind_signal(self):
        ticker_sym = self.parameters["Ticker"].symbol
        all_signals = fetch_reactor_signals(self.signal_date_str)
        ticker_signals = [s for s in all_signals if s.ticker == ticker_sym]

        if not ticker_signals:
            self.log_message(f"[{ticker_sym}] No signals found for date {self.signal_date_str}.")
            return

        # Pick the most recent signal for this ticker
        sig = ticker_signals[-1]
        self.matched_signal = sig

        status = "ELIGIBLE" if sig.is_eligible else "REJECTED (<70% dual floor)"
        self.log_message(
            f"[{ticker_sym}] Loaded signal: {sig.signal_position} | "
            f"Fund: {sig.fundamental_sentiment} (Conf: {sig.fundamental_confidence:.1%}) | "
            f"News: {sig.news_sentiment} (Conv: {sig.news_conviction:.1%}) | "
            f"Selector {sig.selector_id} ({sig.selector_name}, TF: {sig.sr_timeframe}) -> {status}"
        )

    def _get_sr_levels(self, current_price: float, resolution: str) -> Tuple[float, float]:
        """
        Calculates Support and Resistance levels for the target resolution (1H or 15m)
        using the institutional Key Levels cluster engine.
        """
        ticker_sym = self.parameters["Ticker"].symbol
        try:
            res = analyze(ticker_sym, resolutions=[resolution], use_alpaca=False)
            kl_df = res.key_levels
            
            if kl_df is not None and not kl_df.empty:
                supports = kl_df[(kl_df['type'] == 'support') & (kl_df['level_price'] < current_price)]['level_price'].values
                resistances = kl_df[(kl_df['type'] == 'resistance') & (kl_df['level_price'] > current_price)]['level_price'].values

                sup = float(np.max(supports)) if len(supports) > 0 else current_price * 0.98
                res_lvl = float(np.min(resistances)) if len(resistances) > 0 else current_price * 1.02
                return sup, res_lvl
        except Exception as e:
            self.log_message(f"[{ticker_sym}] Error fetching key levels ({resolution}): {e}. Using fallback 2% buffer.")

        # Fallback if no levels exist
        return current_price * 0.98, current_price * 1.02

    def _get_atr(self) -> float:
        """Calculates 14-period ATR from recent market data."""
        try:
            bars = self.get_historical_prices(self.parameters["Ticker"], 30, "15m")
            if bars is not None:
                df = bars.df if hasattr(bars, "df") else bars
                if not df.empty and len(df) >= 14:
                    return calculate_atr(df, length=14)
        except Exception:
            pass
        return self.get_last_price(self.parameters["Ticker"]) * 0.015

    def get_position_sizing(self, entry_price: float, stop_loss: float) -> int:
        """Sizes trade based on portfolio risk percentage."""
        portfolio_val = self.get_portfolio_value()
        risk_budget = portfolio_val * self.risk_pct
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        qty = floor(risk_budget / risk_per_share)
        max_affordable = floor((portfolio_val * 0.95) / max(0.01, entry_price))
        return max(0, min(qty, max_affordable))

    def on_trading_iteration(self):
        """
        Primary execution loop:
        1. On market open, if signal is eligible and not yet entered, submit bracket order.
        2. Monitor active trade for TP/SL trigger.
        """
        ticker = self.parameters["Ticker"]
        ticker_sym = ticker.symbol
        current_price = self.get_last_price(ticker)
        current_dt = self.get_datetime()

        if current_price is None or current_price <= 0:
            return

        # ── 1. CHECK IF POSITION IS OPEN & MONITOR EXITS ─────────────────
        position = self.get_position(ticker)
        if position is not None and position.quantity != 0:
            if self.active_trade_id and self.active_trade_id in self.trade_tracker.trades:
                trade = self.trade_tracker.trades[self.active_trade_id]
                
                # Check Take Profit
                if (trade.trade_type == "BUY" and current_price >= trade.take_profit) or \
                   (trade.trade_type == "SELL" and current_price <= trade.take_profit):
                    self.log_message(f"[{ticker_sym}] TAKE PROFIT HIT @ ${current_price:.2f} (Target: ${trade.take_profit:.2f})")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "TP")
                    self._close_all_positions()
                    self.active_trade_id = None
                    return

                # Check Stop Loss
                if (trade.trade_type == "BUY" and current_price <= trade.stop_loss) or \
                   (trade.trade_type == "SELL" and current_price >= trade.stop_loss):
                    self.log_message(f"[{ticker_sym}] STOP LOSS HIT @ ${current_price:.2f} (Stop: ${trade.stop_loss:.2f})")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "SL")
                    self._close_all_positions()
                    self.active_trade_id = None
                    return
            return

        # ── 2. NEW ENTRY EVALUATION ON MARKET OPEN ────────────────────────
        if self.has_entered_for_signal or not self.matched_signal or not self.matched_signal.is_eligible:
            return

        # Ensure we execute at/after 9:30 AM EST
        current_time = current_dt.time()
        if current_time < time(9, 30):
            return

        sig = self.matched_signal
        trade_type = sig.signal_position  # "BUY" or "SELL"
        side = "buy" if trade_type == "BUY" else "sell"

        # Calculate S/R based on Selector Timeframe
        sup, res = self._get_sr_levels(current_price, sig.resolution_code)
        atr_val = self._get_atr()

        # Define raw TP/SL from S/R
        if trade_type == "BUY":
            raw_sl = sup
            raw_tp = res
        else:
            raw_sl = res
            raw_tp = sup

        # Pass raw levels through StrategyBaseplate ATR optimization
        risk_levels = self.optimize_sl_tp(
            entry_price=current_price,
            stop_loss=raw_sl,
            take_profit=raw_tp,
            side=side,
            trading_style=self.parameters.get("TradingStyle", "swing_trading"),
            atr=atr_val,
            risk_reward_ratio=self.min_rr
        )

        final_sl = risk_levels.stop_loss
        final_tp = risk_levels.take_profit

        # Position Sizing
        quantity = self.get_position_sizing(current_price, final_sl)
        if quantity <= 0:
            self.log_message(f"[{ticker_sym}] Position sizing resulted in 0 shares. Skipping entry.")
            return

        # Execute Entry Order
        self.log_message(
            f"[{ticker_sym}] EXECUTING {trade_type} | Qty: {quantity} @ ${current_price:.2f} | "
            f"SL: ${final_sl:.2f} | TP: ${final_tp:.2f} | R:R: {risk_levels.risk_reward_ratio:.2f} | "
            f"Selector {sig.selector_id} ({sig.selector_name}, TF: {sig.sr_timeframe})"
        )

        # Record in TradeTracker
        self.active_trade_id = self.trade_tracker.open_trade(
            date=current_dt,
            entry_price=current_price,
            quantity=quantity,
            take_profit=final_tp,
            stop_loss=final_sl,
            support_level=sup,
            resistance_level=res,
            trade_type=trade_type
        )

        # Submit Lumibot bracket order
        order = self.create_order(
            ticker,
            quantity,
            side,
            take_profit_price=final_tp,
            stop_loss_price=final_sl,
            type="market"
        )
        self.submit_order(order)
        self.has_entered_for_signal = True

    def on_strategy_end(self):
        """Cleanly liquidate or finalize open trades at end of simulation."""
        ticker = self.parameters["Ticker"]
        current_price = self.get_last_price(ticker)
        current_dt = self.get_datetime()

        if self.active_trade_id and self.active_trade_id in self.trade_tracker.trades:
            trade = self.trade_tracker.trades[self.active_trade_id]
            if trade.is_open:
                self.log_message(f"[{ticker.symbol}] Finalizing simulation. Closing trade @ ${current_price:.2f}")
                self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "END_SIM")
                self._close_all_positions()

        # Log tracker summary
        perf = self.trade_tracker.get_performance_summary()
        self.log_message(f"[{ticker.symbol}] Performance Summary: {perf}")

    def _close_all_positions(self):
        ticker = self.parameters["Ticker"]
        position = self.get_position(ticker)
        if position is not None and position.quantity != 0:
            side = "sell" if position.quantity > 0 else "buy"
            order = self.create_order(ticker, abs(position.quantity), side, type="market")
            self.submit_order(order)
