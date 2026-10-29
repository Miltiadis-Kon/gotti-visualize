"""
Market Reactor Strategy (market_reactor.py)
============================================
Connects Stock Alchemist intelligence signals (fundamental & news sentiment)
with automated market execution using Lumibot, institutional Key Levels clustering,
and the unified StrategyBaseplate ATR volatility risk framework.

Upgraded Architecture:
1. Dual-Cohort Timestamp Routing (Zero Lookahead):
   - Pre-RTH Cohort (<= 09:30 EST): Traded via 9:30-9:45 AM Opening Range Breakout (ORB)
     with a Daily Gap-to-ATR Exhaustion Guard.
   - Intraday RTH Cohort (09:30 - 15:30 EST): Traded dynamically after signal arrival
     using a 5m Catalyst Arrival Bar breakout & RVOL filter.
   - Post-RTH Cohort (>= 16:00 EST): Deferred to the next trading session's morning ORB.
2. Three Regime-Specific Selectors:
   - Selector 1 (Macro Alignment): News & Fundamentals agree -> 1H S/R + 1H ATR
     ("swing_trading" style, overnight allowed, breakeven stop ratchet at +1.5R).
   - Selector 2 (Catalyst Momentum / Fade): News opposes Fundamentals -> 15m S/R + 15m ATR
     ("day_trading" / "scalping" style, supports "momentum", "fade", and "dynamic" modes,
     RVOL > 2.0 check, strict EOD flatten at 15:55 EST).
   - Selector 3 (Directional Flow): News with Neutral/HOLD Fundamentals -> 15m S/R + 15m ATR
     ("day_trading" style, volume-confirmed breakout, conditional overnight only if >= +0.5R at 15:50 EST).
3. Conviction-Weighted Risk & ATR Fallbacks:
   - Scales portfolio risk dynamically between 1.0% and 2.5% based on Composite Conviction.
   - Replaces static 2% S/R fallbacks with dynamic ATR volatility projections on unmapped breakouts.
"""

import os
import sys
import json
import logging
from math import floor, sqrt
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
    signal_position: str          # Raw DB position ("BUY" or "SELL" - follows News)
    fundamental_sentiment: str    # e.g. "BUY", "SELL [VALUE TRAP]", "HOLD"
    news_sentiment: str           # "POSITIVE" or "NEGATIVE"
    headline: str
    fundamental_confidence: float # 0.0 to 1.0 (from fundamental_analysis.data)
    news_conviction: float        # 0.0 to 1.0 (from gemini rating & impact)
    composite_conviction: float   # sqrt(fund_conf * news_conv) for dynamic sizing
    is_eligible: bool             # True if BOTH >= 0.70
    selector_id: int              # 1, 2, or 3
    selector_name: str            # "Macro Alignment", "Catalyst Momentum", "Directional Flow"
    sr_timeframe: str             # "1H" or "15m"
    resolution_code: str          # "1H" or "15m" (supported by KeyLevelDetector)
    trading_style: str            # "swing_trading", "day_trading", or "scalping"
    gemini_rating: int            # 1 to 5
    gemini_impact: str            # "High", "Medium", "Low"
    raw_signal_id: str
    signal_date: Optional[str] = None
    created_at: Optional[datetime] = None


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

        # Extract Fundamental Confidence
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
        composite_conv = round(sqrt(max(0.0, fund_conf) * max(0.0, news_conv)), 4)

        # Parse Directional Alignment
        is_news_bullish = news_sentiment in ("POSITIVE", "BULLISH")
        is_news_bearish = news_sentiment in ("NEGATIVE", "BEARISH")

        is_fund_bullish = (
            ("BUY" in fund_sentiment or "POSITIVE" in fund_sentiment or "UNDERVALUED" in fund_sentiment)
            and ("SELL" not in fund_sentiment)
        )
        is_fund_bearish = (
            "SELL" in fund_sentiment
            or "NEGATIVE" in fund_sentiment
            or "VALUE TRAP" in fund_sentiment
            or "VETO" in fund_sentiment
            or "DISTRESS" in fund_sentiment
        )

        # Map into the 3 Selectors (using KeyLevelDetector supported resolutions: '1H' and '15m')
        if (is_news_bullish and is_fund_bullish) or (is_news_bearish and is_fund_bearish):
            # Selector 1: Macro Alignment (1H S/R, Swing Trading)
            selector_id = 1
            selector_name = "Macro Alignment"
            sr_timeframe = "1H"
            resolution_code = "1H"
            trading_style = "swing_trading"
        elif (is_news_bullish and is_fund_bearish) or (is_news_bearish and is_fund_bullish):
            # Selector 2: Catalyst Momentum Opposing Fundamentals (15m S/R, Day Trading / Scalping)
            selector_id = 2
            selector_name = "Catalyst Momentum"
            sr_timeframe = "15m"
            resolution_code = "15m"
            trading_style = "day_trading"
        else:
            # Selector 3: Directional Flow on Neutral / Hold Fundamentals (15m S/R, Day Trading)
            selector_id = 3
            selector_name = "Directional Flow"
            sr_timeframe = "15m"
            resolution_code = "15m"
            trading_style = "day_trading"

        # Parse created_at timestamp
        raw_created = raw_row.get("created_at")
        created_dt: Optional[datetime] = None
        if isinstance(raw_created, datetime):
            created_dt = raw_created
        elif isinstance(raw_created, str) and raw_created.strip():
            try:
                created_dt = pd.to_datetime(raw_created).to_pydatetime()
            except Exception:
                created_dt = None

        raw_sig_date = raw_row.get("signal_date")
        sig_date_str = str(raw_sig_date) if raw_sig_date is not None else None

        return ReactorSignal(
            ticker=ticker,
            signal_position=signal_pos,
            fundamental_sentiment=fund_sentiment,
            news_sentiment=news_sentiment,
            headline=headline,
            fundamental_confidence=round(fund_conf, 4),
            news_conviction=round(news_conv, 4),
            composite_conviction=composite_conv,
            is_eligible=is_eligible,
            selector_id=selector_id,
            selector_name=selector_name,
            sr_timeframe=sr_timeframe,
            resolution_code=resolution_code,
            trading_style=trading_style,
            gemini_rating=gemini_rating,
            gemini_impact=gemini_impact,
            raw_signal_id=str(raw_row.get("signal_id", "")),
            signal_date=sig_date_str,
            created_at=created_dt,
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


def fetch_reactor_signals(
    target_date: str,
    ticker: Optional[str] = None,
    include_prev_afterhours: bool = True
) -> List[ReactorSignal]:
    """
    Fetches signals for a given target date from MySQL `signals` table.
    If `include_prev_afterhours` is True, also fetches prior-day after-hours signals
    (created_at >= 16:00 EST) so overnight catalysts can be traded at the 9:30 AM open.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    target_dt = pd.to_datetime(target_date).date()
    prev_dt = target_dt - timedelta(days=3 if target_dt.weekday() == 0 else 1)

    params: List[Any] = [str(target_dt)]
    where_clause = "s.signal_date = %s"

    if include_prev_afterhours:
        where_clause = "(s.signal_date = %s OR (s.signal_date = %s AND TIME(s.created_at) >= '16:00:00'))"
        params = [str(target_dt), str(prev_dt)]

    if ticker:
        where_clause += " AND s.ticker = %s"
        params.append(ticker.strip().upper())

    query = f"""
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
    WHERE {where_clause}
    ORDER BY s.created_at ASC, s.updated_at ASC;
    """

    cursor.execute(query, tuple(params))
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
    Upgraded Lumibot Market Reactor Strategy.
    Features:
    - 15-Minute Opening Range Breakout (ORB) confirmation for morning signals.
    - Dynamic Intraday Catalyst Bar breakout for mid-session signals.
    - Gap-to-ATR Exhaustion Guard (> 1.5x Daily ATR gap protection).
    - Selector 1 (Macro Swing), Selector 2 (Momentum/Fade Scalp), Selector 3 (Directional Flow).
    - Conviction-weighted position sizing (1.0% to 2.5% risk).
    - Broker-managed bracket execution with EOD & breakeven ratchet management.
    """

    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "SignalDate": "2026-10-28",         # Fallback date if not dynamically tracking
        "DynamicDateBinding": True,         # Automatically reload signals as simulation date rolls over
        "EnforceNoLookahead": True,         # Only trade intraday signals at/after their created_at timestamp
        "MinConfidence": 0.70,              # Dual hard floor threshold
        "BaseRiskPct": 0.015,               # Base risk at 70% floor (scales up to MaxRiskPct)
        "MaxRiskPct": 0.025,                # Max risk at 100% composite conviction
        "MinRR": 1.5,                       # Minimum Risk-to-Reward ratio
        "Selector2_Mode": "dynamic",        # "momentum", "fade", or "dynamic"
        "Selector2_MinRVOL": 1.8,           # Minimum Relative Volume for Selector 2 momentum breakouts
        "Selector3_MinRVOL": 1.3,           # Minimum Relative Volume for Selector 3 breakouts
        "MaxGapATRMultiple": 1.5,           # Gap-to-ATR ratio that triggers Exhaustion Guard
        "ORBMinutes": 15,                   # Opening Range duration (9:30 - 9:45 AM EST)
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"

        # Configuration parameters
        self.dynamic_date_binding = bool(self.parameters.get("DynamicDateBinding", True))
        self.enforce_no_lookahead = bool(self.parameters.get("EnforceNoLookahead", True))
        self.base_risk_pct = float(self.parameters.get("BaseRiskPct", 0.015))
        self.max_risk_pct = float(self.parameters.get("MaxRiskPct", 0.025))
        self.min_rr = float(self.parameters.get("MinRR", 1.5))
        self.selector2_mode = str(self.parameters.get("Selector2_Mode", "dynamic")).lower()
        self.selector2_min_rvol = float(self.parameters.get("Selector2_MinRVOL", 1.8))
        self.selector3_min_rvol = float(self.parameters.get("Selector3_MinRVOL", 1.3))
        self.max_gap_atr = float(self.parameters.get("MaxGapATRMultiple", 1.5))
        self.orb_minutes = int(self.parameters.get("ORBMinutes", 15))

        # Trade tracker
        self.trade_tracker = TradeTracker(
            ticker=self.parameters["Ticker"].symbol,
            initial_cash=self.get_portfolio_value()
        )
        self.active_trade_id: Optional[str] = None
        self.active_trade_selector: Optional[int] = None
        self.active_entry_price: float = 0.0
        self.active_initial_risk: float = 0.0
        self.breakeven_ratchet_applied: bool = False

        # Daily session state
        self.current_session_date: Optional[date] = None
        self.session_signals: List[ReactorSignal] = []
        self.matched_signal: Optional[ReactorSignal] = None
        self.traded_signal_ids: set = set()

        # Opening Range & Gap metrics for current session
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.open_930_price: Optional[float] = None
        self.prev_close_price: Optional[float] = None
        self.daily_atr: Optional[float] = None
        self.gap_ratio: float = 0.0
        self.is_exhausted_gap: bool = False

        # Initial load for static SignalDate fallback
        self.signal_date_str = str(self.parameters.get("SignalDate", "2026-10-28"))
        self._load_signals_for_date(self.signal_date_str)

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION & SIGNAL BINDING
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_daily_session_state(self, new_date: date):
        """Resets intraday Opening Range and binds signals for the new trading day."""
        self.current_session_date = new_date
        self.or_high = None
        self.or_low = None
        self.open_930_price = None
        self.prev_close_price = None
        self.daily_atr = None
        self.gap_ratio = 0.0
        self.is_exhausted_gap = False
        self.matched_signal = None

        target_date_str = new_date.strftime("%Y-%m-%d") if self.dynamic_date_binding else self.signal_date_str
        self._load_signals_for_date(target_date_str)

    def _load_signals_for_date(self, target_date_str: str):
        ticker_sym = self.parameters["Ticker"].symbol
        all_signals = fetch_reactor_signals(
            target_date=target_date_str,
            ticker=ticker_sym,
            include_prev_afterhours=True
        )

        # Filter out signals that occurred after 16:00 EST on target_date (belong to tomorrow's open)
        valid_session_signals: List[ReactorSignal] = []
        for s in all_signals:
            if s.raw_signal_id in self.traded_signal_ids:
                continue
            if self.enforce_no_lookahead and s.created_at is not None and s.signal_date == target_date_str:
                if s.created_at.time() >= time(16, 0):
                    continue
            valid_session_signals.append(s)

        self.session_signals = valid_session_signals
        if not self.session_signals:
            return

        # Select latest eligible signal available right now (or pre-market)
        eligible = [s for s in self.session_signals if s.is_eligible]
        self.matched_signal = eligible[-1] if eligible else self.session_signals[-1]

        sig = self.matched_signal
        status = "ELIGIBLE" if sig.is_eligible else "REJECTED (<70% dual floor)"
        self.log_message(
            f"[{ticker_sym}] Session {target_date_str} Signal: {sig.signal_position} | "
            f"Fund: {sig.fundamental_sentiment} ({sig.fundamental_confidence:.1%}) | "
            f"News: {sig.news_sentiment} ({sig.news_conviction:.1%}) | "
            f"Composite: {sig.composite_conviction:.1%} | "
            f"Selector {sig.selector_id} ({sig.selector_name}, TF: {sig.sr_timeframe}) -> {status}"
        )

    def _get_available_signal(self, current_dt: datetime) -> Optional[ReactorSignal]:
        """
        Returns the highest-priority eligible signal that has already arrived by `current_dt`
        and has not yet been traded.
        """
        eligible_now: List[ReactorSignal] = []
        cur_date_str = current_dt.strftime("%Y-%m-%d")

        for s in self.session_signals:
            if not s.is_eligible or s.raw_signal_id in self.traded_signal_ids:
                continue

            if self.enforce_no_lookahead and s.created_at is not None:
                # If created on the current session date, ensure current time >= created_at time
                if s.signal_date == cur_date_str and current_dt.time() < s.created_at.time():
                    continue

            eligible_now.append(s)

        if eligible_now:
            self.matched_signal = eligible_now[-1]
            return self.matched_signal
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # QUANTITATIVE METRICS: ATR, S/R, RVOL, OPENING RANGE & GAP GUARD
    # ─────────────────────────────────────────────────────────────────────────

    def _get_atr(self, resolution: str = "15m") -> float:
        """Calculates 14-period ATR matched to the active Selector's timeframe."""
        ticker = self.parameters["Ticker"]
        lumibot_tf = "hour" if resolution in ("1H", "4H") else ("day" if resolution == "1D" else "15m")
        try:
            bars = self.get_historical_prices(ticker, 30, lumibot_tf)
            if bars is not None:
                df = bars.df if hasattr(bars, "df") else bars
                if not df.empty and len(df) >= 14:
                    return float(calculate_atr(df, length=14))
        except Exception:
            pass
        last_px = self.get_last_price(ticker) or 100.0
        return float(last_px * (0.025 if lumibot_tf in ("hour", "day") else 0.012))

    def _compute_daily_gap_metrics(self, current_price: float):
        """Calculates 9:30 AM Opening Gap relative to 14-day Daily ATR."""
        if self.open_930_price is not None:
            return

        self.open_930_price = current_price
        ticker = self.parameters["Ticker"]
        try:
            daily_bars = self.get_historical_prices(ticker, 20, "day")
            if daily_bars is not None:
                df = daily_bars.df if hasattr(daily_bars, "df") else daily_bars
                if not df.empty and len(df) >= 2:
                    self.prev_close_price = float(df["close"].iloc[-2])
                    self.daily_atr = float(calculate_atr(df, length=min(14, len(df) - 1)))
                    if self.daily_atr > 0:
                        self.gap_ratio = abs(self.open_930_price - self.prev_close_price) / self.daily_atr
                        self.is_exhausted_gap = self.gap_ratio > self.max_gap_atr
                        self.log_message(
                            f"[{ticker.symbol}] Morning Gap Check: Open=${self.open_930_price:.2f}, "
                            f"PrevClose=${self.prev_close_price:.2f}, 1D_ATR=${self.daily_atr:.2f} -> "
                            f"Gap Ratio={self.gap_ratio:.2f}x ({'EXHAUSTED' if self.is_exhausted_gap else 'NORMAL'})"
                        )
        except Exception as e:
            self.log_message(f"[{ticker.symbol}] Could not compute daily gap metrics: {e}")

    def _update_opening_range(self):
        """Computes the 9:30 - 9:45 AM EST Opening Range High and Low from 5m bars."""
        if self.or_high is not None and self.or_low is not None:
            return

        ticker = self.parameters["Ticker"]
        try:
            bars = self.get_historical_prices(ticker, 10, "5m")
            if bars is not None:
                df = bars.df if hasattr(bars, "df") else bars
                if not df.empty and self.current_session_date is not None:
                    # Filter bars belonging to today's 09:30 - 09:45 window
                    idx_dates = pd.to_datetime(df.index)
                    today_mask = (idx_dates.date == self.current_session_date) & \
                                 (idx_dates.time >= time(9, 30)) & \
                                 (idx_dates.time < time(9, 30 + self.orb_minutes))
                    or_df = df[today_mask]
                    if not or_df.empty:
                        self.or_high = float(or_df["high"].max())
                        self.or_low = float(or_df["low"].min())
                        self.log_message(
                            f"[{ticker.symbol}] {self.orb_minutes}m Opening Range Locked: "
                            f"OR_High=${self.or_high:.2f} | OR_Low=${self.or_low:.2f}"
                        )
                        return
        except Exception:
            pass

        # Fallback if 5m bar slice unavailable
        px = self.get_last_price(ticker)
        if px:
            self.or_high = px * 1.002
            self.or_low = px * 0.998

    def _get_rvol(self) -> float:
        """Calculates current 5m/15m Relative Volume against the 20-bar rolling average."""
        ticker = self.parameters["Ticker"]
        try:
            bars = self.get_historical_prices(ticker, 25, "5m")
            if bars is not None:
                df = bars.df if hasattr(bars, "df") else bars
                if not df.empty and "volume" in df.columns and len(df) >= 10:
                    avg_vol = float(df["volume"].iloc[:-1].tail(20).mean())
                    cur_vol = float(df["volume"].iloc[-1])
                    if avg_vol > 0:
                        return round(cur_vol / avg_vol, 2)
        except Exception:
            pass
        return 1.5  # Neutral fallback if volume feed unavailable

    def _get_sr_levels(self, current_price: float, resolution: str, atr_val: float) -> Tuple[float, float, bool, bool]:
        """
        Calculates Support and Resistance levels for the target resolution ('1H' or '15m').
        If price gaps beyond all historical S/R levels, projects dynamic ATR targets
        instead of arbitrary 2% buffers.
        Returns: (support, resistance, has_real_support, has_real_resistance)
        """
        ticker_sym = self.parameters["Ticker"].symbol
        valid_res = resolution if resolution in ("1D", "4H", "1H", "15m", "5m") else "15m"

        try:
            res = analyze(ticker_sym, resolutions=[valid_res], use_alpaca=False)
            kl_df = res.key_levels

            if kl_df is not None and not kl_df.empty:
                supports = kl_df[(kl_df["type"] == "support") & (kl_df["level_price"] < current_price)]["level_price"].values
                resistances = kl_df[(kl_df["type"] == "resistance") & (kl_df["level_price"] > current_price)]["level_price"].values

                has_sup = len(supports) > 0
                has_res = len(resistances) > 0

                # Dynamic ATR fallback if price gapped outside historical cluster range
                sup = float(np.max(supports)) if has_sup else (self.or_low if (self.or_low and self.or_low < current_price) else current_price - (1.5 * atr_val))
                res_lvl = float(np.min(resistances)) if has_res else current_price + (2.5 * atr_val)
                return sup, res_lvl, has_sup, has_res
        except Exception as e:
            self.log_message(f"[{ticker_sym}] Key levels error ({valid_res}): {e}. Using ATR projection.")

        return current_price - (1.5 * atr_val), current_price + (2.5 * atr_val), False, False

    def get_position_sizing(self, entry_price: float, stop_loss: float, sig: ReactorSignal) -> int:
        """
        Conviction-weighted position sizing (1.0% to 2.5% risk),
        halved automatically if Gap-to-ATR Exhaustion Guard is active.
        """
        portfolio_val = self.get_portfolio_value()

        # Scale risk linearly from 0.70 floor -> BaseRiskPct up to 1.00 -> MaxRiskPct
        conv_norm = max(0.0, min(1.0, (sig.composite_conviction - 0.70) / 0.30))
        effective_risk_pct = self.base_risk_pct + conv_norm * (self.max_risk_pct - self.base_risk_pct)

        if self.is_exhausted_gap:
            effective_risk_pct *= 0.50

        risk_budget = portfolio_val * effective_risk_pct
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0

        qty = floor(risk_budget / risk_per_share)
        max_affordable = floor((portfolio_val * 0.95) / max(0.01, entry_price))
        return max(0, min(qty, max_affordable))

    # ─────────────────────────────────────────────────────────────────────────
    # SELECTOR REGIME ENTRY EVALUATION
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_entry_trigger(
        self,
        sig: ReactorSignal,
        current_price: float,
        current_dt: datetime,
        sup: float,
        res: float,
        atr_val: float
    ) -> Tuple[bool, str, str]:
        """
        Evaluates whether the regime-specific entry conditions for Selectors 1, 2, or 3 are met.
        Returns: (should_enter, resolved_trade_type ["BUY"|"SELL"], reason_note)
        """
        raw_dir = sig.signal_position  # "BUY" or "SELL" (always matches News direction)
        rvol = self._get_rvol()

        # Determine if this is an Intraday Arrival (> 09:30 EST) vs Morning ORB catalyst
        is_intraday_catalyst = (
            sig.created_at is not None
            and self.current_session_date is not None
            and sig.signal_date == self.current_session_date.strftime("%Y-%m-%d")
            and sig.created_at.time() > time(9, 30)
        )

        # ── COHORT B: INTRADAY BREAKING CATALYST (09:30 - 15:25 EST) ─────────
        if is_intraday_catalyst:
            if current_dt.time() >= time(15, 25):
                return False, raw_dir, "Intraday signal arrived too close to market close (>=15:25)"

            # Wait at least 5 minutes after signal creation for initial spread stabilization
            elapsed_mins = (current_dt - sig.created_at.replace(tzinfo=current_dt.tzinfo)).total_seconds() / 60.0
            if elapsed_mins < 4.0:
                return False, raw_dir, "Waiting for 5m intraday catalyst arrival bar to close"

            if sig.selector_id in (2, 3) and rvol < self.selector3_min_rvol:
                return False, raw_dir, f"Intraday RVOL ({rvol:.2f}) below threshold ({self.selector3_min_rvol})"

            return True, raw_dir, f"Intraday Catalyst Confirmed (RVOL={rvol:.2f})"

        # ── COHORT A: MORNING GAP / ORB CATALYST (Requires >= 09:45 EST) ─────
        if self.or_high is None or self.or_low is None:
            return False, raw_dir, "Opening Range not yet established"

        # 1. SELECTOR 1: Macro Alignment (News + Fundamentals Agree)
        if sig.selector_id == 1:
            if raw_dir == "BUY":
                # Enter on ORB breakout OR healthy pullback holding above OR_Low / 1H Support
                if current_price >= self.or_high:
                    return True, "BUY", "Selector 1 ORB High Breakout"
                if not self.is_exhausted_gap and current_price > self.or_low and (current_price - self.or_low) <= 0.5 * atr_val:
                    return True, "BUY", "Selector 1 Opening Range Support Hold"
                return False, "BUY", "Price below OR_Low (Sell-the-news rejection)"
            else:
                if current_price <= self.or_low:
                    return True, "SELL", "Selector 1 ORB Low Breakdown"
                if not self.is_exhausted_gap and current_price < self.or_high and (self.or_high - current_price) <= 0.5 * atr_val:
                    return True, "SELL", "Selector 1 Opening Range Resistance Hold"
                return False, "SELL", "Price above OR_High (Short squeeze rejection)"

        # 2. SELECTOR 2: Catalyst Momentum vs. Fundamental Fade
        elif sig.selector_id == 2:
            fundamental_dir = "SELL" if raw_dir == "BUY" else "BUY"

            # Mode A: Momentum Scalp (in direction of News)
            if self.selector2_mode in ("momentum", "dynamic"):
                if raw_dir == "BUY" and current_price > self.or_high and rvol >= self.selector2_min_rvol and not self.is_exhausted_gap:
                    return True, "BUY", f"Selector 2 Momentum Squeeze (ORB Break + RVOL {rvol:.2f})"
                if raw_dir == "SELL" and current_price < self.or_low and rvol >= self.selector2_min_rvol and not self.is_exhausted_gap:
                    return True, "SELL", f"Selector 2 Momentum Breakdown (ORB Break + RVOL {rvol:.2f})"

            # Mode B: Fundamental Fade (when morning news spike stalls & breaks OR opposite boundary)
            if self.selector2_mode in ("fade", "dynamic"):
                if raw_dir == "BUY" and current_price < self.or_low:
                    return True, "SELL", f"Selector 2 Fundamental Fade (Bullish news failed OR_Low -> {fundamental_dir})"
                if raw_dir == "SELL" and current_price > self.or_high:
                    return True, "BUY", f"Selector 2 Fundamental Fade (Bearish news squeezed above OR_High -> {fundamental_dir})"

            return False, raw_dir, f"Selector 2 waiting for ORB confirmation (Mode={self.selector2_mode}, RVOL={rvol:.2f})"

        # 3. SELECTOR 3: Directional Flow (Neutral Fundamentals)
        else:
            if raw_dir == "BUY" and current_price >= self.or_high and rvol >= self.selector3_min_rvol:
                return True, "BUY", f"Selector 3 Volume-Confirmed ORB Breakout (RVOL={rvol:.2f})"
            if raw_dir == "SELL" and current_price <= self.or_low and rvol >= self.selector3_min_rvol:
                return True, "SELL", f"Selector 3 Volume-Confirmed ORB Breakdown (RVOL={rvol:.2f})"

            return False, raw_dir, f"Selector 3 awaiting ORB break with RVOL >= {self.selector3_min_rvol} (cur={rvol:.2f})"

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TRADING LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        ticker = self.parameters["Ticker"]
        ticker_sym = ticker.symbol
        current_price = self.get_last_price(ticker)
        current_dt = self.get_datetime()

        if current_price is None or current_price <= 0:
            return

        # ── 0. DAILY ROLLOVER & SESSION INITIALIZATION ───────────────────────
        if self.current_session_date != current_dt.date():
            self._reset_daily_session_state(current_dt.date())

        current_time = current_dt.time()
        if current_time >= time(9, 30):
            self._compute_daily_gap_metrics(current_price)
        if current_time >= time(9, 30 + self.orb_minutes):
            self._update_opening_range()

        # ── 1. SYNC BROKER BRACKET EXITS & MANAGE OPEN POSITIONS ─────────────
        position = self.get_position(ticker)
        has_open_pos = position is not None and position.quantity != 0

        # Detect if broker bracket order (TP or SL) filled since last iteration
        if not has_open_pos and self.active_trade_id is not None:
            if self.active_trade_id in self.trade_tracker.trades:
                trade = self.trade_tracker.trades[self.active_trade_id]
                if trade.is_open:
                    dist_tp = abs(current_price - trade.take_profit)
                    dist_sl = abs(current_price - trade.stop_loss)
                    exit_reason = "TP" if dist_tp <= dist_sl else "SL"
                    exit_px = trade.take_profit if exit_reason == "TP" else trade.stop_loss
                    self.log_message(f"[{ticker_sym}] Bracket {exit_reason} filled @ ~${exit_px:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, exit_px, exit_reason)
            self.active_trade_id = None
            self.active_trade_selector = None
            self.breakeven_ratchet_applied = False

        # Active Position Management (EOD Rules & Breakeven Ratchet)
        if has_open_pos and self.active_trade_id in self.trade_tracker.trades:
            trade = self.trade_tracker.trades[self.active_trade_id]
            signed_pnl_per_share = (current_price - trade.entry_price) if trade.trade_type == "BUY" else (trade.entry_price - current_price)
            current_r_mult = (signed_pnl_per_share / self.active_initial_risk) if self.active_initial_risk > 0 else 0.0

            # Selector 2: Strict Intraday Flatten at 15:55 EST
            if self.active_trade_selector == 2 and current_time >= time(15, 55):
                self.log_message(f"[{ticker_sym}] Selector 2 Hard EOD Flatten @ ${current_price:.2f} ({current_r_mult:+.2f}R)")
                self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "EOD_SEL2")
                self._close_all_positions()
                return

            # Selector 3: Conditional Overnight Check at 15:50 EST (Require >= +0.5R)
            if self.active_trade_selector == 3 and current_time >= time(15, 50) and current_r_mult < 0.50:
                self.log_message(
                    f"[{ticker_sym}] Selector 3 EOD Flatten @ ${current_price:.2f} "
                    f"(Unrealized {current_r_mult:+.2f}R < +0.50R overnight threshold)"
                )
                self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "EOD_SEL3_WEAK")
                self._close_all_positions()
                return

            # Selector 1: Ratchet internal stop to Breakeven once trade reaches +1.5R
            if self.active_trade_selector == 1 and not self.breakeven_ratchet_applied and current_r_mult >= 1.50:
                trade.stop_loss = trade.entry_price
                self.breakeven_ratchet_applied = True
                self.log_message(f"[{ticker_sym}] Selector 1 reached +1.5R -> Stop loss ratcheted to breakeven (${trade.entry_price:.2f})")

            # Enforce ratcheted breakeven stop if price pulls back
            if self.breakeven_ratchet_applied:
                if (trade.trade_type == "BUY" and current_price <= trade.stop_loss) or \
                   (trade.trade_type == "SELL" and current_price >= trade.stop_loss):
                    self.log_message(f"[{ticker_sym}] Breakeven Stop Triggered @ ${current_price:.2f}")
                    self.trade_tracker.close_trade(self.active_trade_id, current_dt, current_price, "BE_STOP")
                    self._close_all_positions()
            return

        # ── 2. NEW ENTRY EVALUATION ──────────────────────────────────────────
        # Wait until Opening Range completes at 9:45 AM EST (and don't open new trades after 15:30 EST)
        if current_time < time(9, 30 + self.orb_minutes) or current_time >= time(15, 30):
            return

        sig = self._get_available_signal(current_dt)
        if sig is None:
            return

        atr_val = self._get_atr(sig.resolution_code)
        sup, res, _, _ = self._get_sr_levels(current_price, sig.resolution_code, atr_val)

        should_enter, resolved_trade_type, entry_note = self._evaluate_entry_trigger(
            sig=sig,
            current_price=current_price,
            current_dt=current_dt,
            sup=sup,
            res=res,
            atr_val=atr_val
        )

        if not should_enter:
            return

        side = "buy" if resolved_trade_type == "BUY" else "sell"
        raw_sl = sup if resolved_trade_type == "BUY" else res
        raw_tp = res if resolved_trade_type == "BUY" else sup

        # Optimize SL/TP using Selector-specific TradingStyle & ATR
        risk_levels = self.optimize_sl_tp(
            entry_price=current_price,
            stop_loss=raw_sl,
            take_profit=raw_tp,
            side=side,
            trading_style=sig.trading_style,
            atr=atr_val,
            risk_reward_ratio=self.min_rr
        )

        final_sl = risk_levels.stop_loss
        final_tp = risk_levels.take_profit

        quantity = self.get_position_sizing(current_price, final_sl, sig)
        if quantity <= 0:
            self.log_message(f"[{ticker_sym}] Position sizing resulted in 0 shares. Skipping signal {sig.raw_signal_id}.")
            self.traded_signal_ids.add(sig.raw_signal_id)
            return

        self.log_message(
            f"[{ticker_sym}] EXECUTING {resolved_trade_type} ({entry_note}) | "
            f"Qty: {quantity} @ ${current_price:.2f} | SL: ${final_sl:.2f} | TP: ${final_tp:.2f} | "
            f"R:R: {risk_levels.risk_reward_ratio:.2f} | Style: {sig.trading_style} | "
            f"Selector {sig.selector_id} ({sig.selector_name}, TF: {sig.sr_timeframe})"
        )

        self.active_trade_id = self.trade_tracker.open_trade(
            date=current_dt,
            entry_price=current_price,
            quantity=quantity,
            take_profit=final_tp,
            stop_loss=final_sl,
            support_level=sup,
            resistance_level=res,
            trade_type=resolved_trade_type
        )
        self.active_trade_selector = sig.selector_id
        self.active_entry_price = current_price
        self.active_initial_risk = abs(current_price - final_sl)
        self.breakeven_ratchet_applied = False
        self.traded_signal_ids.add(sig.raw_signal_id)

        order = self.create_order(
            ticker,
            quantity,
            side,
            take_profit_price=final_tp,
            stop_loss_price=final_sl,
            type="market"
        )
        self.submit_order(order)

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

        perf = self.trade_tracker.get_performance_summary()
        self.log_message(f"[{ticker.symbol}] Performance Summary: {perf}")

    def _close_all_positions(self):
        """Cancels working bracket legs before flattening to prevent orphan fills."""
        self.cancel_open_orders()
        ticker = self.parameters["Ticker"]
        position = self.get_position(ticker)
        if position is not None and position.quantity != 0:
            side = "sell" if position.quantity > 0 else "buy"
            order = self.create_order(ticker, abs(position.quantity), side, type="market")
            self.submit_order(order)
        self.active_trade_id = None
        self.active_trade_selector = None
        self.breakeven_ratchet_applied = False
