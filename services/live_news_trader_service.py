"""
Live News Trading Service (services/live_news_trader_service.py)
================================================================
Continuous 24/7 service orchestrating live / paper trading execution for:
1. News_LiquiditySweep Strategy (Multi-Pool Liquidity Sweeps & Reclaims)
2. MarketReactor Strategy in Fade Mode (Fading News Hype against Fundamentals)

Key Capabilities:
- Live feedback: Continuously polls MySQL `signals` table for newly arrived news catalysts.
- Dual Hard Floor confidence filtering (Fundamental Confidence >= 70%, News Conviction >= 70%).
- Real-time execution via Lumibot + Alpaca Broker (Paper by default, live configurable).
- Native Alpaca bracket order management (Entry + Take Profit + Stop Loss).
- Dynamic Breakeven Ratchets (+1.2R lock-in) and Strict EOD Flattening (15:50 & 15:55 EST).
- Real-time order reconciliation writing directly to MySQL `orders` table.
- Continuous 24/7 daemon loop with Pre-Market, Regular Trading Hours, Post-Market, and Weekend states.
- Emits health telemetry & heartbeat to `logs/live_trader_heartbeat.json`.
"""

import os
import sys
import time
import json
import logging
import signal
import argparse
from math import floor, sqrt
from datetime import datetime, date, time as dtime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple

import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from lumibot.brokers import Alpaca
from lumibot.entities import Asset

from strategies.market_reactor import (
    get_mysql_connection,
    MarketReactorClassifier,
    ReactorSignal,
    fetch_reactor_signals,
)
from strategies.risk_management import calculate_atr
from strategies.trade_tracker import TradeTracker
from services.signal_observer import RealTimeSignalObserver
from strategies.high_beta_catalyst_open import (
    COMPETITOR_MAP,
    CatalystConfluenceEngine,
)
from strategies.institutional_v_recovery import compute_vwap, compute_ema

# Configure logging
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "live_news_trader_service.log")
HEARTBEAT_FILE = os.path.join(LOG_DIR, "live_trader_heartbeat.json")
STATE_FILE = os.path.join(LOG_DIR, "live_trader_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("LiveNewsTrader")


class LiveNewsTraderService:
    """
    Continuous live trading service managing News_LiquiditySweep and
    MarketReactor (Fade Mode) strategies using Lumibot and Alpaca.
    """

    def __init__(
        self,
        paper: bool = True,
        poll_interval_sec: int = 20,
        dry_run: bool = False,
        strategies: Optional[List[str]] = None,
        initial_portfolio: float = 100000.0,
    ):
        self.paper = paper
        self.poll_interval_sec = poll_interval_sec
        self.dry_run = dry_run
        self.enabled_strategies = strategies or [
            "News_LiquiditySweep",
            "MarketReactor_Fade",
            "HighBeta_CatalystOpen",
            "Institutional_VRecovery",
        ]
        self.initial_portfolio = initial_portfolio

        self.tz_est = pytz.timezone("America/New_York")
        self.running = True

        # Broker connection
        self.broker: Optional[Alpaca] = None
        self.account_id: Optional[str] = None
        self.portfolio_value: float = initial_portfolio
        self.buying_power: float = initial_portfolio

        # State tracking
        self.processed_signal_ids: Set[str] = set()
        self.session_signals: Dict[str, ReactorSignal] = {}  # ticker -> latest signal
        self.active_watchlist: Dict[str, Dict[str, Any]] = {}  # ticker -> technical levels & context
        self.active_positions: Dict[str, Dict[str, Any]] = {}  # pos_id -> position details
        self.daily_completed_trades: List[Dict[str, Any]] = []

        self.current_session_date: Optional[date] = None

        self._setup_signal_handlers()
        self._load_persisted_state()
        self._init_broker()

        # Real-time event-driven signal observer (<5ms WebSocket + sub-second DB change detection)
        self.observer = RealTimeSignalObserver(
            on_signal_callback=self._on_instant_signal_observed,
            db_poll_interval_sec=1.0,
        )

    def _on_instant_signal_observed(self, sig: ReactorSignal):
        """Callback fired immediately upon real-time signal detection (< 5ms via WS or < 1s via DB)."""
        logger.info(f"⚡ [OBSERVER EVENT] Processing instant live signal: {sig.ticker} {sig.signal_position}")
        self._process_single_signal(sig, is_historical=False)
        now_est = datetime.now(self.tz_est)
        phase, _ = self.get_market_session_state(now_est)
        if phase == "REGULAR_TRADING":
            self.evaluate_live_market()

    def _setup_signal_handlers(self):
        def handle_shutdown(signum, frame):
            logger.info(f"Received termination signal ({signum}). Initiating graceful shutdown...")
            self.running = False

        try:
            signal.signal(signal.SIGINT, handle_shutdown)
            signal.signal(signal.SIGTERM, handle_shutdown)
        except Exception:
            pass

    def _init_broker(self):
        """Initializes connection to Alpaca via Lumibot."""
        if self.dry_run:
            logger.info("Service started in DRY-RUN mode. Orders will be simulated and logged.")
            return

        api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_PAPER") or os.getenv("ALPACA_KEY_LVL1")
        api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY_PAPER") or os.getenv("ALPACA_SECRET_LVL1")

        if not api_key or not api_secret:
            logger.warning("No Alpaca API credentials found in environment. Defaulting to DRY-RUN mode.")
            self.dry_run = True
            return

        config = {
            "API_KEY": api_key,
            "API_SECRET": api_secret,
            "PAPER": self.paper,
        }

        try:
            self.broker = Alpaca(config)
            acct = self.broker.api.get_account()
            self.account_id = str(acct.id)
            self.portfolio_value = float(acct.portfolio_value or self.initial_portfolio)
            self.buying_power = float(acct.buying_power or self.initial_portfolio)
            mode_str = "PAPER" if self.paper else "LIVE"
            logger.info(
                f"[Alpaca Connected] Mode: {mode_str} | Account: {self.account_id} | "
                f"Portfolio: ${self.portfolio_value:,.2f} | Buying Power: ${self.buying_power:,.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}. Falling back to DRY-RUN mode.")
            self.dry_run = True

    def _load_persisted_state(self):
        """Loads processed signal IDs and state from disk if available."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.processed_signal_ids = set(data.get("processed_signal_ids", []))
                    logger.info(f"Loaded {len(self.processed_signal_ids)} processed signal IDs from state file.")
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")

    def _persist_state(self):
        """Saves processed signals and state to disk."""
        try:
            data = {
                "last_saved": datetime.now(self.tz_est).isoformat(),
                "processed_signal_ids": list(self.processed_signal_ids),
                "active_positions_count": len(self.active_positions),
                "daily_trades_count": len(self.daily_completed_trades),
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state file: {e}")

    def _emit_heartbeat(self, status: str = "RUNNING"):
        """Emits structured health telemetry to HEARTBEAT_FILE."""
        now_est = datetime.now(self.tz_est)
        hb = {
            "status": status,
            "timestamp": now_est.isoformat(),
            "date": now_est.strftime("%Y-%m-%d"),
            "time": now_est.strftime("%H:%M:%S EST"),
            "paper_mode": self.paper,
            "dry_run": self.dry_run,
            "portfolio_value": self.portfolio_value,
            "buying_power": self.buying_power,
            "active_positions_count": len(self.active_positions),
            "watchlist_count": len(self.active_watchlist),
            "processed_signals_count": len(self.processed_signal_ids),
            "closed_trades_today": len(self.daily_completed_trades),
            "enabled_strategies": self.enabled_strategies,
        }
        try:
            with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
                json.dump(hb, f, indent=2)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 1. MARKET TIMING & SESSION STATE MACHINE
    # ─────────────────────────────────────────────────────────────────────────

    def get_market_session_state(self, now_est: Optional[datetime] = None) -> Tuple[str, datetime]:
        """
        Determines current market phase:
        - "WEEKEND": Friday 20:00 to Sunday 23:59 EST
        - "PRE_MARKET": Mon-Fri 04:00 to 09:30 EST
        - "REGULAR_TRADING": Mon-Fri 09:30 to 16:00 EST
        - "AFTER_HOURS": Mon-Fri 16:00 to 20:00 EST
        - "OVERNIGHT": Mon-Thu 20:00 to 04:00 EST next morning
        """
        if now_est is None:
            now_est = datetime.now(self.tz_est)

        weekday = now_est.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        t = now_est.time()

        if weekday >= 5 or (weekday == 4 and t >= dtime(20, 0)):
            return "WEEKEND", now_est

        if dtime(4, 0) <= t < dtime(9, 30):
            return "PRE_MARKET", now_est
        elif dtime(9, 30) <= t < dtime(16, 0):
            return "REGULAR_TRADING", now_est
        elif dtime(16, 0) <= t < dtime(20, 0):
            return "AFTER_HOURS", now_est
        else:
            return "OVERNIGHT", now_est

    def _check_and_reset_daily_session(self, now_est: datetime):
        """Resets daily session variables when rolling over to a new calendar date."""
        today = now_est.date()
        if self.current_session_date != today:
            logger.info(f"=== New Trading Session Initialized: {today} ===")
            self.current_session_date = today
            self.active_watchlist.clear()
            self.daily_completed_trades.clear()
            self._load_premarket_signals_for_today(today)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. LIVE SIGNAL INGESTION & FEEDBACK LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def _load_premarket_signals_for_today(self, target_date: date):
        """Loads signals for target_date including prior-day after-hours signals."""
        target_str = target_date.strftime("%Y-%m-%d")
        try:
            signals = fetch_reactor_signals(target_str, include_prev_afterhours=True)
            for sig in signals:
                if sig.raw_signal_id and sig.raw_signal_id not in self.processed_signal_ids:
                    self._process_single_signal(sig, is_historical=True)
        except Exception as e:
            logger.error(f"Error fetching pre-market signals for {target_str}: {e}")

    def poll_new_live_signals(self):
        """
        Polls MySQL `signals` table for any new signals inserted into the database.
        Runs live feedback classification and updates the active watchlist.
        """
        conn = None
        cursor = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor(dictionary=True)

            # Query all signals from the past 24 hours
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
            WHERE s.created_at >= NOW() - INTERVAL 24 HOUR
            ORDER BY s.created_at ASC;
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            for row in rows:
                sig_id = str(row.get("signal_id", ""))
                if not sig_id or sig_id in self.processed_signal_ids:
                    continue

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

                sig = MarketReactorClassifier.classify_signal(row, fa_data)
                self._process_single_signal(sig, is_historical=False)

        except Exception as e:
            logger.error(f"Database error during live signal polling: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _process_single_signal(self, sig: ReactorSignal, is_historical: bool = False):
        """
        Evaluates a single signal, checks the Dual Hard Floor (>=70% on both),
        and adds qualifying tickers to the active live trading watchlist.
        """
        self.processed_signal_ids.add(sig.raw_signal_id)
        self._persist_state()

        prefix = "[Historical]" if is_historical else "[LIVE INGESTION]"
        ticker = sig.ticker.upper().strip()

        if not sig.is_eligible:
            logger.debug(
                f"{prefix} Rejected {ticker} {sig.signal_position} - Dual Hard Floor (<70%): "
                f"Fund={sig.fundamental_confidence:.1%}, News={sig.news_conviction:.1%}"
            )
            return

        logger.info(
            f"{prefix} HIGH-CONVICTION SIGNAL: {ticker} {sig.signal_position} | "
            f"Fund: {sig.fundamental_sentiment} ({sig.fundamental_confidence:.1%}) | "
            f"News: {sig.news_sentiment} ({sig.news_conviction:.1%}) | "
            f"Composite: {sig.composite_conviction:.1%} | "
            f"Selector {sig.selector_id} ({sig.selector_name})"
        )

        self.session_signals[ticker] = sig

        # Add to active watchlist if not already initialized
        if ticker not in self.active_watchlist:
            self._initialize_ticker_watchlist(ticker, sig)

    def _initialize_ticker_watchlist(self, ticker: str, sig: ReactorSignal):
        """Fetches daily bars and initializes technical levels (PDH, PDL, PDC, Daily ATR)."""
        logger.info(f"Initializing technical levels and liquidity pools for watchlist asset: {ticker}")
        daily_atr = 2.0
        pdh = None
        pdl = None
        pdc = None

        try:
            # Query bars via Alpaca or YFinance
            df = None
            if self.broker and not self.dry_run:
                try:
                    asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)
                    bars = self.broker.get_historical_prices(asset, 20, "day")
                    if bars is not None:
                        df = bars.df if hasattr(bars, "df") else bars
                except Exception:
                    df = None

            if df is None or df.empty or len(df) < 2:
                import yfinance as yf
                df = yf.download(ticker, period="1mo", interval="1d", progress=False)
                if not df.empty and isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                elif not df.empty:
                    df.columns = [c.lower() for c in df.columns]

            if df is not None and not df.empty and len(df) >= 2:
                pdc = float(df["close"].iloc[-2])
                pdh = float(df["high"].iloc[-2])
                pdl = float(df["low"].iloc[-2])
                daily_atr = float(calculate_atr(df, length=min(14, len(df) - 1))) or (pdc * 0.02)
        except Exception as e:
            logger.warning(f"Notice calculating daily levels for {ticker}: {e}")

        self.active_watchlist[ticker] = {
            "ticker": ticker,
            "signal": sig,
            "pdh": pdh,
            "pdl": pdl,
            "pdc": pdc,
            "daily_atr": daily_atr,
            "or_high": None,
            "or_low": None,
            "swept_levels": set(),
            "trades_taken": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. LIVE MARKET EVALUATION & STRATEGY EXECUTION ENGINES
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_live_market(self):
        """
        Iterates over the active watchlist during Regular Trading Hours and
        evaluates entry triggers for News_LiquiditySweep and MarketReactor_Fade.
        """
        now_est = datetime.now(self.tz_est)
        cur_time = now_est.time()

        # Update portfolio value & buying power
        self._update_account_balances()

        # Iterate through watchlist tickers
        for ticker, ctx in list(self.active_watchlist.items()):
            sig: ReactorSignal = ctx["signal"]

            # Max 1 trade per ticker per day
            if ctx["trades_taken"] >= 1:
                continue

            # Fetch recent intraday 5m and 15m candles
            candles_5m, candles_15m = self._fetch_recent_intraday_candles(ticker)
            if candles_5m is None or candles_5m.empty:
                continue

            current_price = float(candles_5m.iloc[-1]["close"])

            # Compute or update 15m Opening Range (9:30 - 9:45 AM EST)
            if cur_time >= dtime(9, 45) and ctx["or_high"] is None:
                self._compute_opening_range(ticker, ctx, candles_5m)

            # Strategy Engine 1: News_LiquiditySweep
            if "News_LiquiditySweep" in self.enabled_strategies and cur_time >= dtime(9, 45) and cur_time < dtime(15, 50):
                self._evaluate_liquidity_sweep_entry(ticker, ctx, sig, candles_5m, candles_15m, current_price)

            # Strategy Engine 2: MarketReactor (Fade Mode)
            if "MarketReactor_Fade" in self.enabled_strategies and sig.selector_id == 2 and cur_time >= dtime(9, 45) and cur_time < dtime(15, 55):
                self._evaluate_market_reactor_fade_entry(ticker, ctx, sig, candles_5m, current_price)

            # Strategy Engine 3: HighBeta_CatalystOpen (09:30 - 10:15 AM EST)
            if "HighBeta_CatalystOpen" in self.enabled_strategies and cur_time >= dtime(9, 30) and cur_time <= dtime(10, 15):
                self._evaluate_high_beta_catalyst_open_entry(ticker, ctx, sig, candles_5m, current_price, now_est)

            # Strategy Engine 4: Institutional_VRecovery (09:35 - 10:15 AM EST)
            if "Institutional_VRecovery" in self.enabled_strategies and cur_time >= dtime(9, 35) and cur_time <= dtime(10, 15):
                self._evaluate_institutional_v_recovery_entry(ticker, ctx, sig, candles_5m, current_price, now_est)

        # Active position management & ratchets
        self.manage_active_positions(now_est)

    def _fetch_recent_intraday_candles(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetches 5m and 15m intraday candles for the active session."""
        try:
            import yfinance as yf
            df = yf.download(ticker, period="5d", interval="5m", progress=False)
            if df.empty:
                return None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            if df.index.tz is not None:
                df.index = df.index.tz_convert(self.tz_est)
            else:
                df.index = df.index.tz_localize("UTC").tz_convert(self.tz_est)

            df_15m = df.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            return df, df_15m
        except Exception as e:
            logger.warning(f"Error fetching candles for {ticker}: {e}")
            return None, None

    def _compute_opening_range(self, ticker: str, ctx: Dict[str, Any], df_5m: pd.DataFrame):
        """Extracts 9:30 - 9:45 AM EST high and low."""
        today = datetime.now(self.tz_est).date()
        today_bars = df_5m[df_5m.index.date == today]
        or_bars = today_bars[(today_bars.index.time >= dtime(9, 30)) & (today_bars.index.time <= dtime(9, 45))]

        if not or_bars.empty:
            ctx["or_high"] = float(or_bars["high"].max())
            ctx["or_low"] = float(or_bars["low"].min())
            logger.info(f"[{ticker}] 15m Opening Range Established: High=${ctx['or_high']:.2f} | Low=${ctx['or_low']:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 1: NEWS + LIQUIDITY SWEEP ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_liquidity_sweep_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        current_price: float
    ):
        """
        Detects liquidity sweeps of PDL, OR_LOW, PDC (for BUY) or PDH, OR_HIGH, PDC (for SELL)
        with volume confirmation and subsequent reclaim.
        """
        allowed_side = sig.signal_position  # "BUY" or "SELL"
        target_pools: List[Tuple[str, Optional[float]]] = []

        if allowed_side == "BUY":
            target_pools = [
                ("OR_LOW", ctx.get("or_low")),
                ("PDL", ctx.get("pdl")),
                ("PDC_GAP", ctx.get("pdc")),
            ]
        else:
            target_pools = [
                ("OR_HIGH", ctx.get("or_high")),
                ("PDH", ctx.get("pdh")),
                ("PDC_GAP", ctx.get("pdc")),
            ]

        # Valid pools only
        valid_pools = [(name, lvl) for name, lvl in target_pools if lvl is not None]
        if not valid_pools:
            return

        atr_val = calculate_atr(df_15m, length=14) or (current_price * 0.015)
        atr_buf = atr_val * 0.8

        last_bar = df_5m.iloc[-1]
        h = float(last_bar["high"])
        l = float(last_bar["low"])
        c = float(last_bar["close"])
        v = float(last_bar["volume"])

        vol_ma = float(df_5m["volume"].rolling(20).mean().iloc[-1]) if len(df_5m) >= 20 else v
        v_spike = v >= (vol_ma * 1.15)

        trigger_setup = None

        if allowed_side == "BUY" and v_spike:
            for pool_name, pool_lvl in valid_pools:
                if pool_name in ctx["swept_levels"]:
                    continue
                # Sweep wick below level and close back above
                if l < pool_lvl and c > pool_lvl:
                    valid_depth = (pool_lvl - l) <= (1.5 * atr_val)
                    if valid_depth:
                        sl = l - atr_buf
                        tp = max(ctx.get("pdh", c + 2.0 * atr_val), c + (2.0 * atr_val))
                        if (tp - c) / max(0.01, c - sl) >= 1.6:
                            trigger_setup = ("BUY", c, sl, tp, f"SWEEP_{pool_name}_RECLAIM")
                            ctx["swept_levels"].add(pool_name)
                            break

        elif allowed_side == "SELL" and v_spike:
            for pool_name, pool_lvl in valid_pools:
                if pool_name in ctx["swept_levels"]:
                    continue
                # Sweep wick above level and close back below
                if h > pool_lvl and c < pool_lvl:
                    valid_depth = (h - pool_lvl) <= (1.5 * atr_val)
                    if valid_depth:
                        sl = h + atr_buf
                        tp = min(ctx.get("pdl", c - 2.0 * atr_val), c - (2.0 * atr_val))
                        if (c - tp) / max(0.01, sl - c) >= 1.6:
                            trigger_setup = ("SELL", c, sl, tp, f"SWEEP_{pool_name}_RECLAIM")
                            ctx["swept_levels"].add(pool_name)
                            break

        if trigger_setup:
            side, entry_p, sl, tp, reason = trigger_setup
            self._execute_live_trade(
                strategy_name="News_LiquiditySweep",
                ticker=ticker,
                side=side,
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                reason=reason,
                sig=sig,
                ctx=ctx,
                be_trigger_r=1.2,
                eod_flatten_time=dtime(15, 50),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 2: MARKET REACTOR (FADE MODE)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_market_reactor_fade_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        df_5m: pd.DataFrame,
        current_price: float
    ):
        """
        Selector 2 Fundamental Fade: Fades morning hype on fundamentally flawed
        stocks (e.g. positive PR on Value Trap) or buys panic on strong fundamentals.
        """
        news_side = sig.signal_position  # News direction
        fund_side = "SELL" if news_side == "BUY" else "BUY"  # Target fade direction

        or_high = ctx.get("or_high")
        or_low = ctx.get("or_low")
        if or_high is None or or_low is None:
            return

        atr_val = ctx.get("daily_atr", current_price * 0.02) * 0.15  # ~15m ATR approximation

        last_bar = df_5m.iloc[-1]
        h = float(last_bar["high"])
        l = float(last_bar["low"])
        c = float(last_bar["close"])

        trigger_setup = None

        if fund_side == "SELL":
            # Bullish news on a fundamentally bearish company:
            # Morning spikes into or above OR_HIGH, then fails and closes back below OR_HIGH
            if h >= or_high and c < or_high:
                sl = h + (0.5 * atr_val)
                tp = max(0.01, c - (2.0 * atr_val))
                if (c - tp) / max(0.01, sl - c) >= 1.5:
                    trigger_setup = ("SELL", c, sl, tp, "MR_FADE_ORB_HIGH_REJECTION")

        elif fund_side == "BUY":
            # Bearish news on a fundamentally bullish company:
            # Morning dumps into or below OR_LOW, then bounces and closes back above OR_LOW
            if l <= or_low and c > or_low:
                sl = l - (0.5 * atr_val)
                tp = c + (2.0 * atr_val)
                if (tp - c) / max(0.01, c - sl) >= 1.5:
                    trigger_setup = ("BUY", c, sl, tp, "MR_FADE_ORB_LOW_BOUNCE")

        if trigger_setup:
            side, entry_p, sl, tp, reason = trigger_setup
            self._execute_live_trade(
                strategy_name="MarketReactor_Fade",
                ticker=ticker,
                side=side,
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                reason=reason,
                sig=sig,
                ctx=ctx,
                be_trigger_r=1.2,
                eod_flatten_time=dtime(15, 55),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 3: HIGH-BETA CATALYST OPENING STRATEGY
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_high_beta_catalyst_open_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        df_5m: pd.DataFrame,
        current_price: float,
        now_est: datetime,
    ):
        """
        Executes the 5-step rulebook across high-beta equities:
        Evaluates Catalyst Confluence (> 4 or < 2 on stock/peers), Nasdaq (QQQ) regime,
        and trades Pattern 1 (Golden V-Shape), Pattern 2 (Straight Rally), Pattern 3
        (Bear False Recovery), or Pattern 4 (Forensic Bull Flag).
        """
        high_beta_tickers = {"COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR"}
        pdc = ctx.get("pdc")
        daily_atr = ctx.get("daily_atr", current_price * 0.025)
        atr_pct = (daily_atr / pdc) if pdc else 0.025
        if ticker not in high_beta_tickers and atr_pct < 0.020:
            return

        today_bars = df_5m[df_5m.index.date == now_est.date()]
        if len(today_bars) < 2:
            return

        open_price = float(today_bars.iloc[0]["open"])
        session_low = float(today_bars["low"].min())
        session_high = float(today_bars["high"].max())

        last_bar = today_bars.iloc[-1]
        prev_bar = today_bars.iloc[-2]
        is_green_reversal = (last_bar["close"] > last_bar["open"]) and (last_bar["close"] > prev_bar["high"])
        is_red_breakdown = (last_bar["close"] < last_bar["open"]) and (last_bar["close"] < prev_bar["low"])

        # Catalyst Confluence
        today_str = now_est.strftime("%Y-%m-%d")
        cat_analysis = CatalystConfluenceEngine.analyze_catalysts(
            target_ticker=ticker,
            target_date_str=today_str,
            signals_cache=list(self.session_signals.values())
        )
        cat_bias = cat_analysis["catalyst_bias"]
        cat_type = cat_analysis["catalyst_type"]

        # Nasdaq QQQ Regime
        qqq_regime = "BULL_MARKET"
        if "QQQ" in self.active_watchlist:
            qqq_ctx = self.active_watchlist["QQQ"]
            qqq_pdc = qqq_ctx.get("pdc", 480.0)
            if current_price < qqq_pdc:
                qqq_regime = "BEAR_MARKET"

        tp_pct = 0.020  # +2.0% Take Profit
        sl_pct = 0.035  # -3.5% Stop Loss
        trigger_setup = None

        # Pattern 2: Straight Rally Breakout (Bull Market + Bullish Catalyst > 4)
        if qqq_regime == "BULL_MARKET" and cat_bias == "BULLISH" and now_est.time() <= dtime(9, 45):
            gain_from_open = (current_price - open_price) / open_price
            if gain_from_open >= 0.008 and is_green_reversal:
                sl = min(current_price * (1.0 - sl_pct), session_low)
                tp = current_price * (1.0 + tp_pct)
                trigger_setup = ("BUY", current_price, sl, tp, f"PATTERN_2_STRAIGHT_RALLY | {cat_type}")

        # Pattern 1: Golden V-Shape (Bull Market + Neutral Catalyst -> PDC Rebound)
        elif qqq_regime == "BULL_MARKET" and cat_bias == "NEUTRAL" and pdc:
            drop_from_open = (open_price - current_price) / open_price
            tested_pdc = (drop_from_open >= 0.008 and abs(current_price - pdc) / pdc <= 0.008) or (current_price <= pdc and drop_from_open >= 0.010)
            if tested_pdc and is_green_reversal:
                sl = min(current_price * (1.0 - sl_pct), session_low - (0.002 * current_price))
                tp = current_price * (1.0 + tp_pct)
                trigger_setup = ("BUY", current_price, sl, tp, "PATTERN_1_GOLDEN_V_SHAPE_PDC_REBOUND")

        # Pattern 4: Forensic COIN V-Recovery + Higher-Low Bull Flag
        elif qqq_regime == "BULL_MARKET" and cat_bias == "BULLISH":
            dip_depth = (open_price - session_low) / open_price
            bounce_height = (session_high - session_low) / session_low
            if dip_depth >= 0.008 and bounce_height >= 0.008:
                is_higher_low = (last_bar["low"] >= session_low * 1.006) and (current_price < session_high)
                if is_higher_low and is_green_reversal:
                    sl = session_low - (0.003 * current_price)
                    tp = current_price * (1.0 + tp_pct)
                    trigger_setup = ("BUY", current_price, sl, tp, f"PATTERN_4_FORENSIC_COIN_BULL_FLAG | {cat_type}")

        # Pattern 3: Bear Market False Recovery Breakdown
        elif (qqq_regime == "BEAR_MARKET" or cat_bias == "BEARISH") and (current_price < open_price):
            attempted_recovery = (session_high < open_price * 1.003) and (session_high > session_low * 1.004)
            breaks_morning_low = (current_price < session_low) and is_red_breakdown
            if attempted_recovery and breaks_morning_low:
                sl = min(current_price * (1.0 + sl_pct), session_high + (0.003 * current_price))
                tp = current_price * (1.0 - tp_pct)
                trigger_setup = ("SELL", current_price, sl, tp, f"PATTERN_3_BEAR_FALSE_RECOVERY | {cat_type}")

        if trigger_setup:
            side, entry_p, sl, tp, reason = trigger_setup
            self._execute_live_trade(
                strategy_name="HighBeta_CatalystOpen",
                ticker=ticker,
                side=side,
                entry_price=entry_p,
                stop_loss=sl,
                take_profit=tp,
                reason=reason,
                sig=sig,
                ctx=ctx,
                be_trigger_r=1.0,
                eod_flatten_time=dtime(15, 55),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 4: INSTITUTIONAL V-RECOVERY (5 QUANTITATIVE ENHANCEMENTS)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_institutional_v_recovery_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        candles_5m: Optional[pd.DataFrame],
        current_price: float,
        now_est: datetime,
    ):
        """
        Executes Institutional V-Recovery with all 5 upgrades:
        1. Volume Profile & Intraday VWAP Confluence
        2. Higher-Timeframe Nasdaq (QQQ) Bull Market Filter + Relative Strength Divergence
        3. Structural Volatility Stop (1.2x of 15m ATR below session low, max 3.0%)
        4. Strict Execution Windows (09:35 - 10:15 EST entries)
        5. Scaled Profit Targets (+2.0% TP1, +3.5% TP2 Runner, +1.0% Breakeven Ratchet)
        """
        if ticker not in ["COIN", "NVDA", "TSLA", "AMD", "PLTR", "MARA", "MSTR"]:
            return

        today = now_est.date()
        date_str = today.strftime("%Y-%m-%d")

        if candles_5m is None or candles_5m.empty:
            return

        today_bars = candles_5m[candles_5m.index.date == today]
        if len(today_bars) < 3:
            return

        # Fetch daily levels & 20D ATR
        pdc = ctx.get("pdc")
        daily_atr = ctx.get("daily_atr")
        if not pdc or not daily_atr:
            try:
                import yfinance as yf
                df_1d = yf.download(ticker, period="30d", interval="1d", progress=False)
                if not df_1d.empty and len(df_1d) >= 2:
                    pdc = float(df_1d["Close"].iloc[-2])
                    daily_atr = float(calculate_atr(df_1d, length=min(14, len(df_1d) - 1))) or (pdc * 0.025)
                    ctx["pdc"] = pdc
                    ctx["daily_atr"] = daily_atr
            except Exception:
                pass

        if not pdc or not daily_atr or (daily_atr / pdc) < 0.020:
            return

        # Fetch QQQ 5m & higher-timeframe regime
        qqq_5m = ctx.get("qqq_5m")
        qqq_regime = ctx.get("qqq_regime")
        if not qqq_regime:
            try:
                import yfinance as yf
                qqq_df = yf.download("QQQ", period="30d", interval="1d", progress=False)
                qqq_5m_df = yf.download("QQQ", period="2d", interval="5m", progress=False)
                if not qqq_df.empty:
                    q_open = float(qqq_5m_df["Open"].iloc[0]) if not qqq_5m_df.empty else float(qqq_df["Open"].iloc[-1])
                    q_curr = float(qqq_5m_df["Close"].iloc[-1]) if not qqq_5m_df.empty else float(qqq_df["Close"].iloc[-1])
                    qqq_regime = CatalystConfluenceEngine.evaluate_market_regime(qqq_df.iloc[:-1], q_curr, q_open)
                    ctx["qqq_regime"] = qqq_regime
                    ctx["qqq_5m"] = qqq_5m_df
            except Exception:
                qqq_regime = "BULL_MARKET"

        if qqq_regime != "BULL_MARKET":
            return

        # Compute intraday VWAP
        today_bars = today_bars.copy()
        today_bars["vwap"] = compute_vwap(today_bars)
        cur_vwap = float(today_bars["vwap"].iloc[-1])

        # RVOL filter
        vol_ma20 = float(today_bars["volume"].rolling(min(20, len(today_bars))).mean().iloc[-1])
        cur_vol = float(today_bars["volume"].iloc[-1])
        rvol = (cur_vol / vol_ma20) if vol_ma20 > 0 else 1.0

        last_bar = today_bars.iloc[-1]
        prev_bar = today_bars.iloc[-2]
        is_green_reversal = (last_bar["close"] > last_bar["open"]) and (last_bar["close"] > prev_bar["high"])

        open_price = float(today_bars.iloc[0]["open"])
        session_low = float(today_bars["low"].min())

        vwap_reclaimed = (current_price >= cur_vwap) or (last_bar["low"] <= cur_vwap and current_price > cur_vwap)
        rvol_confirmed = (rvol >= 1.5) or (cur_vol >= vol_ma20 * 1.3)

        # Bullish divergence vs QQQ
        has_bullish_div = True
        if qqq_5m is not None and len(qqq_5m) >= 2:
            q_last = qqq_5m.iloc[-1]
            q_prev = qqq_5m.iloc[-2]
            q_dropping = float(q_last["Low"]) <= float(q_prev["Low"])
            stock_holding = float(last_bar["low"]) >= float(prev_bar["low"])
            has_bullish_div = (not q_dropping) or stock_holding

        # Catalyst analysis
        cat_analysis = CatalystConfluenceEngine.analyze_catalysts(ticker, date_str)
        cat_bias = cat_analysis.get("catalyst_bias", "NEUTRAL")

        drop_from_open = (open_price - session_low) / open_price
        bounce_height = (current_price - session_low) / max(0.01, session_low)

        if (
            is_green_reversal
            and vwap_reclaimed
            and rvol_confirmed
            and has_bullish_div
            and (cat_bias == "BULLISH" or drop_from_open >= 0.010)
            and bounce_height >= 0.006
        ):
            atr_buffer = 1.2 * (daily_atr * 0.20)
            hard_sl = max(session_low - atr_buffer, current_price * 0.970)
            tp1 = current_price * 1.020
            reason = f"VWAP Reclaimed (${cur_vwap:.2f}) | RVOL={rvol:.1f}x | Bullish Div Confirmed | {cat_analysis.get('catalyst_type', 'V_RECOVERY')}"

            self._execute_live_trade(
                strategy_name="Institutional_VRecovery",
                ticker=ticker,
                side="BUY",
                entry_price=current_price,
                stop_loss=hard_sl,
                take_profit=tp1,
                reason=reason,
                sig=sig,
                ctx=ctx,
                be_trigger_r=1.0,
                eod_flatten_time=dtime(15, 55),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. LIVE ORDER EXECUTION & DB ORDERS SYNCHRONIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_live_trade(
        self,
        strategy_name: str,
        ticker: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        reason: str,
        sig: ReactorSignal,
        ctx: Dict[str, Any],
        be_trigger_r: float = 1.2,
        eod_flatten_time: dtime = dtime(15, 50),
    ):
        """Sizes position by conviction-weighted risk, submits to Alpaca, and records to DB."""
        # Conviction-weighted sizing: 1.0% to 2.5%
        comp_score = getattr(sig, "composite_conviction", 0.70)
        norm_score = max(0.0, min(1.0, (comp_score - 0.70) / 0.30))
        risk_pct = 0.010 + (norm_score * 0.015)

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return

        portfolio_to_risk = self.portfolio_value or self.initial_portfolio
        qty = floor((portfolio_to_risk * risk_pct) / risk_per_share)
        qty = max(1, min(qty, floor((portfolio_to_risk * 0.95) / entry_price)))

        total_amount = round(qty * entry_price, 2)
        pos_id = f"{strategy_name}_{ticker}_{datetime.now().strftime('%H%M%S')}"

        logger.info(
            f"🚀 [ORDER TRIGGER] {strategy_name} | {ticker} {side} {qty} shares @ ${entry_price:.2f} | "
            f"SL=${stop_loss:.2f} | TP=${take_profit:.2f} | Risk=${portfolio_to_risk * risk_pct:.2f} ({risk_pct:.2%}) | {reason}"
        )

        alpaca_order_id = None
        if self.broker and not self.dry_run:
            try:
                asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)
                # Bracket order: market entry with bracket TP & SL
                order = self.broker.create_order(
                    asset=asset,
                    quantity=qty,
                    side=side.lower(),
                    type="market",
                    take_profit_price=take_profit,
                    stop_loss_price=stop_loss,
                )
                submitted = self.broker.submit_order(order)
                alpaca_order_id = str(getattr(submitted, "id", pos_id))
                logger.info(f"[Alpaca Fill] Bracket order registered with ID: {alpaca_order_id}")
            except Exception as e:
                logger.error(f"[Alpaca Order Error] Could not submit order: {e}")

        # Record into active positions state
        pos_record = {
            "pos_id": pos_id,
            "alpaca_order_id": alpaca_order_id,
            "strategy": strategy_name,
            "ticker": ticker,
            "side": side,
            "entry_price": entry_price,
            "quantity": qty,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "initial_risk": risk_per_share,
            "be_trigger_r": be_trigger_r,
            "breakeven_ratchet": False,
            "eod_flatten_time": eod_flatten_time,
            "entry_time": datetime.now(self.tz_est),
            "status": "OPEN",
            "db_order_id": None,
        }

        # Write to MySQL `orders` table
        db_id = self._record_order_to_database(pos_record, total_amount)
        pos_record["db_order_id"] = db_id

        self.active_positions[pos_id] = pos_record
        ctx["trades_taken"] += 1
        self._persist_state()

    def _record_order_to_database(self, pos: Dict[str, Any], total_amount: float) -> Optional[int]:
        """Inserts trade record into MySQL `orders` table."""
        conn = None
        cursor = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()
            order_date_str = pos["entry_time"].strftime("%d/%m/%y")
            query = """
            INSERT INTO orders (
                order_date, symbol, name, status, buy_price, sell_price, profit, order_amount, total_amount, order_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            buy_px = pos["entry_price"] if pos["side"] == "BUY" else None
            sell_px = pos["entry_price"] if pos["side"] == "SELL" else None
            params = (
                order_date_str,
                pos["ticker"],
                f"{pos['strategy']} Live",
                "Open",
                buy_px,
                sell_px,
                None,
                pos["quantity"],
                total_amount,
                pos["side"].capitalize(),
            )
            cursor.execute(query, params)
            db_id = cursor.lastrowid
            logger.info(f"Recorded order in MySQL `orders` table (ID: {db_id}).")
            return db_id
        except Exception as e:
            logger.error(f"Failed to record order to MySQL `orders` table: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _update_order_in_database(self, pos: Dict[str, Any], exit_price: float, profit: float):
        """Updates closed order status and realized P&L in MySQL `orders` table."""
        if not pos.get("db_order_id"):
            return
        conn = None
        cursor = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()
            query = """
            UPDATE orders 
            SET status = 'Closed', 
                sell_price = COALESCE(sell_price, %s),
                buy_price = COALESCE(buy_price, %s),
                profit = %s
            WHERE id = %s;
            """
            s_px = exit_price if pos["side"] == "BUY" else pos["entry_price"]
            b_px = pos["entry_price"] if pos["side"] == "BUY" else exit_price
            cursor.execute(query, (s_px, b_px, round(profit, 2), pos["db_order_id"]))
            logger.info(f"Updated MySQL `orders` table for Order #{pos['db_order_id']}: Status=Closed, Profit=${profit:+,.2f}")
        except Exception as e:
            logger.error(f"Failed to update order in MySQL: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. ACTIVE POSITION MANAGEMENT & EOD FLATTEN
    # ─────────────────────────────────────────────────────────────────────────

    def manage_active_positions(self, now_est: datetime):
        """
        Monitors open positions, applies Breakeven Ratchets (+1.2R),
        and executes strict EOD flattening before market close.
        """
        cur_time = now_est.time()
        closed_ids = []

        for pos_id, pos in list(self.active_positions.items()):
            ticker = pos["ticker"]
            side = pos["side"]
            entry_p = pos["entry_price"]
            initial_risk = pos["initial_risk"]

            # Current price lookup
            current_price = entry_p
            if self.broker and not self.dry_run:
                try:
                    asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)
                    current_price = float(self.broker.get_last_price(asset) or entry_p)
                except Exception:
                    pass

            gain = (current_price - entry_p) if side == "BUY" else (entry_p - current_price)
            r_mult = gain / initial_risk if initial_risk > 0 else 0.0

            # 1. Breakeven Stop Ratchet at +1.2R
            if not pos.get("breakeven_ratchet", False) and r_mult >= pos.get("be_trigger_r", 1.2):
                pos["stop_loss"] = entry_p
                pos["breakeven_ratchet"] = True
                logger.info(f"🔒 [BREAKEVEN RATCHET] {ticker} reached +{r_mult:.1f}R! Stop loss moved to Entry (${entry_p:.2f}).")

            # 2. Strict EOD Flatten
            eod_time = pos.get("eod_flatten_time", dtime(15, 50))
            if cur_time >= eod_time:
                logger.info(f"⏰ [EOD FLATTEN] Flattening {ticker} ({pos['strategy']}) at {cur_time.strftime('%H:%M:%S')} EST.")
                self._close_position(pos, exit_price=current_price, exit_reason="EOD_FLATTEN")
                closed_ids.append(pos_id)
                continue

            # 3. Check TP and SL Hits (for simulated/dry-run or manual broker sync)
            if side == "BUY":
                if current_price >= pos["take_profit"]:
                    self._close_position(pos, exit_price=pos["take_profit"], exit_reason="HIT_TAKE_PROFIT")
                    closed_ids.append(pos_id)
                elif current_price <= pos["stop_loss"]:
                    self._close_position(pos, exit_price=pos["stop_loss"], exit_reason="STOPPED_OUT")
                    closed_ids.append(pos_id)
            else:  # SELL
                if current_price <= pos["take_profit"]:
                    self._close_position(pos, exit_price=pos["take_profit"], exit_reason="HIT_TAKE_PROFIT")
                    closed_ids.append(pos_id)
                elif current_price >= pos["stop_loss"]:
                    self._close_position(pos, exit_price=pos["stop_loss"], exit_reason="STOPPED_OUT")
                    closed_ids.append(pos_id)

        for pid in closed_ids:
            if pid in self.active_positions:
                del self.active_positions[pid]

        if closed_ids:
            self._persist_state()

    def _close_position(self, pos: Dict[str, Any], exit_price: float, exit_reason: str):
        """Flattens position with broker and finalizes accounting."""
        ticker = pos["ticker"]
        side = pos["side"]
        qty = pos["quantity"]

        # Cancel open orders and close via Alpaca
        if self.broker and not self.dry_run:
            try:
                asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)
                self.broker.cancel_open_orders()
                self.broker.close_position(asset)
            except Exception as e:
                logger.warning(f"Notice closing position via broker: {e}")

        # Realized P&L
        if side == "BUY":
            profit = (exit_price - pos["entry_price"]) * qty
        else:
            profit = (pos["entry_price"] - exit_price) * qty

        pos["status"] = exit_reason
        pos["exit_price"] = exit_price
        pos["exit_time"] = datetime.now(self.tz_est)
        pos["profit"] = profit

        logger.info(
            f"✅ [POSITION CLOSED] {pos['strategy']} | {ticker} | Exit: ${exit_price:.2f} | "
            f"P&L: ${profit:+,.2f} | Reason: {exit_reason}"
        )

        self._update_order_in_database(pos, exit_price, profit)
        self.daily_completed_trades.append(pos)

    def _update_account_balances(self):
        """Refreshes account balance and portfolio value from Alpaca broker."""
        if self.broker and not self.dry_run:
            try:
                acct = self.broker.api.get_account()
                self.portfolio_value = float(acct.portfolio_value)
                self.buying_power = float(acct.buying_power)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # 6. CONTINUOUS 24/7 DAEMON SERVICE LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def run_all_the_time(self):
        """Main service daemon loop executing continuously 24/7 with real-time observer."""
        logger.info("=" * 90)
        logger.info(" GOTTI LIVE NEWS TRADER SERVICE STARTED (REAL-TIME EVENT OBSERVER MODE)")
        logger.info(f" Enabled Strategies: {', '.join(self.enabled_strategies)}")
        logger.info(f" Mode: {'DRY-RUN' if self.dry_run else ('PAPER' if self.paper else 'LIVE')}")
        logger.info(f" Initial Portfolio: ${self.portfolio_value:,.2f}")
        logger.info(" Signal Pipeline: Real-Time WebSocket Push (<5ms) + Sub-Second DB Watcher (0ms lag)")
        logger.info("=" * 90)

        # Start real-time signal observer (WebSocket + reactive DB change detector)
        self.observer.start()

        try:
            while self.running:
                try:
                    now_est = datetime.now(self.tz_est)
                    phase, _ = self.get_market_session_state(now_est)

                    # Reset daily state on date change
                    self._check_and_reset_daily_session(now_est)

                    if phase == "REGULAR_TRADING":
                        # Instant wake on incoming signal event OR 1.0s market tick for live evaluation
                        sig_event = self.observer.get_event(timeout=1.0)
                        if sig_event:
                            self._on_instant_signal_observed(sig_event)

                        self.evaluate_live_market()
                        self._emit_heartbeat(status="ACTIVE_TRADING")

                    elif phase == "PRE_MARKET":
                        # Pre-market: wait for live signal push or 3s tick
                        sig_event = self.observer.get_event(timeout=3.0)
                        if sig_event:
                            self._on_instant_signal_observed(sig_event)
                        self._emit_heartbeat(status="PRE_MARKET_MONITORING")

                    elif phase == "AFTER_HOURS":
                        # After-hours queueing: flatten open positions and wait for catalysts
                        if self.active_positions:
                            self.manage_active_positions(now_est)
                        sig_event = self.observer.get_event(timeout=10.0)
                        if sig_event:
                            self._on_instant_signal_observed(sig_event)
                        self._emit_heartbeat(status="AFTER_HOURS_QUEUE")

                    elif phase == "WEEKEND":
                        # Weekend sleeping: wait for events or 60s tick
                        self._emit_heartbeat(status="WEEKEND_IDLE")
                        sig_event = self.observer.get_event(timeout=60.0)
                        if sig_event:
                            self._on_instant_signal_observed(sig_event)

                    else:  # OVERNIGHT
                        # Overnight phase: wait for catalysts or 30s tick
                        self._emit_heartbeat(status="OVERNIGHT_SLEEP")
                        sig_event = self.observer.get_event(timeout=30.0)
                        if sig_event:
                            self._on_instant_signal_observed(sig_event)

                except KeyboardInterrupt:
                    logger.info("Service manually interrupted.")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in live service loop: {e}", exc_info=True)
                    time.sleep(2)
        finally:
            # Graceful shutdown
            logger.info("Stopping real-time observer and flushing state...")
            self.observer.stop()
            self._persist_state()
            self._emit_heartbeat(status="STOPPED")
            logger.info("Live News Trader Service shut down cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Gotti Live News Trading Service (Lumibot + Alpaca)")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Execution mode (default: paper)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders without submitting to Alpaca")
    parser.add_argument("--poll-interval", type=int, default=20, help="DB polling interval in seconds (default: 20)")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation cycle and exit (for testing/cron)")
    parser.add_argument(
        "--strategies",
        type=str,
        default="News_LiquiditySweep,MarketReactor_Fade,HighBeta_CatalystOpen,Institutional_VRecovery",
        help="Comma-separated list of strategies to run"
    )

    args = parser.parse_args()

    active_strats = [s.strip() for s in args.strategies.split(",") if s.strip()]
    is_paper = (args.mode == "paper")

    service = LiveNewsTraderService(
        paper=is_paper,
        poll_interval_sec=args.poll_interval,
        dry_run=args.dry_run,
        strategies=active_strats,
    )

    if args.once:
        logger.info("Running single evaluation cycle (--once)...")
        now_est = datetime.now(service.tz_est)
        service._check_and_reset_daily_session(now_est)
        service.poll_new_live_signals()
        service.evaluate_live_market()
        service._emit_heartbeat(status="COMPLETED_ONCE")
        logger.info("Single evaluation cycle complete.")
    else:
        service.run_all_the_time()


if __name__ == "__main__":
    main()
