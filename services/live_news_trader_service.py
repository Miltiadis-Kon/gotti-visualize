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
from dataclasses import asdict

import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

import html
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

from lumibot.brokers import Alpaca
from lumibot.entities import Asset, Order

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
    ZedStrategy,
    check_7day_consecutive_volatility,
    scan_premarket_7day_volatility_universe,
)
from strategies.institutional_v_recovery import (
    compute_vwap,
    compute_ema,
    ZedInstitutionalStrategy,
    ZedNoVwapStrategy,
)

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



class EtfAccountContext:
    """
    Represents an isolated Alpaca Paper/Live Account tied strictly to a specific ETF tier.
    Enforces strict risk sizing, portfolio isolation, and trade routing:
    - Level 3 (Diamond Hands ETF): ZedStrategy (V2) & News_LiquiditySweep ONLY (max 5% cap)
    - Level 2 (Steady Grind ETF): MarketOpen_Catalyst & MarketReactor_Fade ONLY (max 5% cap)
    - Level 1 (Boomer Haven ETF): Automated Daily 1% Index DCA (QQQ + SPY)
    """
    def __init__(
        self,
        level: int,
        name: str,
        sub_account_id: str,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        initial_portfolio: float = 1000000.0,
        max_trade_pct: float = 0.05,
    ):
        self.level = level
        self.name = name
        self.sub_account_id = sub_account_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.max_trade_pct = max_trade_pct
        self.account_id: Optional[str] = None
        self.portfolio_value: float = initial_portfolio
        self.buying_power: float = initial_portfolio
        self.cash: float = initial_portfolio
        self.trading_client: Optional[TradingClient] = None
        self.broker: Optional[Alpaca] = None
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        if api_key and api_secret:
            try:
                self.trading_client = TradingClient(api_key, api_secret, paper=paper)
                acct = self.trading_client.get_account()
                self.account_id = str(acct.id)
                self.portfolio_value = float(acct.portfolio_value or initial_portfolio)
                self.buying_power = float(acct.buying_power or initial_portfolio)
                self.cash = float(acct.cash or initial_portfolio)
                config = {"API_KEY": api_key, "API_SECRET": api_secret, "PAPER": paper}
                self.broker = Alpaca(config)
                logger.info(
                    f"[Alpaca ETF Connected] Level {self.level}: {self.name} | "
                    f"Account: {self.account_id} | Portfolio: ${self.portfolio_value:,.2f} | "
                    f"Cash: ${self.cash:,.2f} | BP: ${self.buying_power:,.2f}"
                )
            except Exception as e:
                logger.error(f"Failed to connect Alpaca for Level {self.level} ({self.name}): {e}")

class LiveNewsTraderService:
    """
    Continuous live trading service managing segregated multi-tier ETF accounts:
    1. Level 3 (Diamond Hands ETF): ZedStrategy (V2) & News_LiquiditySweep ONLY (5% cap).
    2. Level 2 (Steady Grind ETF): MarketOpen_Catalyst & MarketReactor_Fade ONLY (5% cap).
    3. Level 1 (Boomer Haven ETF): Daily Index DCA (1% QQQ + 1% SPY daily).
    """

    def __init__(
        self,
        paper: bool = True,
        poll_interval_sec: int = 20,
        dry_run: bool = False,
        strategies: Optional[List[str]] = None,
        initial_portfolio: float = 1000000.0,
    ):
        self.paper = paper
        self.poll_interval_sec = poll_interval_sec
        self.dry_run = dry_run
        self.enabled_strategies = strategies or [
            "News_LiquiditySweep",
            "MarketReactor_Fade",
            "ZedStrategy_Institutional",
            "MarketOpen_Catalyst",
            "Index_DCA",
        ]
        self.initial_portfolio = initial_portfolio

        self.tz_est = pytz.timezone("America/New_York")
        self.running = True

        # Multi-Tier ETF Account contexts
        self.etf_accounts: Dict[int, EtfAccountContext] = {}
        self.l1_dca_executed_date: Optional[date] = None
        self.last_subaccount_sync_time: float = 0.0

        # Legacy pointers mapped to L3 for backward compatibility
        self.broker: Optional[Alpaca] = None
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self.account_id: Optional[str] = None
        self.portfolio_value: float = initial_portfolio
        self.buying_power: float = initial_portfolio

        # Telegram reporting (@gotti_trades_bot)
        self.telegram_token = os.getenv("TELEGRAM_TRADES_BOT_TOKEN", "9051838449:AAF6qwEyBlMeVqA6Yy-fg6gDOoD8JZJDIHQ")
        self.telegram_chat_id = os.getenv("TELEGRAM_TRADES_CHAT_ID", os.getenv("TELEGRAM_SIGNALS_CHAT_ID", "8139983484"))

        # State tracking & Premarket Queue
        self.processed_signal_ids: Set[str] = set()
        self.premarket_catalyst_queue: Dict[str, ReactorSignal] = {}  # ticker -> premarket signal for open execution
        self.session_signals: Dict[str, ReactorSignal] = {}  # ticker -> latest signal
        self.active_watchlist: Dict[str, Dict[str, Any]] = {}  # ticker -> technical levels & context
        self.active_positions: Dict[str, Dict[str, Any]] = {}  # pos_id -> position details
        self.daily_completed_trades: List[Dict[str, Any]] = []
        self.qualified_7day_volatility_tickers: Set[str] = set()

        self.current_session_date: Optional[date] = None

        self._setup_signal_handlers()
        self._load_persisted_state()
        self._init_broker()

        # Real-time event-driven signal observer (<5ms WebSocket + sub-second DB change detection)
        self.observer = RealTimeSignalObserver(
            on_signal_callback=None,
            db_poll_interval_sec=1.0,
        )

    def _on_instant_signal_observed(self, sig: ReactorSignal):
        """Processes instant live signal (< 5ms via WS or < 1s via DB) sequentially."""
        logger.info(f"[OBSERVER EVENT] Processing instant live signal: {sig.ticker} {sig.signal_position}")
        self._process_single_signal(sig, is_historical=False)

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
        """Initializes isolated Alpaca connections for all three ETF account tiers."""
        if self.dry_run:
            logger.info("Service started in DRY-RUN mode. Orders will be simulated and logged.")
            return

        # 1. Level 1: Boomer Haven ETF (Account ID: f18df431-7573-4d6a-a5f7-362ee95c81c4)
        k1 = os.getenv("ALPACA_KEY_LVL1")
        s1 = os.getenv("ALPACA_SECRET_LVL1")
        ctx_l1 = EtfAccountContext(
            level=1,
            name="Level 1: Boomer Haven ETF",
            sub_account_id="sub-defensive-01",
            api_key=k1 or "",
            api_secret=s1 or "",
            paper=self.paper,
            initial_portfolio=1000000.0,
            max_trade_pct=0.01,
        )
        self.etf_accounts[1] = ctx_l1

        # 2. Level 2: Steady Grind ETF (Account ID: bc4ca0cd-ff0e-42f1-a550-16e1000d87fe)
        k2 = os.getenv("ALPACA_KEY_LVL2")
        s2 = os.getenv("ALPACA_SECRET_LVL2")
        ctx_l2 = EtfAccountContext(
            level=2,
            name="Level 2: Steady Grind ETF",
            sub_account_id="sub-main-02",
            api_key=k2 or "",
            api_secret=s2 or "",
            paper=self.paper,
            initial_portfolio=101251.95,
            max_trade_pct=0.05,
        )
        self.etf_accounts[2] = ctx_l2

        # 3. Level 3: Diamond Hands ETF (Account ID: 4452438e-eb6c-4f44-b7ca-34da00aa780d)
        k3 = os.getenv("ALPACA_KEY_LVL3") or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_PAPER")
        s3 = os.getenv("ALPACA_SECRET_LVL3") or os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY_PAPER")
        ctx_l3 = EtfAccountContext(
            level=3,
            name="Level 3: Diamond Hands ETF",
            sub_account_id="sub-alpha-03",
            api_key=k3 or "",
            api_secret=s3 or "",
            paper=self.paper,
            initial_portfolio=1000000.0,
            max_trade_pct=0.05,
        )
        self.etf_accounts[3] = ctx_l3

        # Default legacy pointers mapped to Level 3 for compatibility
        self.trading_client = ctx_l3.trading_client
        self.broker = ctx_l3.broker
        self.account_id = ctx_l3.account_id
        self.portfolio_value = ctx_l3.portfolio_value
        self.buying_power = ctx_l3.buying_power

        # Shared market data client (SIP feed quotes/bars)
        data_key = k3 or k1 or k2
        data_secret = s3 or s1 or s2
        if data_key and data_secret:
            try:
                self.data_client = StockHistoricalDataClient(data_key, data_secret)
                logger.info("[Alpaca Market Data] StockHistoricalDataClient initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca Market Data client: {e}")

        self._sync_active_positions_from_broker()
        self._sync_sub_accounts_database()

    def _notify_telegram(self, text: str):
        """Sends rich formatted alert to the dedicated Telegram Trades Bot (@gotti_trades_bot)."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.debug(f"Telegram notification error: {e}")

    def _load_persisted_state(self):
        """Loads processed signal IDs, premarket queue, and active positions from disk if available."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.processed_signal_ids = set(data.get("processed_signal_ids", []))
                    if data.get("l1_dca_executed_date"):
                        try:
                            self.l1_dca_executed_date = date.fromisoformat(data["l1_dca_executed_date"])
                        except Exception:
                            pass

                    raw_positions = data.get("active_positions", {})
                    for pid, pos in raw_positions.items():
                        if "entry_time" in pos and isinstance(pos["entry_time"], str):
                            pos["entry_time"] = datetime.fromisoformat(pos["entry_time"])
                        if "eod_flatten_time" in pos and isinstance(pos["eod_flatten_time"], str):
                            parts = [int(x) for x in pos["eod_flatten_time"].split(":")]
                            pos["eod_flatten_time"] = dtime(*parts)
                        self.active_positions[pid] = pos

                    raw_premarket = data.get("premarket_catalyst_queue", {})
                    for tk, sig_dict in raw_premarket.items():
                        try:
                            self.premarket_catalyst_queue[tk] = ReactorSignal(**sig_dict)
                        except Exception:
                            pass

                    logger.info(
                        f"Loaded state: {len(self.processed_signal_ids)} signals, "
                        f"{len(self.active_positions)} active positions, "
                        f"{len(self.premarket_catalyst_queue)} queued premarket catalysts."
                    )
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")

    def _persist_state(self):
        """Saves processed signal IDs, premarket queue, and active positions to disk."""
        try:
            serializable_positions = {}
            for pid, p in self.active_positions.items():
                cp = p.copy()
                if isinstance(cp.get("entry_time"), (datetime, date)):
                    cp["entry_time"] = cp["entry_time"].isoformat()
                if isinstance(cp.get("eod_flatten_time"), dtime):
                    cp["eod_flatten_time"] = cp["eod_flatten_time"].strftime("%H:%M:%S")
                serializable_positions[pid] = cp

            serializable_premarket = {}
            for tk, sig in self.premarket_catalyst_queue.items():
                if hasattr(sig, "__dict__"):
                    serializable_premarket[tk] = {
                        k: v for k, v in sig.__dict__.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }

            data = {
                "last_saved": datetime.now(self.tz_est).isoformat(),
                "l1_dca_executed_date": self.l1_dca_executed_date.isoformat() if self.l1_dca_executed_date else None,
                "processed_signal_ids": list(self.processed_signal_ids),
                "active_positions": serializable_positions,
                "premarket_catalyst_queue": serializable_premarket,
                "active_positions_count": len(self.active_positions),
                "daily_trades_count": len(self.daily_completed_trades),
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save state file: {e}")

    def _sync_active_positions_from_broker(self):
        """
        Synchronizes open positions on each Alpaca ETF account with in-memory state and MySQL on startup.
        Ensures restart-safety and continuous EOD flatten & stop management across all 3 tiers.
        """
        if self.dry_run:
            return

        for lvl, etf_ctx in self.etf_accounts.items():
            if not etf_ctx.trading_client:
                continue

            try:
                alpaca_positions = etf_ctx.trading_client.get_all_positions()
                if not alpaca_positions:
                    logger.info(f"[Broker Sync L{lvl}] No active open positions on Alpaca {etf_ctx.name}.")
                    continue

                logger.info(f"[Broker Sync L{lvl}] Reconciling {len(alpaca_positions)} open positions from {etf_ctx.name}...")

                open_orders_map = {}
                try:
                    conn = get_mysql_connection()
                    cur = conn.cursor(dictionary=True)
                    cur.execute("SELECT * FROM orders WHERE status = 'Open' AND etf_level = %s ORDER BY id DESC", (lvl,))
                    for r in cur.fetchall():
                        sym = r["symbol"].upper()
                        if sym not in open_orders_map:
                            open_orders_map[sym] = r
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"[Broker Sync L{lvl}] Could not query MySQL open orders: {e}")

                for p in alpaca_positions:
                    ticker = p.symbol
                    qty = abs(int(float(p.qty)))
                    side = "BUY" if float(p.qty) > 0 else "SELL"
                    entry_p = float(p.avg_entry_price)

                    db_record = open_orders_map.get(ticker)
                    db_id = db_record["id"] if db_record else None
                    strat_name = (
                        db_record["name"].replace(f"[L{lvl}] ", "").replace(" Live", "")
                        if db_record and db_record.get("name")
                        else ("MarketOpen_Catalyst" if lvl == 2 else ("Index_DCA" if lvl == 1 else "ZedStrategy"))
                    )

                    if side == "BUY":
                        sl = round(entry_p * 0.970, 2)
                        tp = round(entry_p * 1.035, 2)
                    else:
                        sl = round(entry_p * 1.030, 2)
                        tp = round(entry_p * 0.965, 2)

                    pos_id = f"L{lvl}_{strat_name}_{ticker}_SYNC"
                    if not any(pos.get("ticker") == ticker and pos.get("etf_level") == lvl for pos in self.active_positions.values()):
                        record = {
                            "pos_id": pos_id,
                            "alpaca_order_id": None,
                            "strategy": strat_name,
                            "ticker": ticker,
                            "side": side,
                            "entry_price": entry_p,
                            "quantity": qty,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "initial_risk": abs(entry_p - sl),
                            "be_trigger_r": 1.2,
                            "breakeven_ratchet": False,
                            "eod_flatten_time": dtime(15, 55),
                            "entry_time": datetime.now(self.tz_est),
                            "status": "OPEN",
                            "etf_level": lvl,
                            "db_order_id": db_id,
                        }
                        self.active_positions[pos_id] = record
                        etf_ctx.active_positions[pos_id] = record
                        logger.info(f"[Broker Sync L{lvl}] Reconciled active position: {ticker} ({side} {qty} @ ${entry_p:.2f})")
            except Exception as e:
                logger.error(f"[Broker Sync L{lvl}] Error syncing positions: {e}")

    def _sync_sub_accounts_database(self):
        """Syncs Alpaca broker balances, cash, and PnL to MySQL `sub_accounts` table."""
        conn = None
        cursor = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()
            for lvl, acct_ctx in self.etf_accounts.items():
                if not acct_ctx.trading_client:
                    continue
                try:
                    acct = acct_ctx.trading_client.get_account()
                    pv = float(acct.portfolio_value or acct_ctx.portfolio_value)
                    cash = float(acct.cash or acct_ctx.cash)
                    acct_ctx.portfolio_value = pv
                    acct_ctx.cash = cash
                    acct_ctx.buying_power = float(acct.buying_power or acct_ctx.buying_power)

                    cursor.execute(
                        "SELECT allocated_capital FROM sub_accounts WHERE id = %s",
                        (acct_ctx.sub_account_id,)
                    )
                    row = cursor.fetchone()
                    alloc_cap = float(row[0]) if (row and row[0]) else pv
                    if alloc_cap <= 0:
                        alloc_cap = pv

                    invested = max(0.0, round(pv - cash, 2))
                    pnl = round(pv - alloc_cap, 2)
                    pnl_pct = round((pnl / alloc_cap) * 100, 4) if alloc_cap > 0 else 0.0

                    cursor.execute("""
                        UPDATE sub_accounts
                        SET current_value = %s,
                            cash_balance = %s,
                            invested_amount = %s,
                            pnl = %s,
                            pnl_percentage = %s,
                            updated_at = NOW()
                        WHERE id = %s;
                    """, (pv, cash, invested, pnl, pnl_pct, acct_ctx.sub_account_id))
                except Exception as ex:
                    logger.debug(f"Subaccount sync error for L{lvl}: {ex}")
            conn.commit()
        except Exception as e:
            logger.warning(f"Notice updating sub_accounts table: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

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
            "etf_tiers": {
                lvl: {
                    "name": ctx.name,
                    "account_id": ctx.account_id,
                    "portfolio_value": ctx.portfolio_value,
                    "cash": ctx.cash,
                    "buying_power": ctx.buying_power,
                    "positions_count": len(ctx.active_positions),
                }
                for lvl, ctx in self.etf_accounts.items()
            },
            "active_positions_count": len(self.active_positions),
            "watchlist_count": len(self.active_watchlist),
            "processed_signals_count": len(self.processed_signal_ids),
            "premarket_catalysts_count": len(self.premarket_catalyst_queue),
            "closed_trades_today": len(self.daily_completed_trades),
            "l1_dca_executed_date": self.l1_dca_executed_date.isoformat() if self.l1_dca_executed_date else None,
            "enabled_strategies": self.enabled_strategies,
        }
        try:
            with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
                json.dump(hb, f, indent=2)
        except Exception:
            pass

    def get_market_session_state(self, now_est: Optional[datetime] = None) -> Tuple[str, datetime]:
        """
        Determines market phase:
        - WEEKEND: Saturday or Sunday
        - PRE_MARKET: Mon-Fri 04:00 - 09:30 EST
        - REGULAR_TRADING: Mon-Fri 09:30 - 16:00 EST
        - AFTER_HOURS: Mon-Fri 16:00 - 20:00 EST
        - OVERNIGHT: Mon-Fri 20:00 - 04:00 EST
        """
        weekday = now_est.weekday()
        if weekday >= 5:
            monday_open = (now_est + timedelta(days=(7 - weekday))).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            return "WEEKEND", monday_open - now_est

        cur_time = now_est.time()
        if cur_time < dtime(4, 0):
            pre_start = now_est.replace(hour=4, minute=0, second=0, microsecond=0)
            return "OVERNIGHT", pre_start - now_est
        elif cur_time < dtime(9, 30):
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            return "PRE_MARKET", market_open - now_est
        elif cur_time <= dtime(16, 0):
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
            return "REGULAR_TRADING", market_close - now_est
        elif cur_time <= dtime(20, 0):
            overnight_start = now_est.replace(hour=20, minute=0, second=0, microsecond=0)
            return "AFTER_HOURS", overnight_start - now_est
        else:
            next_day_pre = (now_est + timedelta(days=1)).replace(
                hour=4, minute=0, second=0, microsecond=0
            )
            return "OVERNIGHT", next_day_pre - now_est

    def _check_and_reset_daily_session(self, now_est: datetime):
        """Resets intraday counters and loads pre-market catalysts at the start of a new trading day."""
        today = now_est.date()
        if self.current_session_date != today:
            logger.info(f"=== New Trading Session Initialized: {today} ===")
            self.current_session_date = today
            self.active_watchlist.clear()
            self.daily_completed_trades.clear()
            self.qualified_7day_volatility_tickers.clear()
            self._load_premarket_signals_for_today(today)
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

        # Track and queue premarket signals for Market Open (09:30 AM EST) execution
        now_est = datetime.now(self.tz_est)
        phase, _ = self.get_market_session_state(now_est)
        if phase in ("PRE_MARKET", "OVERNIGHT", "WEEKEND", "AFTER_HOURS") or now_est.time() < dtime(9, 30):
            self.premarket_catalyst_queue[ticker] = sig
            logger.info(
                f"📋 [PRE-MARKET QUEUE] Successfully queued {ticker} ({sig.signal_position}) for Market Open (09:30 AM EST) execution!"
            )

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
        is_qualified_zed = False

        try:
            # Query bars directly via Alpaca Market Data
            df = None
            if self.data_client:
                try:
                    from datetime import timezone
                    start_utc = datetime.now(timezone.utc) - timedelta(days=45)
                    req = StockBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=TimeFrame.Day,
                        start=start_utc,
                    )
                    bars = self.data_client.get_stock_bars(req)
                    if bars and hasattr(bars, "df") and not bars.df.empty:
                        df = bars.df
                        if isinstance(df.index, pd.MultiIndex):
                            if ticker in df.index.levels[0]:
                                df = df.xs(ticker)
                            else:
                                df = df.droplevel(0)
                        df.columns = [str(c).lower() for c in df.columns]
                except Exception as e:
                    logger.debug(f"Alpaca daily bars error for {ticker}: {e}")
                    df = None

            if (df is None or df.empty or len(df) < 2) and self.broker and not self.dry_run:
                try:
                    asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)
                    bars = self.broker.get_historical_prices(asset, 20, "day")
                    if bars is not None:
                        df = bars.df if hasattr(bars, "df") else bars
                        if df is not None and not df.empty:
                            df.columns = [str(c).lower() for c in df.columns]
                except Exception:
                    df = None

            if df is not None and not df.empty and len(df) >= 2:
                pdc = float(df["close"].iloc[-2])
                pdh = float(df["high"].iloc[-2])
                pdl = float(df["low"].iloc[-2])
                daily_atr = float(calculate_atr(df, length=min(14, len(df) - 1))) or (pdc * 0.02)

            # Dynamic 7-day consecutive volatility pre-qualification across broad universe
            if df is not None and not df.empty and len(df) >= 7:
                now_d = datetime.now(self.tz_est).date()
                ok_zed, avg_r, _ = check_7day_consecutive_volatility(
                    ticker, now_d, df, min_daily_range_pct=0.020, lookback_days=7
                )
                if ok_zed:
                    self.qualified_7day_volatility_tickers.add(ticker)
                    is_qualified_zed = True
                    logger.info(f"[{ticker}] PASSED 7-Day Volatility Screen ({avg_r:.2%}) -> Dynamically qualified for ZedStrategy!")
        except Exception as e:
            logger.warning(f"Notice calculating daily levels for {ticker}: {e}")

        self.active_watchlist[ticker] = {
            "ticker": ticker,
            "signal": sig,
            "pdh": pdh,
            "pdl": pdl,
            "pdc": pdc,
            "daily_atr": daily_atr,
            "df_1d": df,
            "is_qualified_zed": is_qualified_zed,
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
        evaluates entry triggers for all active strategy engines.
        """
        now_est = datetime.now(self.tz_est)
        cur_time = now_est.time()

        # Update portfolio value & buying power across all ETF tiers
        self._update_account_balances()

        # Strategy Engine -1: Level 1 Boomer Haven Index DCA (Daily 1% QQQ & SPY)
        self._evaluate_l1_index_dca(now_est)

        # Periodic subaccounts sync every 60 seconds
        if time.time() - getattr(self, "last_subaccount_sync_time", 0) > 60:
            self._sync_sub_accounts_database()
            self.last_subaccount_sync_time = time.time()

        # Strategy Engine 0: Market Open Catalysts (09:30 - 09:45 AM EST)
        if cur_time >= dtime(9, 30) and cur_time < dtime(9, 50):
            self._evaluate_market_open_catalysts(now_est)

        # Iterate through watchlist tickers
        for ticker, ctx in list(self.active_watchlist.items()):
            sig: ReactorSignal = ctx["signal"]

            # Max 1 trade per ticker per day
            if ctx["trades_taken"] >= 1:
                continue

            # Fetch recent intraday 5m and 15m candles directly from Alpaca
            candles_5m, candles_15m = self._fetch_recent_intraday_candles(ticker)
            if candles_5m is None or candles_5m.empty:
                continue

            # Real-time execution price directly from Alpaca
            current_price = self.get_realtime_price(ticker)
            if not current_price or current_price <= 0:
                current_price = float(candles_5m.iloc[-1]["close"])

            # Compute or update 15m Opening Range (9:30 - 9:45 AM EST)
            if cur_time >= dtime(9, 45) and ctx["or_high"] is None:
                self._compute_opening_range(ticker, ctx, candles_5m)

            # Strategy Engine 0B: Intraday Aligned Catalyst Entry for High-Conviction News (Selectors 1 & 3)
            if any(s in self.enabled_strategies for s in ("MarketOpen_Catalyst", "HighBeta_CatalystOpen")) and sig.selector_id in (1, 3):
                if cur_time >= dtime(9, 30) and cur_time < dtime(15, 45):
                    self._evaluate_aligned_catalyst_entry(ticker, ctx, sig, current_price, now_est)

            # Strategy Engine 1: News_LiquiditySweep
            if "News_LiquiditySweep" in self.enabled_strategies and cur_time >= dtime(9, 45) and cur_time < dtime(15, 50):
                self._evaluate_liquidity_sweep_entry(ticker, ctx, sig, candles_5m, candles_15m, current_price)

            # Strategy Engine 2: MarketReactor (Fade Mode)
            if "MarketReactor_Fade" in self.enabled_strategies and sig.selector_id == 2 and cur_time >= dtime(9, 45) and cur_time < dtime(15, 55):
                self._evaluate_market_reactor_fade_entry(ticker, ctx, sig, candles_5m, current_price)

            # Strategy Engine 3: ZedStrategy / HighBeta_CatalystOpen (09:30 - 10:15 AM EST)
            if any(s in self.enabled_strategies for s in ("ZedStrategy", "HighBeta_CatalystOpen")) and cur_time >= dtime(9, 30) and cur_time <= dtime(10, 15):
                self._evaluate_zed_strategy_entry(ticker, ctx, sig, candles_5m, current_price, now_est)

            # Strategy Engine 4: ZedStrategy_Institutional (V2: WITH VWAP & RVOL) (09:35 - 10:15 AM EST)
            if any(s in self.enabled_strategies for s in ("ZedStrategy_Institutional", "ZedInstitutionalStrategy", "Institutional_VRecovery")) and cur_time >= dtime(9, 35) and cur_time <= dtime(10, 15):
                self._evaluate_zed_institutional_entry(ticker, ctx, sig, candles_5m, current_price, now_est)

        # Active position management & ratchets
        self.manage_active_positions(now_est)

    def _fetch_recent_intraday_candles(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetches 5m and 15m intraday candles directly from Alpaca Market Data API.
        Never relies on rate-limited or stale third-party providers.
        """
        try:
            if not self.data_client:
                return None, None

            from datetime import timezone
            now_utc = datetime.now(timezone.utc)
            start_utc = now_utc - timedelta(days=5)

            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_utc,
            )
            bars = self.data_client.get_stock_bars(req)
            if not bars or not hasattr(bars, "df") or bars.df.empty:
                return None, None

            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                if ticker in df.index.levels[0]:
                    df = df.xs(ticker)
                else:
                    df = df.droplevel(0)

            df.columns = [str(c).lower() for c in df.columns]

            if df.index.tz is not None:
                df.index = df.index.tz_convert(self.tz_est)
            else:
                df.index = df.index.tz_localize("UTC").tz_convert(self.tz_est)

            # Explicit column resampling
            o = df["open"].resample("15min").first()
            h = df["high"].resample("15min").max()
            l = df["low"].resample("15min").min()
            c = df["close"].resample("15min").last()
            v = df["volume"].resample("15min").sum()
            df_15m = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
            return df, df_15m
        except Exception as e:
            logger.warning(f"Error fetching Alpaca candles for {ticker}: {e}")
            return None, None

    def get_realtime_price(self, ticker: str) -> Optional[float]:
        """Fetches accurate real-time market price directly from Alpaca latest quote or bar."""
        if not self.data_client:
            return None
        try:
            q_req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = self.data_client.get_stock_latest_quote(q_req)
            if quotes and ticker in quotes:
                q = quotes[ticker]
                bid = float(q.bid_price or 0.0)
                ask = float(q.ask_price or 0.0)
                if bid > 0 and ask > 0:
                    return round((bid + ask) / 2.0, 2)
                elif ask > 0:
                    return round(ask, 2)
                elif bid > 0:
                    return round(bid, 2)
        except Exception as e:
            logger.debug(f"Could not fetch Alpaca quote for {ticker}: {e}")

        try:
            b_req = StockLatestBarRequest(symbol_or_symbols=ticker)
            bars = self.data_client.get_stock_latest_bar(b_req)
            if bars and ticker in bars and bars[ticker]:
                return round(float(bars[ticker].close), 2)
        except Exception:
            pass
        return None

    def _evaluate_market_open_catalysts(self, now_est: datetime):
        """
        Executes Market Open (09:30 AM EST) catalyst trades for all premarket news signals
        (e.g., CON, CNQ) that arrived before market open. Routes strictly to Level 2.
        """
        if not self.premarket_catalyst_queue:
            return

        for ticker, sig in list(self.premarket_catalyst_queue.items()):
            ctx = self.active_watchlist.get(ticker)
            if not ctx:
                self._initialize_ticker_watchlist(ticker, sig)
                ctx = self.active_watchlist.get(ticker)

            if not ctx or ctx.get("trades_taken", 0) >= 1:
                continue

            current_price = self.get_realtime_price(ticker)
            if not current_price or current_price <= 0:
                candles_5m, _ = self._fetch_recent_intraday_candles(ticker)
                if candles_5m is not None and not candles_5m.empty:
                    current_price = float(candles_5m.iloc[-1]["close"])

            if not current_price or current_price <= 0:
                try:
                    import yfinance as yf
                    tk_info = yf.Ticker(ticker).fast_info
                    current_price = getattr(tk_info, "last_price", None)
                except Exception:
                    pass

            if not current_price or current_price <= 0:
                logger.warning(f"[{ticker}] No real-time price available for Market Open execution.")
                continue

            daily_atr = ctx.get("daily_atr", current_price * 0.025)
            pos_side = sig.signal_position  # "BUY" or "SELL"

            if pos_side == "BUY":
                atr_stop = max(current_price - (1.2 * daily_atr), current_price * 0.965)
                sl = round(max(atr_stop, current_price * 0.970), 2)  # Max -3.0% Stop Loss
                tp = round(current_price * 1.035, 2)  # +3.5% Take Profit
                reason = f"Pre-Market Aligned Catalyst Open | {sig.news_sentiment} News ({sig.news_conviction:.0%}) | Fund: {sig.fundamental_sentiment} ({sig.fundamental_confidence:.0%})"

                self._execute_live_trade(
                    strategy_name="MarketOpen_Catalyst",
                    ticker=ticker,
                    side="BUY",
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=reason,
                    sig=sig,
                    ctx=ctx,
                    be_trigger_r=1.0,
                    eod_flatten_time=dtime(15, 55),
                )
                self.premarket_catalyst_queue.pop(ticker, None)
            elif pos_side == "SELL":
                atr_stop = min(current_price + (1.2 * daily_atr), current_price * 1.035)
                sl = round(min(atr_stop, current_price * 1.030), 2)  # Max +3.0% Stop Loss
                tp = round(current_price * 0.965, 2)  # -3.5% Take Profit
                reason = f"Pre-Market Bearish Catalyst Open | {sig.news_sentiment} News ({sig.news_conviction:.0%}) | Fund: {sig.fundamental_sentiment}"

                self._execute_live_trade(
                    strategy_name="MarketOpen_Catalyst",
                    ticker=ticker,
                    side="SELL",
                    entry_price=current_price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=reason,
                    sig=sig,
                    ctx=ctx,
                    be_trigger_r=1.0,
                    eod_flatten_time=dtime(15, 55),
                )
                self.premarket_catalyst_queue.pop(ticker, None)


    def _evaluate_l1_index_dca(self, now_est: datetime):
        """
        Executes Level 1 Boomer Haven ETF Daily Index DCA:
        Buys QQQ (1% of L1 equity) and SPY (1% of L1 equity) once per trading day during market hours.
        """
        today = now_est.date()
        if self.l1_dca_executed_date == today:
            return

        cur_time = now_est.time()
        # Execute during regular trading hours after initial opening volatility (09:35 - 15:50 EST)
        if cur_time < dtime(9, 35) or cur_time > dtime(15, 50):
            return

        l1 = self.etf_accounts.get(1)
        if not l1 or not l1.trading_client:
            return

        logger.info("[L1 INDEX DCA TRIGGER] Executing daily index DCA on Boomer Haven ETF...")
        pv = l1.portfolio_value
        target_allocation_per_index = pv * 0.01  # 1% of total account

        for symbol in ["QQQ", "SPY"]:
            price = self.get_realtime_price(symbol)
            if not price or price <= 0:
                try:
                    import yfinance as yf
                    tk_info = yf.Ticker(symbol).fast_info
                    price = getattr(tk_info, "last_price", None)
                except Exception:
                    pass

            if not price or price <= 0:
                logger.warning(f"[L1 DCA] Cannot fetch price for {symbol}. Will retry next tick.")
                return

            qty = floor(target_allocation_per_index / price)
            if qty < 1:
                qty = 1

            total_amount = round(qty * price, 2)
            logger.info(f"[L1 DCA BUY] {symbol} | {qty} shares @ ${price:.2f} (Total: ${total_amount:,.2f})")

            alpaca_order_id = None
            if not self.dry_run and l1.trading_client:
                try:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    submitted = l1.trading_client.submit_order(req)
                    alpaca_order_id = str(submitted.id)
                    time.sleep(1.0)
                    try:
                        fill = l1.trading_client.get_order_by_id(alpaca_order_id)
                        if fill and fill.filled_avg_price:
                            price = float(fill.filled_avg_price)
                            total_amount = round(qty * price, 2)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"[L1 DCA Alpaca Order Error] {symbol}: {e}")
                    continue

            pos_record = {
                "pos_id": f"L1_Index_DCA_{symbol}_{datetime.now().strftime('%H%M%S')}",
                "alpaca_order_id": alpaca_order_id,
                "strategy": "Index_DCA",
                "ticker": symbol,
                "side": "BUY",
                "entry_price": price,
                "quantity": qty,
                "stop_loss": round(price * 0.90, 2),
                "take_profit": round(price * 1.20, 2),
                "entry_time": datetime.now(self.tz_est),
                "status": "Closed",
                "etf_level": 1,
            }
            db_id = self._record_order_to_database(pos_record, total_amount, etf_level=1)
            pos_record["db_order_id"] = db_id

            self._notify_l1_dca_executed(symbol, qty, price, total_amount, alpaca_order_id)

        self.l1_dca_executed_date = today
        self._sync_sub_accounts_database()
        self._persist_state()

    def _evaluate_aligned_catalyst_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        current_price: float,
        now_est: datetime
    ):
        """Evaluates entry for high-conviction aligned catalyst signals during market hours."""
        if ctx.get("trades_taken", 0) >= 1:
            return

        daily_atr = ctx.get("daily_atr", current_price * 0.025)
        pos_side = sig.signal_position

        if pos_side == "BUY":
            sl = round(max(current_price - (1.2 * daily_atr), current_price * 0.970), 2)
            tp = round(current_price * 1.035, 2)
            reason = f"High-Conviction Aligned Catalyst Entry | {sig.news_sentiment} News ({sig.news_conviction:.0%}) | Fund: {sig.fundamental_sentiment}"
            self._execute_live_trade(
                strategy_name="MarketOpen_Catalyst",
                ticker=ticker,
                side="BUY",
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                reason=reason,
                sig=sig,
                ctx=ctx,
                be_trigger_r=1.0,
                eod_flatten_time=dtime(15, 55),
            )
        elif pos_side == "SELL":
            sl = round(min(current_price + (1.2 * daily_atr), current_price * 1.030), 2)
            tp = round(current_price * 0.965, 2)
            reason = f"High-Conviction Bearish Catalyst Entry | {sig.news_sentiment} News ({sig.news_conviction:.0%}) | Fund: {sig.fundamental_sentiment}"
            self._execute_live_trade(
                strategy_name="MarketOpen_Catalyst",
                ticker=ticker,
                side="SELL",
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                reason=reason,
                sig=sig,
                ctx=ctx,
                be_trigger_r=1.0,
                eod_flatten_time=dtime(15, 55),
            )

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
    # STRATEGY 3: ZED STRATEGY (HIGH-BETA CATALYST OPENING)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_zed_strategy_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        df_5m: pd.DataFrame,
        current_price: float,
        now_est: datetime,
    ):
        """
        Executes ZedStrategy 5-step rulebook across high-beta equities:
        1. 7-Day Consecutive Volatility Screen (moves >= 2.0% daily 7 days in a row)
        2. Higher-timeframe Nasdaq Composite (QQQ) regime
        3. 30-minute market open patterns (Golden V-Shape, Straight Rally, Bear False Recovery, Bull Flag)
        4. Catalyst Confluence (> 4 or < 2 on stock/peers)
        5. Mechanical +2.0% Take Profit, -3.5% Stop Loss, +1.0% Breakeven Ratchet
        """
        # Step 1: Verify 7-Day Consecutive Volatility Screen across dynamic universe
        if ticker not in self.qualified_7day_volatility_tickers:
            df_1d = ctx.get("df_1d")
            if df_1d is not None and not df_1d.empty and len(df_1d) >= 7:
                is_ok, avg_r, _ = check_7day_consecutive_volatility(
                    ticker, now_est.date(), df_1d, min_daily_range_pct=0.020, lookback_days=7
                )
                if is_ok:
                    self.qualified_7day_volatility_tickers.add(ticker)
                    logger.info(f"[{ticker}] PASSED 7-Day Volatility Screen ({avg_r:.2%}) -> Active in ZedStrategy universe.")
                else:
                    return
            else:
                return

        pdc = ctx.get("pdc")
        daily_atr = ctx.get("daily_atr", current_price * 0.025)

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
            qqq_px = self.get_realtime_price("QQQ") or qqq_pdc
            if qqq_px < qqq_pdc:
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
                strategy_name="ZedStrategy",
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

    # Backward compatibility alias
    _evaluate_high_beta_catalyst_open_entry = _evaluate_zed_strategy_entry

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 4: ZED INSTITUTIONAL V-RECOVERY (5 QUANTITATIVE ENHANCEMENTS)
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_zed_institutional_entry(
        self,
        ticker: str,
        ctx: Dict[str, Any],
        sig: ReactorSignal,
        candles_5m: Optional[pd.DataFrame],
        current_price: float,
        now_est: datetime,
    ):
        """
        Executes ZedInstitutionalStrategy with all 5 upgrades:
        1. 7-Day Consecutive Volatility Screen (>= 2.0% daily)
        2. Volume Profile & Intraday VWAP Confluence
        3. Higher-Timeframe Nasdaq (QQQ) Bull Market Filter + Relative Strength Divergence
        4. Structural Volatility Stop (1.2x of 15m ATR below session low, max 3.0%)
        5. Scaled Profit Targets (+2.0% TP1, +3.5% TP2 Runner, +1.0% Breakeven Ratchet)
        """
        # Step 1: 7-Day Volatility Qualification across dynamic universe
        if ticker not in self.qualified_7day_volatility_tickers:
            df_1d = ctx.get("df_1d")
            if df_1d is not None and not df_1d.empty and len(df_1d) >= 7:
                is_ok, avg_r, _ = check_7day_consecutive_volatility(
                    ticker, now_est.date(), df_1d, min_daily_range_pct=0.020, lookback_days=7
                )
                if is_ok:
                    self.qualified_7day_volatility_tickers.add(ticker)
                else:
                    return
            else:
                return

        today = now_est.date()
        date_str = today.strftime("%Y-%m-%d")

        if candles_5m is None or candles_5m.empty:
            return

        today_bars = candles_5m[candles_5m.index.date == today]
        if len(today_bars) < 3:
            return

        # Fetch daily levels & 20D ATR via Alpaca
        pdc = ctx.get("pdc")
        daily_atr = ctx.get("daily_atr")
        if not pdc or not daily_atr:
            try:
                if self.data_client:
                    from datetime import timezone
                    start_utc = datetime.now(timezone.utc) - timedelta(days=40)
                    req_d = StockBarsRequest(symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=start_utc)
                    b_d = self.data_client.get_stock_bars(req_d)
                    if b_d and hasattr(b_d, "df") and not b_d.df.empty:
                        df_d = b_d.df
                        if isinstance(df_d.index, pd.MultiIndex):
                            df_d = df_d.xs(ticker) if ticker in df_d.index.levels[0] else df_d.droplevel(0)
                        df_d.columns = [str(c).lower() for c in df_d.columns]
                        if len(df_d) >= 2:
                            pdc = float(df_d["close"].iloc[-2])
                            daily_atr = float(calculate_atr(df_d, length=min(14, len(df_d) - 1))) or (pdc * 0.025)
                            ctx["pdc"] = pdc
                            ctx["daily_atr"] = daily_atr
            except Exception:
                pass

        if not pdc or not daily_atr or (daily_atr / pdc) < 0.020:
            return

        # Fetch QQQ 5m & higher-timeframe regime via Alpaca
        qqq_5m = ctx.get("qqq_5m")
        qqq_regime = ctx.get("qqq_regime")
        if not qqq_regime:
            try:
                if self.data_client:
                    from datetime import timezone
                    start_utc = datetime.now(timezone.utc) - timedelta(days=5)
                    req_q = StockBarsRequest(symbol_or_symbols="QQQ", timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start_utc)
                    b_q = self.data_client.get_stock_bars(req_q)
                    if b_q and hasattr(b_q, "df") and not b_q.df.empty:
                        df_q = b_q.df
                        if isinstance(df_q.index, pd.MultiIndex):
                            df_q = df_q.xs("QQQ") if "QQQ" in df_q.index.levels[0] else df_q.droplevel(0)
                        df_q.columns = [str(c).lower() for c in df_q.columns]
                        q_open = float(df_q["open"].iloc[0])
                        q_curr = float(df_q["close"].iloc[-1])
                        qqq_regime = "BULL_MARKET" if q_curr >= q_open * 0.998 else "BEAR_MARKET"
                        ctx["qqq_regime"] = qqq_regime
                        ctx["qqq_5m"] = df_q
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
            q_dropping = float(q_last["low"]) <= float(q_prev["low"])
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
                strategy_name="ZedInstitutionalStrategy",
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

    # Backward compatibility alias
    _evaluate_institutional_v_recovery_entry = _evaluate_zed_institutional_entry

    # ─────────────────────────────────────────────────────────────────────────
    # 4. LIVE ORDER EXECUTION & DB ORDERS SYNCHRONIZATION
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # 4. MULTI-TIER ETF ORDER EXECUTION & DB ORDERS SYNCHRONIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def _get_etf_level_for_strategy(self, strategy_name: str) -> int:
        """Determines target ETF level based on strategy rules."""
        s = strategy_name.lower()
        if "zed" in s or "institutional" in s or "sweep" in s or "liquidity" in s:
            return 3  # Level 3: Diamond Hands
        elif "reactor" in s or "fade" in s or "catalyst" in s or "marketopen" in s:
            return 2  # Level 2: Steady Grind
        elif "dca" in s or "index" in s or "boomer" in s:
            return 1  # Level 1: Boomer Haven
        return 3

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
        target_level = self._get_etf_level_for_strategy(strategy_name)
        etf_ctx = self.etf_accounts.get(target_level)
        if not etf_ctx:
            logger.error(f"ETF account context for Level {target_level} not found. Aborting execution.")
            return

        # Strict single-position protection: prevent duplicate orders for same ticker on this ETF
        if any(p.get("ticker") == ticker and p.get("etf_level") == target_level for p in self.active_positions.values()):
            logger.info(f"[{ticker}] Active position already open in ETF Level {target_level}. Skipping duplicate execution.")
            return

        if ctx.get("trades_taken", 0) >= 1:
            logger.info(f"[{ticker}] Maximum daily trades already reached (1/1). Skipping execution.")
            return

        # Mark trade immediately to lock out race conditions
        ctx["trades_taken"] = ctx.get("trades_taken", 0) + 1

        # Conviction-weighted sizing: 1.0% to 2.5%
        comp_score = getattr(sig, "composite_conviction", 0.70)
        norm_score = max(0.0, min(1.0, (comp_score - 0.70) / 0.30))
        risk_pct = 0.010 + (norm_score * 0.015)

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            ctx["trades_taken"] = max(0, ctx.get("trades_taken", 1) - 1)
            return

        portfolio_to_risk = etf_ctx.portfolio_value or (100000.0 if target_level == 2 else 1000000.0)

        # Strict 5% portfolio allocation cap per trade (prevents going all in)
        MAX_TRADE_ALLOCATION_PCT = etf_ctx.max_trade_pct
        max_trade_notional = portfolio_to_risk * MAX_TRADE_ALLOCATION_PCT
        max_qty_from_capital = floor(max_trade_notional / max(0.01, entry_price))

        # Risk-based lot sizing (1.0% to 2.5% risk budget)
        qty_from_risk = floor((portfolio_to_risk * risk_pct) / risk_per_share)

        # Strictly cap quantity so position never exceeds 5% of portfolio value
        qty = min(qty_from_risk, max_qty_from_capital)

        # Ensure at least 1 share if affordable
        if qty < 1 and entry_price <= max_trade_notional * 1.10:
            qty = 1

        # Check broker buying power buffer
        if etf_ctx.buying_power > 0:
            max_bp_qty = floor((etf_ctx.buying_power * 0.90) / max(0.01, entry_price))
            qty = min(qty, max_bp_qty)

        if qty <= 0:
            logger.warning(
                f"[{ticker}] Calculated quantity is 0 shares (ETF L{target_level} 5% Cap=${max_trade_notional:,.2f} | "
                f"Entry=${entry_price:.2f}). Skipping order execution."
            )
            ctx["trades_taken"] = max(0, ctx.get("trades_taken", 1) - 1)
            return

        total_amount = round(qty * entry_price, 2)
        pos_id = f"L{target_level}_{strategy_name}_{ticker}_{datetime.now().strftime('%H%M%S')}"

        actual_alloc_pct = (total_amount / portfolio_to_risk) if portfolio_to_risk > 0 else 0.0
        logger.info(
            f"[ORDER TRIGGER] Level {target_level} ({etf_ctx.name}) | {strategy_name} | {ticker} {side} {qty} shares @ ${entry_price:.2f} | "
            f"Notional=${total_amount:,.2f} ({actual_alloc_pct:.2%} of L{target_level} PV, <=5% cap) | "
            f"SL=${stop_loss:.2f} | TP=${take_profit:.2f} | Risk=${risk_per_share * qty:,.2f} | {reason}"
        )

        alpaca_order_id = None
        if not self.dry_run:
            if not etf_ctx.trading_client:
                logger.error(f"[Alpaca Error] Trading client for ETF Level {target_level} not initialized. Cannot execute order.")
                return

            try:
                # Alpaca bracket validation safety guard: ensure SL & TP are strictly on correct sides of entry
                min_spread = max(0.05, round(entry_price * 0.005, 2))
                if side.upper() == "BUY":
                    if stop_loss >= entry_price - min_spread:
                        stop_loss = round(entry_price - min_spread, 2)
                    if take_profit <= entry_price + min_spread:
                        take_profit = round(entry_price + min_spread, 2)
                else:  # SELL / Short
                    if stop_loss <= entry_price + min_spread:
                        stop_loss = round(entry_price + min_spread, 2)
                    if take_profit >= entry_price - min_spread:
                        take_profit = round(entry_price - min_spread, 2)

                order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
                req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                    stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
                )
                submitted = etf_ctx.trading_client.submit_order(order_data=req)
                alpaca_order_id = str(submitted.id)
                logger.info(f"[Alpaca L{target_level} Bracket Order Placed] Order ID: {alpaca_order_id} | Status: {submitted.status}")

                time.sleep(1.0)
                try:
                    fill_chk = etf_ctx.trading_client.get_order_by_id(alpaca_order_id)
                    if fill_chk and fill_chk.filled_avg_price:
                        entry_price = float(fill_chk.filled_avg_price)
                        total_amount = round(qty * entry_price, 2)
                        logger.info(f"[Alpaca Fill Confirmed] Actual Fill Price: ${entry_price:.2f} (Total: ${total_amount:,.2f})")
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"[Alpaca L{target_level} Order REJECTED/FAILED] Could not submit {side} order for {ticker}: {e}")
                ctx["trades_taken"] = max(0, ctx.get("trades_taken", 1) - 1)
                return

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
            "etf_level": target_level,
            "db_order_id": None,
        }

        # Write to MySQL `orders` table with exact etf_level and prefix
        db_id = self._record_order_to_database(pos_record, total_amount, etf_level=target_level)
        pos_record["db_order_id"] = db_id

        self.active_positions[pos_id] = pos_record
        etf_ctx.active_positions[pos_id] = pos_record
        self._persist_state()
        self._sync_sub_accounts_database()

        # Send Real-Time Alert to Telegram Trades Bot (@gotti_trades_bot)
        self._notify_trade_opened(pos_record, total_amount, reason)

    def _record_order_to_database(self, pos: Dict[str, Any], total_amount: float, etf_level: int = 3) -> Optional[int]:
        """Inserts trade record into MySQL `orders` table with etf_level."""
        conn = None
        cursor = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()
            order_date_str = pos["entry_time"].strftime("%d/%m/%y")
            query = """
            INSERT INTO orders (
                order_date, symbol, name, etf_level, status, buy_price, sell_price, profit, order_amount, total_amount, order_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            buy_px = pos["entry_price"] if pos["side"] == "BUY" else None
            sell_px = pos["entry_price"] if pos["side"] == "SELL" else None
            strat_name = pos['strategy']
            # Do not add [L1], [L2], [L3] to order names per user requirement
            full_name = f"{strat_name} Live" if not strat_name.endswith("Live") else strat_name

            params = (
                order_date_str,
                pos["ticker"],
                full_name,
                etf_level,
                pos.get("status", "Open"),
                buy_px,
                sell_px,
                pos.get("profit"),
                pos["quantity"],
                total_amount,
                pos["side"].capitalize(),
            )
            cursor.execute(query, params)
            conn.commit()
            db_id = cursor.lastrowid
            logger.info(f"Recorded order in MySQL `orders` table (ID: {db_id}, ETF: L{etf_level}, Name: {full_name}).")
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
            conn.commit()
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

            current_price = self.get_realtime_price(ticker) or entry_p

            gain = (current_price - entry_p) if side == "BUY" else (entry_p - current_price)
            r_mult = gain / initial_risk if initial_risk > 0 else 0.0

            # 1. Breakeven Stop Ratchet at +1.2R
            if not pos.get("breakeven_ratchet", False) and r_mult >= pos.get("be_trigger_r", 1.2):
                pos["stop_loss"] = entry_p
                pos["breakeven_ratchet"] = True
                logger.info(f"[BREAKEVEN RATCHET] {ticker} reached +{r_mult:.1f}R! Stop loss moved to Entry (${entry_p:.2f}).")

            # 2. Strict EOD Flatten (NEVER flatten Level 1 Index DCA Buy & Hold)
            if pos.get("strategy") == "Index_DCA" or pos.get("etf_level") == 1:
                continue

            eod_time = pos.get("eod_flatten_time", dtime(15, 50))
            if cur_time >= eod_time:
                logger.info(f"[EOD FLATTEN] Flattening {ticker} ({pos['strategy']}) at {cur_time.strftime('%H:%M:%S')} EST.")
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
        """Flattens position with Alpaca broker for its respective ETF tier and finalizes accounting."""
        ticker = pos["ticker"]
        side = pos["side"]
        qty = pos["quantity"]
        target_level = pos.get("etf_level", 3)
        etf_ctx = self.etf_accounts.get(target_level)

        if not self.dry_run and etf_ctx and etf_ctx.trading_client:
            try:
                try:
                    req_orders = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
                    ticker_orders = etf_ctx.trading_client.get_orders(req_orders)
                    for o in ticker_orders:
                        etf_ctx.trading_client.cancel_order_by_id(o.id)
                except Exception:
                    pass
                try:
                    etf_ctx.trading_client.close_position(ticker)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Notice closing position via Alpaca L{target_level}: {e}")

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
            f"[POSITION CLOSED] L{target_level} | {pos['strategy']} | {ticker} | Exit: ${exit_price:.2f} | "
            f"P&L: ${profit:+,.2f} | Reason: {exit_reason}"
        )

        self._update_order_in_database(pos, exit_price, profit)
        self.daily_completed_trades.append(pos)
        self._notify_trade_closed(pos, exit_price, profit, exit_reason)
        self._sync_sub_accounts_database()

    def _notify_trade_opened(self, pos: Dict[str, Any], notional: float, reason: str):
        """Dispatches rich HTML alert to Telegram @gotti_trades_bot upon new execution."""
        lvl = pos.get("etf_level", 3)
        lvl_icons = {
            1: "🛡️ [Level 1: Boomer Haven ETF]",
            2: "⚖️ [Level 2: Steady Grind ETF]",
            3: "🚀 [Level 3: Diamond Hands ETF]",
        }
        lvl_header = lvl_icons.get(lvl, f"[Level {lvl} ETF]")
        side_icon = "🟢 BUY" if pos["side"] == "BUY" else "🔴 SELL"
        msg = (
            "🚀 <b>NEW TRADE EXECUTED</b>\n\n"
            f"<b>Portfolio Tier:</b> <code>{lvl_header}</code>\n"
            f"<b>Strategy:</b> <code>{pos['strategy']}</code>\n"
            f"<b>Asset:</b> <b>{pos['ticker']}</b> | {side_icon}\n"
            f"<b>Quantity:</b> <code>{pos['quantity']:,}</code> shares\n"
            f"<b>Entry Price:</b> <code>${pos['entry_price']:.2f}</code> (Alpaca Executed)\n"
            f"<b>Total Capital:</b> <code>${notional:,.2f}</code> (<=5% cap)\n"
            f"<b>Bracket SL:</b> <code>${pos['stop_loss']:.2f}</code> | <b>TP:</b> <code>${pos['take_profit']:.2f}</code>\n"
            f"<b>Broker Order ID:</b> <code>{pos.get('alpaca_order_id') or 'N/A'}</code>\n"
            f"<b>Confluence:</b> <i>{html.escape(reason)}</i>\n\n"
            f"<i>⚡ Tracked live on {lvl_header}</i>"
        )
        self._notify_telegram(msg)

    def _notify_trade_closed(self, pos: Dict[str, Any], exit_price: float, profit: float, exit_reason: str):
        """Dispatches rich HTML alert to Telegram @gotti_trades_bot upon position close."""
        lvl = pos.get("etf_level", 3)
        lvl_icons = {
            1: "🛡️ [Level 1: Boomer Haven ETF]",
            2: "⚖️ [Level 2: Steady Grind ETF]",
            3: "🚀 [Level 3: Diamond Hands ETF]",
        }
        lvl_header = lvl_icons.get(lvl, f"[Level {lvl} ETF]")
        pnl_badge = "🟢" if profit > 0 else ("🔴" if profit < 0 else "⚪")
        cost = pos["quantity"] * pos["entry_price"]
        pct = (profit / cost * 100) if cost > 0 else 0.0
        msg = (
            "🏁 <b>POSITION CLOSED</b>\n\n"
            f"<b>Portfolio Tier:</b> <code>{lvl_header}</code>\n"
            f"<b>Strategy:</b> <code>{pos['strategy']}</code>\n"
            f"<b>Asset:</b> <b>{pos['ticker']}</b> ({pos['side']})\n"
            f"<b>Exit Price:</b> <code>${exit_price:.2f}</code> (Entry: ${pos['entry_price']:.2f})\n"
            f"<b>Realized P&L:</b> {pnl_badge} <b><code>${profit:+,.2f}</code></b> (<b>{pct:+.2f}%</b>)\n"
            f"<b>Exit Reason:</b> <code>{exit_reason}</code>\n"
            f"<b>Executed At:</b> <code>{datetime.now(self.tz_est).strftime('%H:%M:%S EST')}</code>\n\n"
            "<i>💡 Use /trades or /pnl in @gotti_trades_bot to view updated performance.</i>"
        )
        self._notify_telegram(msg)

    def _notify_l1_dca_executed(self, symbol: str, qty: int, price: float, notional: float, order_id: Optional[str]):
        """Dispatches notification for Level 1 Index DCA execution."""
        msg = (
            "🛡️ <b>DAILY INDEX DCA EXECUTED</b>\n\n"
            "<b>Portfolio Tier:</b> <code>Level 1: Boomer Haven ETF</code>\n"
            "<b>Strategy:</b> <code>Index_DCA (Daily 1% Accumulation)</code>\n"
            f"<b>Asset:</b> <b>{symbol}</b> | 🟢 BUY\n"
            f"<b>Quantity:</b> <code>{qty:,}</code> shares\n"
            f"<b>Executed Price:</b> <code>${price:.2f}</code>\n"
            f"<b>Total Allocation:</b> <code>${notional:,.2f}</code> (~1.0% of L1 Equity)\n"
            f"<b>Broker Order ID:</b> <code>{order_id or 'N/A'}</code>\n\n"
            "<i>🏛️ Managed by Gotti Defensive Fund Accumulation</i>"
        )
        self._notify_telegram(msg)

    def _update_account_balances(self):
        """Refreshes account balance and portfolio value from Alpaca broker across all ETF tiers."""
        if not self.dry_run:
            for lvl, etf_ctx in self.etf_accounts.items():
                if etf_ctx.trading_client:
                    try:
                        acct = etf_ctx.trading_client.get_account()
                        etf_ctx.portfolio_value = float(acct.portfolio_value or etf_ctx.portfolio_value)
                        etf_ctx.buying_power = float(acct.buying_power or etf_ctx.buying_power)
                        etf_ctx.cash = float(acct.cash or etf_ctx.cash)
                    except Exception:
                        pass
            if 3 in self.etf_accounts:
                self.portfolio_value = self.etf_accounts[3].portfolio_value
                self.buying_power = self.etf_accounts[3].buying_power

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
        default="News_LiquiditySweep,MarketReactor_Fade,ZedStrategy_Institutional,MarketOpen_Catalyst",
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
