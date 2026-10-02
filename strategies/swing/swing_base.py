"""
SwingStrategyBase
=================
Abstract base class for all swing (1D) trading strategies.

Key features vs. the old approach:
  - Pyramiding: up to MAX_PYRAMIDS simultaneous bracket orders on the SAME ticker,
    each capped so total portfolio risk stays ≤ TOTAL_RISK_PCT.
  - Bracket orders: every entry submits take_profit_price + stop_loss_price
    directly to the exchange (Lumibot OCA bracket), so the broker manages exits
    automatically without bar-by-bar polling.
  - Unique Trade IDs: every entry gets a UUID stored in self._active_trades dict,
    keyed by Lumibot order.identifier. on_filled_order() matches fills back to IDs.
  - Dual-loop iteration: each bar (a) scans for new setups, (b) manages live
    positions whose bracket hasn't filled yet (e.g. manual trailing update).
  - Parameter DB hook: load_parameters_from_db() is implemented but commented out
    at call-site so you can wire it up when ready.
"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from typing import Optional, Dict, Any, List
from math import floor

import pandas as pd
import pandas_ta as ta
from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset


class SwingStrategyBase(Strategy):
    """
    Abstract swing-trading base. Child classes only implement:
        get_setup_signal(df, atr, current_price) -> Optional[Dict]
        get_strategy_name() -> str   (for logging)

    get_setup_signal must return None or a dict with keys:
        side          : "buy" | "sell"
        take_profit   : float
        stop_loss     : float
        setup_tag     : str   (e.g. "HCR_LONG", "GAP_SHORT") — used as dedup key
    """

    # ── Defaults (child classes override via their own `parameters` dict) ──────
    parameters: Dict[str, Any] = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RiskPct": 0.02,           # Risk % per individual trade leg
        "MAX_PYRAMIDS": 3,         # Max concurrent open positions on this ticker
        "TOTAL_RISK_PCT": 0.06,    # Hard cap: total open risk ≤ 6 % of portfolio
        "ATR_Multiplier": 1.5,     # ATR multiplier used for stop distance
        "ATR_Length": 14,
        "Lookback": 40,            # Bars of daily history to fetch
        "Plot": False,
    }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self):
        self.sleeptime = "1D"
        self._load_params()

        # active_trades: { lumibot_order_id -> trade_meta_dict }
        # Each entry holds: trade_id, side, entry_price, take_profit, stop_loss,
        #                   quantity, setup_tag, status ("open"|"closed")
        self._active_trades: Dict[str, Dict[str, Any]] = {}

        # Dedup: set of setup_tags that already have an open order today
        self._open_setup_tags: set = set()

        # Running total risk of currently open trades (as fraction of portfolio)
        self._total_open_risk: float = 0.0

        self.log_message(
            f"[{self.get_strategy_name()}] Initialized | "
            f"MAX_PYRAMIDS={self._max_pyramids} | "
            f"TOTAL_RISK_PCT={self._total_risk_pct:.1%}"
        )

    def _load_params(self):
        """Pull parameters into typed instance attributes."""
        ticker = self.parameters.get("Ticker", "NVDA")
        self._symbol: str = ticker.symbol if hasattr(ticker, "symbol") else str(ticker)
        self._risk_pct: float = float(self.parameters.get("RiskPct", 0.02))
        self._max_pyramids: int = int(self.parameters.get("MAX_PYRAMIDS", 3))
        self._total_risk_pct: float = float(self.parameters.get("TOTAL_RISK_PCT", 0.06))
        self._atr_mult: float = float(self.parameters.get("ATR_Multiplier", 1.5))
        self._atr_len: int = int(self.parameters.get("ATR_Length", 14))
        self._lookback: int = int(self.parameters.get("Lookback", 40))

    # ── Main loop ──────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        """
        Every daily bar:
          A) Scan for new setups  (pyramiding-aware)
          B) Review open positions (log status; bracket handles actual exits)
        """
        # ── Fetch data ─────────────────────────────────────────────────────────
        bars = self.get_historical_prices(self._symbol, self._lookback, "day")
        if bars is None or len(bars.df) < max(20, self._lookback // 2):
            return

        df = bars.df.copy()
        atr_col = f"ATRr_{self._atr_len}"
        df.ta.atr(length=self._atr_len, append=True)
        if atr_col not in df.columns or pd.isna(df[atr_col].iloc[-1]):
            return

        atr = float(df[atr_col].iloc[-1])
        current_price = self.get_last_price(self._symbol)
        if current_price is None:
            return

        # ── A) Scan for new setups ─────────────────────────────────────────────
        self._scan_new_setup(df, atr, current_price)

        # ── B) Review open positions ───────────────────────────────────────────
        self._review_open_positions(current_price, atr)

    # ── A) New setup scan ─────────────────────────────────────────────────────

    def _scan_new_setup(self, df: pd.DataFrame, atr: float, current_price: float):
        """
        Ask the child class for a signal; gate it with pyramiding & risk caps,
        then submit a bracket order to the exchange.
        """
        open_count = self._count_open_trades()

        # Gate 1: pyramid limit
        if open_count >= self._max_pyramids:
            return

        # Gate 2: total risk cap
        if self._total_open_risk >= self._total_risk_pct:
            return

        # Ask child for a signal
        signal = self.get_setup_signal(df, atr, current_price)
        if signal is None:
            return

        side: str = signal["side"]               # "buy" | "sell"
        take_profit: float = signal["take_profit"]
        stop_loss: float = signal["stop_loss"]
        setup_tag: str = signal.get("setup_tag", self.get_strategy_name())

        # Gate 3: no duplicate setup_tag already open
        if setup_tag in self._open_setup_tags:
            return

        # Gate 4: risk-per-trade stays within per-leg cap
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share <= 0:
            return

        portfolio_value = self.get_portfolio_value()
        max_risk_this_trade = min(
            self._risk_pct,                          # per-trade cap
            self._total_risk_pct - self._total_open_risk  # remaining budget
        )
        risk_budget = portfolio_value * max_risk_this_trade
        quantity = floor(risk_budget / risk_per_share)

        if quantity < 1:
            self.log_message(
                f"[{self.get_strategy_name()}] {setup_tag}: qty=0 (budget ${risk_budget:.0f} / "
                f"risk ${risk_per_share:.2f}). Skipping."
            )
            return

        # Generate a unique trade ID
        trade_id = str(uuid.uuid4())[:8].upper()

        # Record meta BEFORE submission (order.identifier available after submit)
        trade_meta = {
            "trade_id":    trade_id,
            "setup_tag":   setup_tag,
            "side":        side,
            "entry_price": current_price,
            "take_profit": round(take_profit, 2),
            "stop_loss":   round(stop_loss, 2),
            "quantity":    quantity,
            "status":      "pending",      # → "open" on entry fill, "closed" on exit fill
            "pnl":         None,
        }

        # Submit bracket order to the exchange
        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=quantity,
            side=side,
            take_profit_price=round(take_profit, 2),
            stop_loss_price=round(stop_loss, 2),
        )
        self.submit_order(order)

        # Store in active trades keyed by Lumibot order identifier
        self._active_trades[order.identifier] = trade_meta
        self._open_setup_tags.add(setup_tag)

        # Update running risk
        trade_risk_pct = (risk_per_share * quantity) / portfolio_value
        self._total_open_risk += trade_risk_pct

        self.log_message(
            f"\n{'='*60}\n"
            f"[ENTRY #{trade_id}] {setup_tag}\n"
            f"  Side:        {side.upper()}\n"
            f"  Quantity:    {quantity} @ ${current_price:.2f}\n"
            f"  Take Profit: ${take_profit:.2f}\n"
            f"  Stop Loss:   ${stop_loss:.2f}\n"
            f"  Leg Risk:    {trade_risk_pct:.2%} | Total Open Risk: {self._total_open_risk:.2%}\n"
            f"  Open Trades: {open_count + 1} / {self._max_pyramids}\n"
            f"{'='*60}"
        )

    # ── B) Open position review ────────────────────────────────────────────────

    def _review_open_positions(self, current_price: float, atr: float):
        """
        Log the status of every open bracket trade.
        The actual TP/SL fills are handled broker-side.
        Override in child if you need to ratchet trailing stops manually.
        """
        for order_id, meta in list(self._active_trades.items()):
            if meta["status"] != "open":
                continue

            side = meta["side"]
            tp = meta["take_profit"]
            sl = meta["stop_loss"]

            if side == "buy":
                dist_tp = tp - current_price
                dist_sl = current_price - sl
            else:
                dist_tp = current_price - tp
                dist_sl = sl - current_price

            self.log_message(
                f"[{self.get_strategy_name()}] #{meta['trade_id']} ({meta['setup_tag']}) "
                f"Price=${current_price:.2f} | "
                f"ΔTP={dist_tp:+.2f} | ΔSL={dist_sl:+.2f}"
            )

    # ── Order fill handler ────────────────────────────────────────────────────

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """
        Match each fill back to its trade_id.
        Entry fill  → mark status 'open'.
        Exit fill   → mark status 'closed', compute PnL, clean up.
        """
        order_id = order.identifier
        # Child orders (TP/SL legs) may have their own identifier; search parent map
        meta = self._active_trades.get(order_id)

        # Also search children (Lumibot attaches child_orders to the bracket)
        if meta is None:
            for parent_id, m in self._active_trades.items():
                parent_order = self._get_order_by_id(parent_id)
                if parent_order and getattr(parent_order, "child_orders", None):
                    for child in parent_order.child_orders:
                        if child.identifier == order_id:
                            meta = m
                            # Re-key so future lookups hit instantly
                            self._active_trades[order_id] = meta
                            break
                if meta:
                    break

        if meta is None:
            return  # Order not managed by this strategy instance

        now = self.get_datetime().strftime("%Y-%m-%d %H:%M")
        side = meta["side"]
        trade_id = meta["trade_id"]

        # Determine if this fill is an ENTRY or an EXIT
        is_entry = (order.side == side)
        is_exit  = not is_entry

        if is_entry and meta["status"] == "pending":
            meta["status"] = "open"
            meta["entry_price"] = price   # actual fill may differ from signal price
            self.log_message(
                f"[ENTRY CONFIRMED #{trade_id}] {now} | {side.upper()} "
                f"{quantity} @ ${price:.2f}"
            )

        elif is_exit and meta["status"] == "open":
            meta["status"] = "closed"
            entry = meta["entry_price"]
            if side == "buy":
                pnl = (price - entry) * quantity
            else:
                pnl = (entry - price) * quantity
            meta["pnl"] = round(pnl, 2)
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            icon = "✓" if pnl >= 0 else "✗"
            exit_reason = self._classify_exit(price, meta)

            # Release risk budget
            risk_per_share = abs(entry - meta["stop_loss"])
            trade_risk_pct = (risk_per_share * quantity) / max(1, self.get_portfolio_value())
            self._total_open_risk = max(0.0, self._total_open_risk - trade_risk_pct)

            # Allow same setup_tag to trigger again
            self._open_setup_tags.discard(meta["setup_tag"])

            self.log_message(
                f"\n{'='*60}\n"
                f"{icon} [EXIT #{trade_id}] {meta['setup_tag']} | {now}\n"
                f"  Reason:    {exit_reason}\n"
                f"  Exit:      {quantity} @ ${price:.2f}\n"
                f"  Realized:  {pnl_str}\n"
                f"  Remaining open risk: {self._total_open_risk:.2%}\n"
                f"{'='*60}"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _count_open_trades(self) -> int:
        return sum(1 for m in self._active_trades.values() if m["status"] in ("pending", "open"))

    def _classify_exit(self, exit_price: float, meta: Dict) -> str:
        tp = meta["take_profit"]
        sl = meta["stop_loss"]
        tol = 0.01  # 1 % tolerance
        if meta["side"] == "buy":
            if exit_price >= tp * (1 - tol): return "TP"
            if exit_price <= sl * (1 + tol): return "SL"
        else:
            if exit_price <= tp * (1 + tol): return "TP"
            if exit_price >= sl * (1 - tol): return "SL"
        return "MANUAL"

    def _get_order_by_id(self, order_id: str):
        """Try to retrieve a Lumibot order object by identifier from broker."""
        try:
            orders = self.get_orders()
            for o in orders:
                if o.identifier == order_id:
                    return o
        except Exception:
            pass
        return None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def get_setup_signal(
        self,
        df: pd.DataFrame,
        atr: float,
        current_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Return a trade signal dict or None.

        Dict keys:
            side        : "buy" | "sell"
            take_profit : float
            stop_loss   : float
            setup_tag   : str   (unique name per pattern instance, e.g. "HCR_LONG")
        """
        ...

    def get_strategy_name(self) -> str:
        return self.__class__.__name__

    # ── Parameter DB hook (COMMENTED OUT — wire up when ready) ───────────────

    # def before_market_opens(self):
    #     """Called once ~20 minutes before the session opens."""
    #     self._load_parameters_from_db()
    #
    # def _load_parameters_from_db(self):
    #     """
    #     Fetch strategy parameters from the `strategy_parameters` SQLite table
    #     and update self.parameters + instance attributes in-place.
    #
    #     Table: strategy_parameters
    #     Columns: strategy_name TEXT, param_key TEXT, param_value TEXT,
    #              param_type TEXT, updated_at TIMESTAMP
    #
    #     Example:
    #         SELECT param_key, param_value, param_type
    #         FROM strategy_parameters
    #         WHERE strategy_name = '<StrategyClassName>'
    #     """
    #     import sqlite3, os
    #     db_path = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "strategy_params.db"
    #     )
    #     if not os.path.exists(db_path):
    #         self.log_message(f"[{self.get_strategy_name()}] Params DB not found at {db_path}, using defaults.")
    #         return
    #
    #     try:
    #         conn = sqlite3.connect(db_path)
    #         cursor = conn.execute(
    #             "SELECT param_key, param_value, param_type FROM strategy_parameters "
    #             "WHERE strategy_name = ?",
    #             (self.get_strategy_name(),)
    #         )
    #         rows = cursor.fetchall()
    #         conn.close()
    #
    #         type_map = {"int": int, "float": float, "bool": lambda x: x.lower() == "true", "str": str}
    #         updated = []
    #         for key, value, ptype in rows:
    #             caster = type_map.get(ptype, str)
    #             try:
    #                 self.parameters[key] = caster(value)
    #                 updated.append(f"{key}={value}")
    #             except Exception as e:
    #                 self.log_message(f"  ⚠ Could not cast {key}={value} as {ptype}: {e}")
    #
    #         if updated:
    #             self._load_params()   # Refresh instance attributes from updated parameters
    #             self.log_message(
    #                 f"[{self.get_strategy_name()}] DB params loaded: {', '.join(updated)}"
    #             )
    #         else:
    #             self.log_message(
    #                 f"[{self.get_strategy_name()}] No DB params found for this strategy. Using defaults."
    #             )
    #     except Exception as e:
    #         self.log_message(f"[{self.get_strategy_name()}] DB param load error: {e}")
