"""
SMC Intraday Bias Strategy (smc_bias_strategy.py)
=================================================
Automated implementation of Lewis Kelly's 5-Rule SMC Intraday Bias Model:
  Rule 1: Directional Bias via The Box Method (15-Minute Macro Context)
  Rule 2: Time and Price (Strict Killzones in EST)
  Rule 3: Liquidation (Asian / London / Premarket Extreme Sweeps)
  Rule 4: Reversal Confirmation (1-Minute Micro-CHoCH Body Close)
  Rule 5: Entry Model (5-Minute POI: Order Block, FVG, IFVG & Confluence)
  Risk: Strict 1:5R Minimum Veto + Hybrid Partial Scaling (40% @ 3R + BE, 60% @ M15 Target)
"""

from __future__ import annotations

import os
import sys
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from strategies.strat_baseplate import StrategyBaseplate
except ImportError:
    from strat_baseplate import StrategyBaseplate
from smc_bias_model.bias_structure import M15BoxStructureDetector, BiasDirection, ExternalBoxRange
from smc_bias_model.session_killzones import SessionKillzoneManager, LiquidationStatus
from smc_bias_model.m1_choch import M1CHoCHDetector, M1CHoCHConfirmation
from smc_bias_model.m5_poi import M5POISelector, PointOfInterest, POIType
from smc_bias_model.bias_order_manager import BiasOrderManager, BiasTradeRecord, OrderLifecycleStatus


class SMCIntradayBiasStrategy(StrategyBaseplate):
    """
    Algorithmic 5-Rule SMC Intraday Bias Strategy.
    Orchestrates M15 Macro Context, Session Killzones, Liquidation Sweeps,
    M1 Micro-CHoCH Confirmation, and M5 POI Execution with 1:5R+ Asymmetry.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "TradingStyle": "intraday",
        "RiskPct": 0.01,                      # 1% portfolio risk per trade
        "MinRiskReward": 5.0,                 # Strict Asymmetric 1:5R Veto Rule
        "TP1_Pct": 0.40,                      # 40% TP1 @ 3.0R (+ Breakeven stop)
        "TP2_Pct": 0.60,                      # 60% TP2 @ M15 External Swing (>= 5.0R)
        "TickSize": 0.01,                     # Minimum price tick
        "Plot": True,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "1M"  # 1-minute candle iteration for micro-CHoCH precision

        # Managers & Detectors
        self.box_detector = M15BoxStructureDetector(left_bars=3, right_bars=3)
        self.session_mgr = SessionKillzoneManager(allow_ny_extension=True)
        self.choch_detector = M1CHoCHDetector(left_bars=2, right_bars=2)
        self.poi_selector = M5POISelector(min_displacement_ratio=1.0)
        self.order_mgr = BiasOrderManager(
            min_rr=self.parameters.get("MinRiskReward", 5.0),
            risk_pct=self.parameters.get("RiskPct", 0.01),
            tp1_pct=self.parameters.get("TP1_Pct", 0.40),
            tp2_pct=self.parameters.get("TP2_Pct", 0.60),
            tick_size=self.parameters.get("TickSize", 0.01),
        )

        # State tracking
        self.state: str = "IDLE"  # "IDLE", "SWEPT", "WAITING_FOR_FILL", "IN_POSITION"
        self.active_trade: Optional[BiasTradeRecord] = None
        self.pending_setup: Optional[Dict[str, Any]] = None
        self.current_box: Optional[ExternalBoxRange] = None
        self.last_liquidation: Optional[LiquidationStatus] = None
        self.trade_records: List[Dict[str, Any]] = []

        ticker_sym = self.parameters["Ticker"].symbol if hasattr(self.parameters["Ticker"], "symbol") else str(self.parameters["Ticker"])
        self.log_message(f"[{ticker_sym}] 5-Rule SMC Intraday Bias Strategy initialized. Min R:R = 1:{self.parameters['MinRiskReward']:.1f}R.")

    def on_trading_iteration(self):
        ticker = self.parameters["Ticker"]

        # Fetch historical 1-minute bars (fetch 600 bars for M15 & M5 resampling + session context)
        try:
            bars_raw = self.get_historical_prices(ticker, 600, "minute").df
        except Exception as e:
            self.log_message(f"Could not retrieve candles: {e}")
            return

        if bars_raw.empty or len(bars_raw) < 50:
            return

        current_bar = bars_raw.iloc[-1]
        current_time = current_bar.name if hasattr(current_bar, "name") else datetime.now()
        current_price = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        # 1. Manage Active Trade Tranches (Stop loss, TP1 @ 3.0R + Breakeven, TP2 @ M15 Target)
        if self.active_trade and not self.active_trade.is_completed:
            events = self.active_trade.check_exits(current_bar, current_time)
            for ev in events:
                self.log_message(f"[{ticker.symbol}] BIAS TRADE EVENT: {ev}")
            if self.active_trade.is_completed:
                self._record_completed_trade(self.active_trade)
                self.active_trade = None
                self.state = "IDLE"
            return

        # 2. Check Pending Limit Order fills
        if self.state == "WAITING_FOR_FILL" and self.pending_setup:
            side = self.pending_setup["side"]
            limit_p = self.pending_setup["limit_price"]
            sl_p = self.pending_setup["stop_loss"]
            tp1_p = self.pending_setup["tp1_price"]

            # Invalidation Check: Did price breach the structural stop before limit fill?
            if (side == "BUY" and current_low <= sl_p) or (side == "SELL" and current_high >= sl_p):
                self.log_message(f"[{ticker.symbol}] Invalidation extreme breached before limit fill. Setup cancelled.")
                self.pending_setup = None
                self.state = "IDLE"
                return

            # Missed Pullback Check: Did price reach TP1 before fill?
            if (side == "BUY" and current_high >= tp1_p) or (side == "SELL" and current_low <= tp1_p):
                self.log_message(f"[{ticker.symbol}] Price reached TP1 before fill. Pullback missed; setup cancelled.")
                self.pending_setup = None
                self.state = "IDLE"
                return

            # Fill Check
            filled = (current_low <= limit_p) if side == "BUY" else (current_high >= limit_p)
            if filled:
                equity = self.get_portfolio_value()
                self.active_trade = self.order_mgr.create_trade(
                    ticker=ticker.symbol,
                    side=side,
                    entry_price=limit_p,
                    structural_extreme_price=self.pending_setup["structural_extreme"],
                    m15_target_price=self.pending_setup["target_price"],
                    equity=equity,
                    entry_time=current_time,
                )
                if self.active_trade:
                    self.log_message(f"[{ticker.symbol}] LIMIT ORDER FILLED: {side} {self.active_trade.total_quantity} shares @ ${limit_p:.2f}.")
                    self.state = "IN_POSITION"
                else:
                    self.state = "IDLE"
                self.pending_setup = None
                return

        # 3. Rule 2: Session Killzone Gate (London 02:00-05:00 EST, NY 07:00-11:30 EST)
        in_killzone, session_name = self.session_mgr.is_in_killzone(current_time)
        if not in_killzone:
            self.state = "IDLE"
            return

        # 4. Rule 1: Directional Bias via The Box Method (15-Minute Macro Context)
        df_m15 = bars_raw.resample("15min", closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        if len(df_m15) > 1:
            df_m15 = df_m15.iloc[:-1]  # Exclude current forming 15M bar

        if len(df_m15) < 10:
            return

        bias, box_range = self.box_detector.analyze_external_structure(df_m15)
        if bias == BiasDirection.NEUTRAL or box_range is None:
            return
        self.current_box = box_range

        bias_str = "BULLISH" if bias == BiasDirection.BULLISH else "BEARISH"
        side = "BUY" if bias == BiasDirection.BULLISH else "SELL"
        m15_target = box_range.target_price

        # 5. Rule 3: Liquidation (Session Benchmark Sweeps)
        benchmarks = self.session_mgr.extract_prior_session_benchmarks(bars_raw, current_time)
        sweep_status = self.session_mgr.check_liquidation_sweep(
            cur_high=current_high,
            cur_low=current_low,
            bias=bias_str,
            active_killzone=session_name,
            benchmarks=benchmarks,
            cur_time=current_time,
        )

        if not sweep_status.swept:
            return

        self.last_liquidation = sweep_status

        # 6. Rule 4: Reversal Confirmation (1-Minute Micro-CHoCH)
        df_m1_recent = bars_raw.iloc[-40:].copy()
        m1_choch = self.choch_detector.detect_m1_choch(
            df_m1=df_m1_recent,
            bias=bias_str,
            sweep_time=sweep_status.sweep_time,
            sweep_level=sweep_status.benchmark_level,
        )

        if not m1_choch:
            return

        # 7. Rule 5: Entry Model (5-Minute POI Selection & Confluence Stacking)
        df_m5 = bars_raw.resample("5min", closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        if len(df_m5) > 1:
            df_m5 = df_m5.iloc[:-1]

        best_poi = self.poi_selector.select_best_poi_with_confluence(df_m5, bias=bias_str)
        if not best_poi:
            return

        entry_limit_price = best_poi.proximate_price
        structural_extreme = m1_choch.structural_extreme_price
        tick_size = self.parameters.get("TickSize", 0.01)
        stop_loss_price = (structural_extreme - tick_size) if side == "BUY" else (structural_extreme + tick_size)

        # 8. Strict Asymmetric 1:5R Veto Gate
        min_rr = self.parameters.get("MinRiskReward", 5.0)
        approved, rr_ratio, veto_msg = self.order_mgr.evaluate_5r_veto(
            entry_price=entry_limit_price,
            stop_loss=stop_loss_price,
            m15_target=m15_target,
            side=side,
        )

        if not approved:
            self.log_message(f"[{ticker.symbol}] {veto_msg}")
            return

        # Setup is fully qualified! Place pending limit order
        self.pending_setup = {
            "side": side,
            "limit_price": entry_limit_price,
            "stop_loss": stop_loss_price,
            "structural_extreme": structural_extreme,
            "target_price": m15_target,
            "tp1_price": (entry_limit_price + 3.0 * abs(entry_limit_price - stop_loss_price)) if side == "BUY" else (entry_limit_price - 3.0 * abs(entry_limit_price - stop_loss_price)),
            "rr_ratio": rr_ratio,
            "poi": best_poi,
            "choch": m1_choch,
        }
        self.state = "WAITING_FOR_FILL"
        self.log_message(
            f"[{ticker.symbol}] 5-RULE SETUP QUALIFIED ({side}): Limit=${entry_limit_price:.2f}, "
            f"SL=${stop_loss_price:.2f}, TP1(3R)=${self.pending_setup['tp1_price']:.2f}, "
            f"TP2(M15)=${m15_target:.2f} ({rr_ratio:.2f}R). POI: {best_poi.description}."
        )

    def _record_completed_trade(self, trade: BiasTradeRecord) -> None:
        """Stores standardized completed trade records."""
        rec = {
            "trade_id": trade.trade_id,
            "ticker": trade.ticker,
            "side": trade.side,
            "entry_time": str(trade.entry_time),
            "exit_time": str(trade.completion_time),
            "entry_price": trade.entry_price,
            "initial_sl": trade.initial_stop_loss,
            "total_shares": trade.total_quantity,
            "pnl": trade.total_realized_pnl,
            "rr_ratio": trade.rr_ratio,
            "tp1_hit": trade.tp1_hit,
            "tp2_hit": trade.tp2_hit,
            "tranches": {
                tid: {
                    "shares": t.quantity,
                    "target_price": t.target_price,
                    "exit_price": t.exit_price,
                    "status": t.status,
                    "pnl": t.pnl,
                    "target_r": t.target_r,
                }
                for tid, t in trade.tranches.items()
            }
        }
        self.trade_records.append(rec)
