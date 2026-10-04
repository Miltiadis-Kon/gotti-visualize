"""
Smart Money Concepts (SMC) Strategy (smc_strategy.py)
======================================================
Comprehensive quantitative implementation of the 8-Step Macro-to-Micro SMC Manual:
  1. 1-Hour HTF Market Structure (Higher Highs / Lower Lows)
  2. External Liquidity Sweeps (Wick fakeouts vs Body breaks)
  3. Direction confirmation with CHoCH (Candle body close filter)
  4. Premium and Discount equilibrium marking (50% Fibonacci OTE rule)
  5. Qualified Fair Value Gaps (Displacement + Unmitigated lifecycle)
  6. Pre-planned Liquidity Targets with strict 3R Minimum Veto Rule
  7. 5-Minute LTF micro-CHoCH confirmation inside 1H POI tap
  8. Execution via 3-Tier Multi-Tranche Management (TP1+BE, TP2+Trail, TP3 runner)
"""

from __future__ import annotations

import os
import sys
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import pandas_ta as ta

from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from strategies.strat_baseplate import StrategyBaseplate
except ImportError:
    from strat_baseplate import StrategyBaseplate
from smc.smc_structure import (
    FractalPivotDetector,
    MarketStructure,
    SwingPoint,
    SwingType,
    TrendBias,
    detect_structure_shifts,
)
from smc.smc_fvg import (
    FairValueGap,
    FVGType,
    detect_fvgs,
    filter_fvgs_by_equilibrium,
    update_fvg_mitigation,
)
from smc.smc_liquidity import (
    LiquidityPool,
    LiquidityType,
    find_liquidity_pools,
    evaluate_3r_veto,
    extract_pdh_pdl,
)
from smc.smc_tranche_manager import (
    SMCTrancheManager,
    SMCTradeLifecycle,
    TrancheStatus,
)


class SmartMoneyConceptsStrategy(StrategyBaseplate):
    """
    Algorithmic Smart Money Concepts Strategy for US Equities.
    Integrates 1H Macro Analysis with 5M Micro Execution.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "TradingStyle": "swing_trading",      # Allows holding overnight
        "RiskPct": 0.02,                      # 2% portfolio risk budget per trade
        "MinRiskReward": 3.0,                 # Strict 3R Minimum Veto Rule
        "TP1_Pct": 0.40,                      # 40% TP1 (Internal Liquidity)
        "TP2_Pct": 0.40,                      # 40% TP2 (Primary Structural Target >= 3R)
        "TP3_Pct": 0.20,                      # 20% TP3 (External Liquidity Runner)
        "HTF_Pivots_Lookback": 3,             # 1H fractal pivot lookback
        "LTF_Pivots_Lookback": 2,             # 5M micro-fractal lookback
        "MinDisplacementATR": 1.2,            # FVG displacement filter
        "MinBodyRatio": 0.60,                 # FVG candle body ratio
        "AllowOvernight": True,               # User confirmed: hold overnight
        "TickSize": 0.01,                     # Equity tick size
        "Plot": True,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"  # 5-minute candle iteration

        # Tranche and lifecycle manager
        self.tranche_mgr = SMCTrancheManager(
            tp1_pct=self.parameters.get("TP1_Pct", 0.40),
            tp2_pct=self.parameters.get("TP2_Pct", 0.40),
            tp3_pct=self.parameters.get("TP3_Pct", 0.20),
        )

        # State tracking
        self.state: str = "IDLE"  # "IDLE", "HTF_POI_ACTIVE", "WAITING_FOR_FILL", "IN_POSITION"
        self.active_trade: Optional[SMCTradeLifecycle] = None
        self.pending_setup: Optional[Dict[str, Any]] = None
        self.current_bar_idx: int = 0
        self.poi_expiry_idx: int = 0
        self.active_poi: Optional[FairValueGap] = None

        # Data & POI caching
        self.htf_trend: TrendBias = TrendBias.NEUTRAL
        self.known_1h_fvgs: Dict[str, FairValueGap] = {}
        self.htf_pivots: List[SwingPoint] = []
        
        # Standardized trade records for reporting & Plotly visualization
        self.trade_records: List[Dict[str, Any]] = []

        ticker_sym = self.parameters["Ticker"].symbol if hasattr(self.parameters["Ticker"], "symbol") else str(self.parameters["Ticker"])
        self.log_message(f"[{ticker_sym}] SMC Strategy initialized. Min R:R = {self.parameters['MinRiskReward']:.1f}R, Overnight = {self.parameters['AllowOvernight']}.")

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-Timeframe Resampling & Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def _resample_to_1h(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples 5-minute candles to 1-Hour candles strictly without lookahead bias.
        """
        if df_5m.empty:
            return pd.DataFrame()

        df_1h = df_5m.resample("1h", closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        if len(df_1h) > 1:
            df_1h = df_1h.iloc[:-1]

        return df_1h

    def _analyze_htf(self, df_1h: pd.DataFrame) -> None:
        """
        Executes Steps 1 to 5 on the 1-Hour Higher Time Frame:
          1. Market Structure & Trend Bias
          2. External Highs/Lows
          3. CHoCH vs Wick Sweeps
          4. Premium/Discount Range
          5. Qualified Unmitigated FVGs
        """
        if len(df_1h) < 10:
            return

        lookback = self.parameters.get("HTF_Pivots_Lookback", 3)
        ms = MarketStructure(left_bars=lookback, right_bars=lookback)
        self.htf_trend, _ = ms.analyze(df_1h)
        self.htf_pivots = ms.pivots

        # Detect 1H FVGs
        all_1h_fvgs = detect_fvgs(
            df_1h,
            min_displacement_atr=self.parameters.get("MinDisplacementATR", 0.8),
            min_body_ratio=self.parameters.get("MinBodyRatio", 0.50)
        )

        for f in all_1h_fvgs:
            fvg_key = f"{f.timestamp}_{f.fvg_type.value}_{f.bottom:.2f}"
            if fvg_key not in self.known_1h_fvgs:
                f.mitigated = False  # fresh FVG awaiting first 5M touch
                self.known_1h_fvgs[fvg_key] = f

    # ──────────────────────────────────────────────────────────────────────────
    # Core Strategy Iteration
    # ──────────────────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        ticker = self.parameters["Ticker"]
        self.current_bar_idx += 1
        
        # 1. Fetch historical 5-minute bars (fetch 1000 minutes for robust 1H resampling)
        try:
            bars_5m = self.get_historical_prices(ticker, 1000, "minute").df
        except Exception as e:
            self.log_message(f"Could not retrieve 5m candles: {e}")
            return

        if bars_5m.empty or len(bars_5m) < 30:
            return

        current_bar = bars_5m.iloc[-1]
        current_time = current_bar.name if hasattr(current_bar, "name") else datetime.now()
        current_price = float(current_bar["close"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])

        # 2. Manage Active Trade Tranches (Exits, Breakeven, Trailing)
        if self.active_trade and not self.active_trade.is_completed:
            events = self.active_trade.check_exits(current_bar, current_time)
            for ev in events:
                self.log_message(f"[{ticker.symbol}] SMC EXECUTION EVENT: {ev}")
                if ev.get("event") == "TP1_SCALED_OUT":
                    self.log_message(f"[{ticker.symbol}] TP1 HIT @ ${ev['exit_price']:.2f}. STOP LOSS MOVED TO BREAKEVEN @ ${self.active_trade.current_stop_loss:.2f}.")
                elif ev.get("event") == "TP2_SCALED_OUT":
                    self.log_message(f"[{ticker.symbol}] TP2 HIT @ ${ev['exit_price']:.2f} ({ev['r_multiple']:.2f}R). ACTIVATING STRUCTURAL TRAIL FOR RUNNER.")
                elif ev.get("event") in ("STOP_LOSS_HIT", "TP3_RUNNER_HIT"):
                    self.log_message(f"[{ticker.symbol}] TRADE COMPLETED: {ev['event']} @ ${ev['exit_price']:.2f}. Total PnL: ${self.active_trade.total_realized_pnl:+.2f}")

            # If trade completed, record trade details and reset state
            if self.active_trade.is_completed:
                self._record_completed_trade(self.active_trade)
                self.active_trade = None
                self.state = "IDLE"
                return

            # Update trailing stop behind recent 5M structural swings if in runner phase
            if self.active_trade.is_trailing:
                ltf_detector = FractalPivotDetector(left_bars=2, right_bars=2)
                pivots_5m = ltf_detector.find_pivots(bars_5m.iloc[-40:])
                if self.active_trade.side == "BUY":
                    low_pivots = [p.price for p in pivots_5m if p.swing_type == SwingType.LOW and p.price > self.active_trade.entry_price]
                    if low_pivots:
                        recent_trail_sl = max(low_pivots)
                        if self.active_trade.update_trailing_stop(recent_trail_sl):
                            self.log_message(f"[{ticker.symbol}] Trailing SL updated to ${recent_trail_sl:.2f}")
                else:
                    high_pivots = [p.price for p in pivots_5m if p.swing_type == SwingType.HIGH and p.price < self.active_trade.entry_price]
                    if high_pivots:
                        recent_trail_sl = min(high_pivots)
                        if self.active_trade.update_trailing_stop(recent_trail_sl):
                            self.log_message(f"[{ticker.symbol}] Trailing SL updated to ${recent_trail_sl:.2f}")
            return

        # 3. Check Pending Limit Order fills
        if self.state == "WAITING_FOR_FILL" and self.pending_setup:
            side = self.pending_setup["side"]
            limit_price = self.pending_setup["entry_price"]
            sl_price = self.pending_setup["stop_loss"]
            tp1_price = self.pending_setup["tp1"]

            # Invalidation Check: Did price breach the structural stop loss before filling?
            if (side == "BUY" and current_low <= sl_price) or (side == "SELL" and current_high >= sl_price):
                self.log_message(f"[{ticker.symbol}] Setup invalidated: Structural extreme breached prior to fill. Cancelled.")
                self.pending_setup = None
                self.state = "IDLE"
                return

            # Missed Pullback Check: Did price reach TP1 before filling?
            if (side == "BUY" and current_high >= tp1_price) or (side == "SELL" and current_low <= tp1_price):
                self.log_message(f"[{ticker.symbol}] Missed pullback: Price reached TP1 before limit fill. Cancelled.")
                self.pending_setup = None
                self.state = "IDLE"
                return

            # Fill Check
            filled = (current_low <= limit_price) if side == "BUY" else (current_high >= limit_price)
            if filled:
                fill_price = limit_price
                qty = self.pending_setup["quantity"]
                self.log_message(f"[{ticker.symbol}] LIMIT ORDER FILLED: {side} {qty} shares @ ${fill_price:.2f}.")

                self.active_trade = self.tranche_mgr.create_trade(
                    ticker=ticker.symbol,
                    side=side,
                    entry_price=fill_price,
                    stop_loss=sl_price,
                    tp1_price=self.pending_setup["tp1"],
                    tp2_price=self.pending_setup["tp2"],
                    tp3_price=self.pending_setup["tp3"],
                    total_quantity=qty,
                    entry_time=current_time,
                )
                self.state = "IN_POSITION"
                self.pending_setup = None
                self.active_poi = None
                return

        # 4. If IDLE, Scan HTF (1-Hour)
        df_1h = self._resample_to_1h(bars_5m)
        self._analyze_htf(df_1h)

        if self.current_bar_idx % 200 == 0:
            print(f"Bar {self.current_bar_idx} at {current_time}: 1H FVGs={len(self.known_1h_fvgs)}, Active POI={self.active_poi is not None}, State={self.state}")

        # Check if 5M price taps an active 1H FVG (POI)
        if self.active_poi is None or self.current_bar_idx > self.poi_expiry_idx:
            for fvg in self.known_1h_fvgs.values():
                if fvg.mitigated:
                    continue
                if fvg.fvg_type == FVGType.BULLISH and current_low <= fvg.top and current_price >= fvg.bottom:
                    fvg.mitigated = True
                    self.active_poi = fvg
                    self.poi_expiry_idx = self.current_bar_idx + 24  # Stays active for up to 2 hours
                    break
                elif fvg.fvg_type == FVGType.BEARISH and current_high >= fvg.bottom and current_price <= fvg.top:
                    fvg.mitigated = True
                    self.active_poi = fvg
                    self.poi_expiry_idx = self.current_bar_idx + 24
                    break

        if self.active_poi is None:
            return

        # 5. Step 7: Lower Timeframe (5M) Confirmation inside Active POI
        # Look for a micro-CHoCH (solid body close) on the 5-minute chart
        ltf_window = bars_5m.iloc[-50:].copy()
        ltf_lookback = self.parameters.get("LTF_Pivots_Lookback", 2)
        ms_5m = MarketStructure(left_bars=ltf_lookback, right_bars=ltf_lookback)
        ltf_trend, ltf_events = ms_5m.analyze(ltf_window)

        expected_event = "CHOCH_BULLISH" if self.active_poi.fvg_type == FVGType.BULLISH else "CHOCH_BEARISH"
        confirmed_choch = next((e for e in reversed(ltf_events) if e.event_type == expected_event), None)

        if not confirmed_choch:
            return

        # Only process fresh CHoCH (within last 3 bars)
        if (len(ltf_window) - 1) - confirmed_choch.bar_idx > 3:
            return

        origin_pivot = confirmed_choch.origin_swing
        if not origin_pivot:
            return

        side = "BUY" if self.active_poi.fvg_type == FVGType.BULLISH else "SELL"
        sl_price = origin_pivot.price

        # 6. Step 4 & 5 (LTF): Detect 5M unmitigated FVG in impulse leg OR OTE 61.8% Retracement
        fvgs_5m = detect_fvgs(ltf_window, min_displacement_atr=0.5, min_body_ratio=0.45)
        leg_fvgs = [
            f for f in fvgs_5m
            if not f.mitigated
            and f.bar_idx >= max(0, origin_pivot.bar_idx - 2)
            and ((side == "BUY" and f.fvg_type == FVGType.BULLISH) or (side == "SELL" and f.fvg_type == FVGType.BEARISH))
        ]

        if leg_fvgs:
            entry_fvg = leg_fvgs[-1]
            entry_price = entry_fvg.top if side == "BUY" else entry_fvg.bottom
        else:
            # OTE (Optimal Trade Entry) 61.8% Fibonacci Retracement of the confirmed CHoCH leg
            fib_diff = abs(confirmed_choch.trigger_price - origin_pivot.price)
            if side == "BUY":
                entry_price = confirmed_choch.trigger_price - 0.618 * fib_diff
            else:
                entry_price = confirmed_choch.trigger_price + 0.618 * fib_diff

        # Stop loss with 1 tick buffer safely beyond the structural extreme wick
        tick_size = self.parameters.get("TickSize", 0.01)
        if side == "BUY":
            stop_loss = sl_price - tick_size
            risk_per_share = entry_price - stop_loss
        else:
            stop_loss = sl_price + tick_size
            risk_per_share = stop_loss - entry_price

        if risk_per_share <= 0:
            return

        # 7. Step 6: Liquidity Pool Targets & 3R Minimum Veto Rule
        pdh, pdl = extract_pdh_pdl(bars_5m)
        opposing_fvgs_5m = [f for f in fvgs_5m if (f.fvg_type == FVGType.BEARISH if side == "BUY" else f.fvg_type == FVGType.BULLISH)]
        
        tp1_pool, tp2_pool, tp3_pool = find_liquidity_pools(
            current_price=entry_price,
            side=side,
            pivots=ms_5m.pivots,
            opposing_fvgs=opposing_fvgs_5m,
            pdh=pdh,
            pdl=pdl,
        )

        if not tp2_pool:
            return

        # ENFORCE 3R MINIMUM VETO RULE
        min_rr = self.parameters.get("MinRiskReward", 3.0)
        approved, rr_val, veto_msg = evaluate_3r_veto(
            entry_price=entry_price,
            stop_loss=stop_loss,
            primary_target=tp2_pool.price,
            side=side,
            min_rr=min_rr
        )

        if not approved:
            self.log_message(f"[{ticker.symbol}] {veto_msg}")
            return

        # 8. Position Sizing & Place Limit Order
        portfolio_val = self.get_portfolio_value()
        risk_pct = self.parameters.get("RiskPct", 0.02)
        total_risk_budget = portfolio_val * risk_pct
        shares = max(3, int(math.floor(total_risk_budget / risk_per_share)))

        tp1_val = tp1_pool.price if tp1_pool else (entry_price + 1.5 * risk_per_share if side == "BUY" else entry_price - 1.5 * risk_per_share)
        tp2_val = tp2_pool.price
        tp3_val = tp3_pool.price if tp3_pool else (entry_price + 4.0 * risk_per_share if side == "BUY" else entry_price - 4.0 * risk_per_share)

        self.pending_setup = {
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "tp1": tp1_val,
            "tp2": tp2_val,
            "tp3": tp3_val,
            "quantity": shares,
            "rr_ratio": rr_val,
            "poi_fvg": self.active_poi,
            "choch_bar": confirmed_choch.bar_idx,
        }
        self.state = "WAITING_FOR_FILL"

        print(f"[{ticker.symbol}] >>> SETUP QUALIFIED ({side}) at {current_time}: Entry=${entry_price:.2f}, SL=${stop_loss:.2f}, TP2=${tp2_val:.2f} ({rr_val:.2f}R), Qty={shares}")
        self.log_message(
            f"[{ticker.symbol}] SETUP QUALIFIED ({side}): Entry Limit=${entry_price:.2f}, "
            f"SL=${stop_loss:.2f} (Risk=${risk_per_share:.2f}), TP1=${tp1_val:.2f}, "
            f"TP2=${tp2_val:.2f} ({rr_val:.2f}R), TP3=${tp3_val:.2f}. Shares={shares}."
        )

    def _record_completed_trade(self, trade: SMCTradeLifecycle) -> None:
        """Stores standardized trade records for the dashboard and Plotly charting."""
        record = {
            "trade_id": trade.trade_id,
            "ticker": trade.ticker,
            "side": trade.side,
            "entry_time": str(trade.entry_time),
            "exit_time": str(trade.completion_time),
            "entry_price": trade.entry_price,
            "initial_sl": trade.initial_stop_loss,
            "total_shares": trade.total_quantity,
            "pnl": trade.total_realized_pnl,
            "tp1_hit": trade.tp1_hit,
            "tp2_hit": trade.tp2_hit,
            "tp3_hit": trade.tp3_hit,
            "tranches": {
                tid: {
                    "shares": t.quantity,
                    "target_price": t.target_price,
                    "exit_price": t.exit_price,
                    "status": t.status.value,
                    "pnl": t.pnl,
                    "r_multiple": t.r_multiple,
                }
                for tid, t in trade.tranches.items()
            }
        }
        self.trade_records.append(record)
