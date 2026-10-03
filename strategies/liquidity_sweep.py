"""
Institutional Liquidity Sweep Strategy (strategies/liquidity_sweep.py)
======================================================================
Intraday 5-minute liquidity trap & market structure shift trading engine.

Key Execution Architecture:
  1. The Map:
     - Previous Day High (PDH) & Previous Day Low (PDL)
     - Session High & Session Low (Asian / Overnight pre-market)
     - Clean Swing Pivots (filtered via Isolation, Kaufman Efficiency Ratio, and ATR Velocity)
     - Intraday Session VWAP (dynamic fair-value anchor)

  2. Time & Volume Gates (US Eastern Time):
     - 03:00 - 05:00 EST: London Open (targets Asian Session extremes)
     - 09:30 - 11:30 EST: NYSE Open (targets Overnight or PDH/PDL)
     - 11:30 - 14:00 EST: Dead Zone (all trading halted; pending orders purged)
     - 14:00 - 15:45 EST: EOD Cleanup (targets untouched session extremes)
     - Volume Spike: Sweep bar must exceed its 20-period moving average

  3. 5-Stage State Machine:
     - IDLE -> SWEPT: Wick pierces clean level + volume spike (branched into REVERSAL or CONTINUATION)
     - SWEPT -> RECLAIMED: Close back inside level within <= 3 bars (avoids price acceptance)
     - RECLAIMED -> CONFIRMED: Minor structure shift (breaks pivot) OR aggressive VWAP reclaim
     - CONFIRMED -> WAITING_FOR_FILL: Limit order placed at broken pivot / VWAP retest
     - WAITING_FOR_FILL -> IN_POSITION: Filled on pullback
     - Missed Pullback Rule: Cancel immediately if price reaches TP before filling entry limit
     - Invalidation: Hard cancel if sweep extreme is broken prior to fill

  4. Risk Management & Stop Loss Buffering:
     - Designated stop loss at the sweep extreme wick (+/- 1 tick).
     - ATR Buffer: Added ATR levels below (for longs) or above (for shorts) the sweep wick
       to prevent stop-outs from noise, spread widening, or secondary micro-wicks.
     - Hard bracket stop: Executed exchange-side on tick pierce.
     - Take Profit: The very next immediate clean swing pivot (or opposing session extreme).
"""

from __future__ import annotations

import os
import sys
import datetime
from typing import Optional, Dict, Any, List, Tuple
from math import floor
import pandas as pd
import numpy as np
import pandas_ta as ta

from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_baseplate import StrategyBaseplate


class LiquiditySweepStrategy(StrategyBaseplate):
    """
    Algorithmic Institutional Liquidity Sweep Strategy.
    Subclasses StrategyBaseplate for unified ATR risk framework, order tracking, and plotting.
    """

    parameters: Dict[str, Any] = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "TradingStyle": "day_trading",
        "RiskPct": 0.02,                      # 2% portfolio risk budget per trade
        "MaxReclaimCandles": 3,               # Max candles allowed outside level before acceptance abort
        "MinVwapSlope": 0.05,                 # Disqualify flat VWAP drift
        "VolumeMAPeriod": 20,                 # Period for volume spike evaluation
        "IsolationRadius": 15,                # Radius in bars to ensure pivot prominence
        "MinEfficiencyRatio": 0.60,           # Kaufman ER threshold for straight moves (>0.60)
        "MinVelocityATRMult": 1.5,            # Bar velocity vs ATR threshold
        "ATR_Length": 14,                     # ATR period
        "ATR_Stop_Buffer_Multiplier": 0.5,     # ATR levels added below/above designated stop loss
        "TickSize": 0.01,                     # Minimum tick offset
        "Plot": True,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"  # 5-Minute intraday candles

        # State Machine Tracking
        self.state: str = "IDLE"  # "IDLE", "SWEPT_LONG", "SWEPT_SHORT", "RECLAIMED_LONG", "RECLAIMED_SHORT", "WAITING_FOR_FILL", "IN_POSITION"
        self.trade_model: Optional[str] = None  # "REVERSAL" or "CONTINUATION"
        self.sweep_extreme: Optional[float] = None
        self.sweep_bar_idx: Optional[int] = None
        self.target_level: Optional[float] = None
        self.target_pivot: Optional[float] = None
        
        # Order Tracking
        self.pending_side: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.active_order = None

        self.current_session_date: Optional[datetime.date] = None

        ticker_sym = self.parameters["Ticker"].symbol if hasattr(self.parameters["Ticker"], "symbol") else str(self.parameters["Ticker"])
        self.log_message(f"[{ticker_sym}] LiquiditySweepStrategy initialized. ATR Buffer={self.parameters['ATR_Stop_Buffer_Multiplier']}x ATR.")

    # ──────────────────────────────────────────────────────────────────────────
    # Intraday Indicators & VWAP Calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_intraday_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume MA, ATR, Session VWAP, and VWAP slope."""
        data = df.copy()

        # Volume Moving Average
        vol_period = int(self.parameters.get("VolumeMAPeriod", 20))
        data["vol_ma"] = data["volume"].rolling(vol_period).mean()

        # ATR Calculation
        atr_len = int(self.parameters.get("ATR_Length", 14))
        data.ta.atr(length=atr_len, append=True)
        atr_col = [c for c in data.columns if c.startswith("ATRr_")]
        data["atr"] = data[atr_col[0]] if atr_col else 1.0

        # Session VWAP (resetting daily)
        data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        data["pv"] = data["typical_price"] * data["volume"]

        if isinstance(data.index, pd.DatetimeIndex):
            session_dates = data.index.date
        elif "datetime" in data.columns:
            session_dates = pd.to_datetime(data["datetime"]).dt.date
        else:
            session_dates = np.zeros(len(data))

        data["session_date"] = session_dates
        data["cum_pv"] = data.groupby("session_date")["pv"].cumsum()
        data["cum_vol"] = data.groupby("session_date")["volume"].cumsum()
        data["vwap"] = data["cum_pv"] / data["cum_vol"].replace(0, np.nan)
        data["vwap"] = data["vwap"].ffill().bfill()

        # VWAP 10-period Slope (Abs net displacement)
        data["vwap_slope"] = (data["vwap"] - data["vwap"].shift(10)).abs()

        return data

    # ──────────────────────────────────────────────────────────────────────────
    # Clean Level Mathematical Filters
    # ──────────────────────────────────────────────────────────────────────────

    def _is_clean_level(self, df: pd.DataFrame, pivot_idx: int, level_price: float, is_high: bool) -> bool:
        """
        Verify if a level is clean and attracts institutional liquidity:
          1. Isolation: Stands alone within +/- 15 bars, no other wicks within 0.1%.
          2. Density (Kaufman Efficiency Ratio): Net change / sum of noise >= 0.60.
          3. Velocity (ATR Expansion): Bar velocity >= 1.5 * ATR(14).
        """
        radius = int(self.parameters.get("IsolationRadius", 15))
        min_er = float(self.parameters.get("MinEfficiencyRatio", 0.60))
        vel_mult = float(self.parameters.get("MinVelocityATRMult", 1.5))

        if pivot_idx < 10 or pivot_idx >= len(df) - 2:
            return True  # Fallback for boundary conditions

        # 1. Isolation
        price_buffer = level_price * 0.001
        start_idx = max(0, pivot_idx - radius)
        end_idx = min(len(df), pivot_idx + radius)

        for i in range(start_idx, end_idx):
            if i == pivot_idx:
                continue
            if is_high and df["high"].iloc[i] > (level_price - price_buffer):
                return False
            if not is_high and df["low"].iloc[i] < (level_price + price_buffer):
                return False

        # 2. Kaufman Efficiency Ratio over leading 10 bars
        lookback = min(10, pivot_idx)
        origin_idx = pivot_idx - lookback
        net_change = abs(df["close"].iloc[pivot_idx] - df["close"].iloc[origin_idx])
        sum_noise = (df["close"].iloc[origin_idx:pivot_idx + 1].diff().abs()).sum()

        if sum_noise > 0:
            er = net_change / sum_noise
            if er < min_er:
                return False

        # 3. Velocity ATR Expansion
        current_atr = df["atr"].iloc[pivot_idx]
        velocity_per_bar = net_change / max(1, lookback)
        if velocity_per_bar < (current_atr * vel_mult):
            return False

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Session Targets & Active Time Gates
    # ──────────────────────────────────────────────────────────────────────────

    def _get_active_session_targets(
        self, df: pd.DataFrame, current_dt: datetime.datetime
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Evaluate time gates and map active targets:
          - 03:00 - 05:00 EST: London Open (targets Asian Session 18:00 - 03:00)
          - 09:30 - 11:30 EST: NYSE Open (targets Overnight 00:00 - 09:30 or PDH/PDL)
          - 11:30 - 14:00 EST: Dead Zone (Halt & reset)
          - 14:00 - 15:45 EST: EOD Cleanup (targets untouched session extremes)
        """
        current_time = current_dt.time()

        # Dead Zone: London close + NY lunch (11:30 - 14:00 EST)
        if datetime.time(11, 30) <= current_time < datetime.time(14, 0):
            return None, None

        # Check Active Windows
        is_london = datetime.time(3, 0) <= current_time <= datetime.time(5, 0)
        is_nyse = datetime.time(9, 30) <= current_time <= datetime.time(11, 30)
        is_cleanup = datetime.time(14, 0) <= current_time <= datetime.time(15, 45)

        if not (is_london or is_nyse or is_cleanup):
            return None, None

        today = current_dt.date()
        today_bars = df[df["session_date"] == today]
        prior_bars = df[df["session_date"] < today]

        # Previous Day High / Low
        pdh = float(prior_bars["high"].iloc[-78:].max()) if len(prior_bars) >= 10 else float(df["high"].max())
        pdl = float(prior_bars["low"].iloc[-78:].min()) if len(prior_bars) >= 10 else float(df["low"].min())

        if is_nyse:
            # Overnight session (00:00 to 09:30)
            overnight_bars = today_bars[today_bars.index.time < datetime.time(9, 30)]
            if not overnight_bars.empty:
                target_high = max(float(overnight_bars["high"].max()), pdh)
                target_low = min(float(overnight_bars["low"].min()), pdl)
            else:
                target_high, target_low = pdh, pdl
            return target_high, target_low

        elif is_cleanup:
            # Untouched session extremes before close
            target_high = float(today_bars["high"].max()) if not today_bars.empty else pdh
            target_low = float(today_bars["low"].min()) if not today_bars.empty else pdl
            return target_high, target_low

        else:
            return pdh, pdl

    # ──────────────────────────────────────────────────────────────────────────
    # Market Structure & Immediate Next Clean Swing Liquidity Targets
    # ──────────────────────────────────────────────────────────────────────────

    def _get_last_minor_pivot(self, df: pd.DataFrame, is_high: bool) -> Optional[float]:
        """Find the most recent 3-bar swing pivot high or low to define the structure break."""
        for i in range(len(df) - 2, 2, -1):
            if is_high:
                if df["high"].iloc[i] > df["high"].iloc[i - 1] and df["high"].iloc[i] > df["high"].iloc[i + 1]:
                    return float(df["high"].iloc[i])
            else:
                if df["low"].iloc[i] < df["low"].iloc[i - 1] and df["low"].iloc[i] < df["low"].iloc[i + 1]:
                    return float(df["low"].iloc[i])
        return None

    def _get_next_clean_swing_high(self, df: pd.DataFrame, current_price: float, fallback: float) -> float:
        """
        Locate the very next immediate clean swing high above current price.
        Does not arbitrarily target far session extremes if an intermediate clean pool exists.
        """
        clean_highs = []
        for i in range(len(df) - 2, 15, -1):
            # 3-bar pivot high check
            if df["high"].iloc[i] > df["high"].iloc[i - 1] and df["high"].iloc[i] > df["high"].iloc[i + 1]:
                lvl = float(df["high"].iloc[i])
                if lvl > current_price:
                    if self._is_clean_level(df, i, lvl, is_high=True):
                        clean_highs.append(lvl)

        if clean_highs:
            # Return the nearest clean swing high above price
            return min(clean_highs)
        return fallback

    def _get_next_clean_swing_low(self, df: pd.DataFrame, current_price: float, fallback: float) -> float:
        """
        Locate the very next immediate clean swing low below current price.
        Does not arbitrarily target far session extremes if an intermediate clean pool exists.
        """
        clean_lows = []
        for i in range(len(df) - 2, 15, -1):
            # 3-bar pivot low check
            if df["low"].iloc[i] < df["low"].iloc[i - 1] and df["low"].iloc[i] < df["low"].iloc[i + 1]:
                lvl = float(df["low"].iloc[i])
                if lvl < current_price:
                    if self._is_clean_level(df, i, lvl, is_high=False):
                        clean_lows.append(lvl)

        if clean_lows:
            # Return the nearest clean swing low below price
            return max(clean_lows)
        return fallback

    # ──────────────────────────────────────────────────────────────────────────
    # Main Trading Iteration (Every 5-Minute Bar)
    # ──────────────────────────────────────────────────────────────────────────

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        ticker_str = symbol.symbol if hasattr(symbol, "symbol") else str(symbol)

        # 1. Fetch 5-Minute Candle Data
        bars = self.get_historical_prices(symbol, 200, "5M")
        if bars is None or len(bars.df) < 30:
            return

        df = self._compute_intraday_indicators(bars.df)
        current_dt = self.get_datetime()
        current_price = self.get_last_price(symbol)
        if current_price is None:
            return

        last_candle = df.iloc[-1]
        candle_idx = len(df) - 1
        current_atr = float(last_candle["atr"])
        vwap = float(last_candle["vwap"])
        is_vwap_flat = bool(last_candle["vwap_slope"] < self.parameters["MinVwapSlope"])

        # 2. Active Time Window Targets & Flat VWAP Filter
        target_high, target_low = self._get_active_session_targets(df, current_dt)
        if target_high is None or target_low is None or is_vwap_flat:
            if self.state in ["SWEPT_LONG", "SWEPT_SHORT", "WAITING_FOR_FILL"]:
                self.log_message(f"[{ticker_str}] Entered Dead Zone or Flat VWAP. Purging pending state.")
                self._reset_to_idle()
            return

        vol_spike = bool(last_candle["volume"] > last_candle["vol_ma"])

        # ──────────────────────────────────────────────────────────────────────
        # STATE 1: WAITING FOR THE SWEEP (THE TRAP)
        # ──────────────────────────────────────────────────────────────────────
        if self.state == "IDLE":
            pos = self.get_position(symbol)
            if pos is not None and pos.quantity != 0:
                return

            # Long Sweep (Wick below support target)
            if last_candle["low"] < target_low and vol_spike:
                if self._is_clean_level(df, candle_idx, target_low, is_high=False):
                    self.trade_model = "REVERSAL" if last_candle["close"] < vwap else "CONTINUATION"
                    self.state = "SWEPT_LONG"
                    self.sweep_extreme = float(last_candle["low"])
                    self.sweep_bar_idx = candle_idx
                    self.target_level = target_low
                    self.target_pivot = self._get_last_minor_pivot(df, is_high=True) or target_low
                    self.log_message(
                        f"[{ticker_str}] STATE 1 -> SWEPT_LONG: Low={last_candle['low']:.2f}, "
                        f"Level={target_low:.2f}, Model={self.trade_model}, Extreme={self.sweep_extreme:.2f}"
                    )

            # Short Sweep (Wick above resistance target)
            elif last_candle["high"] > target_high and vol_spike:
                if self._is_clean_level(df, candle_idx, target_high, is_high=True):
                    self.trade_model = "REVERSAL" if last_candle["close"] > vwap else "CONTINUATION"
                    self.state = "SWEPT_SHORT"
                    self.sweep_extreme = float(last_candle["high"])
                    self.sweep_bar_idx = candle_idx
                    self.target_level = target_high
                    self.target_pivot = self._get_last_minor_pivot(df, is_high=False) or target_high
                    self.log_message(
                        f"[{ticker_str}] STATE 1 -> SWEPT_SHORT: High={last_candle['high']:.2f}, "
                        f"Level={target_high:.2f}, Model={self.trade_model}, Extreme={self.sweep_extreme:.2f}"
                    )

        # ──────────────────────────────────────────────────────────────────────
        # STATE 2: WAITING FOR THE RECLAIM (THE REJECTION)
        # ──────────────────────────────────────────────────────────────────────
        elif self.state in ["SWEPT_LONG", "SWEPT_SHORT"]:
            bars_since_sweep = candle_idx - (self.sweep_bar_idx or candle_idx)

            if self.state == "SWEPT_LONG":
                # Update extreme wick if price pushed lower during the sweep
                if last_candle["low"] < self.sweep_extreme:
                    self.sweep_extreme = float(last_candle["low"])

                # Acceptance Timeout: If price stays below level > MAX_RECLAIM_CANDLES, cancel
                if bars_since_sweep > self.parameters["MaxReclaimCandles"]:
                    self.log_message(f"[{ticker_str}] Long sweep timed out (> {self.parameters['MaxReclaimCandles']} bars). Price acceptance. Resetting.")
                    self._reset_to_idle()
                    return

                # Reclaim: Candle closes back above the swept level
                if last_candle["close"] > self.target_level:
                    self.state = "RECLAIMED_LONG"
                    self.log_message(f"[{ticker_str}] STATE 2 -> RECLAIMED_LONG: Close={last_candle['close']:.2f} > Level={self.target_level:.2f}")

            elif self.state == "SWEPT_SHORT":
                if last_candle["high"] > self.sweep_extreme:
                    self.sweep_extreme = float(last_candle["high"])

                if bars_since_sweep > self.parameters["MaxReclaimCandles"]:
                    self.log_message(f"[{ticker_str}] Short sweep timed out (> {self.parameters['MaxReclaimCandles']} bars). Price acceptance. Resetting.")
                    self._reset_to_idle()
                    return

                # Reclaim: Candle closes back below the swept level
                if last_candle["close"] < self.target_level:
                    self.state = "RECLAIMED_SHORT"
                    self.log_message(f"[{ticker_str}] STATE 2 -> RECLAIMED_SHORT: Close={last_candle['close']:.2f} < Level={self.target_level:.2f}")

        # ──────────────────────────────────────────────────────────────────────
        # STATE 3: WAITING FOR THE TRIGGER (STRUCTURE SHIFT)
        # ──────────────────────────────────────────────────────────────────────
        elif self.state in ["RECLAIMED_LONG", "RECLAIMED_SHORT"]:
            if self.state == "RECLAIMED_LONG":
                # Fail-safe: Price violates the sweep wick low
                if last_candle["low"] < self.sweep_extreme:
                    self.log_message(f"[{ticker_str}] Long setup invalidated: low pierced sweep extreme.")
                    self._reset_to_idle()
                    return

                struct_shift = bool(self.target_pivot and (last_candle["close"] > self.target_pivot))
                vwap_reclaim = bool((last_candle["close"] > vwap) and (self.trade_model == "REVERSAL"))

                if (struct_shift or vwap_reclaim) and vol_spike:
                    # Next Immediate Clean Liquidity Pool Target
                    next_tp = self._get_next_clean_swing_high(df, current_price, fallback=target_high)
                    self._arm_trade_execution(
                        side="buy",
                        pivot_entry=self.target_pivot or vwap,
                        sweep_extreme=self.sweep_extreme,
                        atr=current_atr,
                        target=next_tp,
                        df=df
                    )

            elif self.state == "RECLAIMED_SHORT":
                if last_candle["high"] > self.sweep_extreme:
                    self.log_message(f"[{ticker_str}] Short setup invalidated: high pierced sweep extreme.")
                    self._reset_to_idle()
                    return

                struct_shift = bool(self.target_pivot and (last_candle["close"] < self.target_pivot))
                vwap_reclaim = bool((last_candle["close"] < vwap) and (self.trade_model == "REVERSAL"))

                if (struct_shift or vwap_reclaim) and vol_spike:
                    # Next Immediate Clean Liquidity Pool Target
                    next_tp = self._get_next_clean_swing_low(df, current_price, fallback=target_low)
                    self._arm_trade_execution(
                        side="sell",
                        pivot_entry=self.target_pivot or vwap,
                        sweep_extreme=self.sweep_extreme,
                        atr=current_atr,
                        target=next_tp,
                        df=df
                    )

        # ──────────────────────────────────────────────────────────────────────
        # STATE 5: PENDING ORDER MANAGEMENT & MISSED PULLBACK RULE
        # ──────────────────────────────────────────────────────────────────────
        elif self.state == "WAITING_FOR_FILL":
            # 1. Missed Pullback Rule: Price reached Take Profit before filling entry limit
            if self.pending_side == "buy" and last_candle["high"] >= self.take_profit:
                self.log_message(f"[{ticker_str}] MISSED PULLBACK: Price reached TP (${self.take_profit:.2f}) without us. Purging order.")
                self._reset_to_idle()
                return

            if self.pending_side == "sell" and last_candle["low"] <= self.take_profit:
                self.log_message(f"[{ticker_str}] MISSED PULLBACK: Price reached TP (${self.take_profit:.2f}) without us. Purging order.")
                self._reset_to_idle()
                return

            # 2. Invalidation: Price breached the sweep extreme while waiting for pullback
            if self.pending_side == "buy" and last_candle["low"] < self.sweep_extreme:
                self.log_message(f"[{ticker_str}] Setup invalidated: Low dropped below sweep extreme while waiting for fill.")
                self._reset_to_idle()
                return

            if self.pending_side == "sell" and last_candle["high"] > self.sweep_extreme:
                self.log_message(f"[{ticker_str}] Setup invalidated: High rose above sweep extreme while waiting for fill.")
                self._reset_to_idle()
                return

    # ──────────────────────────────────────────────────────────────────────────
    # Execution & ATR Buffered Stop Loss
    # ──────────────────────────────────────────────────────────────────────────

    def _arm_trade_execution(
        self, side: str, pivot_entry: float, sweep_extreme: float, atr: float, target: float, df: pd.DataFrame
    ):
        """
        Build and submit bracket limit order with ATR buffered stop loss.
        
        Stop Loss Mechanics:
          - Designated Stop = sweep_extreme +/- 1 tick.
          - ATR Buffer = ATR_Stop_Buffer_Multiplier * ATR added beyond the designated stop
            to absorb sudden market spread noise or secondary wick taps.
        """
        symbol = self.parameters["Ticker"]
        ticker_str = symbol.symbol if hasattr(symbol, "symbol") else str(symbol)
        tick = float(self.parameters.get("TickSize", 0.01))
        atr_buffer_mult = float(self.parameters.get("ATR_Stop_Buffer_Multiplier", 0.5))
        atr_buffer = atr * atr_buffer_mult

        if side == "buy":
            designated_sl = sweep_extreme - tick
            buffered_sl = designated_sl - atr_buffer
        else:
            designated_sl = sweep_extreme + tick
            buffered_sl = designated_sl + atr_buffer

        entry_price = round(pivot_entry, 2)
        final_sl = round(buffered_sl, 2)
        final_tp = round(target, 2)

        # Risk and Position Sizing
        risk_per_share = abs(entry_price - final_sl)
        if risk_per_share <= 0:
            self._reset_to_idle()
            return

        portfolio_val = self.get_portfolio_value()
        risk_budget = portfolio_val * float(self.parameters.get("RiskPct", 0.02))
        qty = floor(risk_budget / risk_per_share)

        # Affordability cap (95% cash)
        max_affordable = floor((portfolio_val * 0.95) / max(0.01, entry_price))
        qty = min(qty, max_affordable)

        if qty < 1:
            self.log_message(f"[{ticker_str}] Calculated qty=0 (Budget ${risk_budget:.2f} / Risk/share ${risk_per_share:.2f}). Skipping.")
            self._reset_to_idle()
            return

        # Submit Lumibot OCA Bracket Order
        order = self.create_order(
            asset=symbol,
            quantity=qty,
            side=side,
            limit_price=entry_price,
            stop_loss_price=final_sl,
            take_profit_price=final_tp,
            time_in_force="gtc"
        )
        self.submit_order(order)
        self.active_order = order

        self.pending_side = side
        self.entry_price = entry_price
        self.stop_loss = final_sl
        self.take_profit = final_tp
        self.state = "WAITING_FOR_FILL"

        self.log_message(
            f"\n{'='*65}\n"
            f"[{ticker_str}] SUBMITTED {side.upper()} LIMIT ORDER:\n"
            f"  Qty:            {qty}\n"
            f"  Limit Entry:    ${entry_price:.2f} (Retest of broken structure)\n"
            f"  Sweep Wick:     ${sweep_extreme:.2f}\n"
            f"  Designated SL:  ${designated_sl:.2f}\n"
            f"  ATR Buffer:     ${atr_buffer:.2f} ({atr_buffer_mult}x ATR)\n"
            f"  Buffered SL:    ${final_sl:.2f}\n"
            f"  Take Profit:    ${final_tp:.2f} (Immediate next clean liquidity pool)\n"
            f"  Risk/Reward:    {(abs(final_tp - entry_price) / risk_per_share):.2f}\n"
            f"{'='*65}"
        )

    def _reset_to_idle(self):
        """Purge pending state and cancel working entry orders."""
        if self.active_order is not None and self.state == "WAITING_FOR_FILL":
            try:
                self.cancel_order(self.active_order)
            except Exception:
                pass
        self.state = "IDLE"
        self.trade_model = None
        self.sweep_extreme = None
        self.sweep_bar_idx = None
        self.target_level = None
        self.target_pivot = None
        self.pending_side = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.active_order = None

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Handle execution fills."""
        super().on_filled_order(position, order, price, quantity, multiplier)
        if self.state == "WAITING_FOR_FILL":
            self.state = "IN_POSITION"
            self.log_message(f"Trade filled: In position {quantity} @ ${price:.2f}. Bracket orders active on broker.")
        elif self.state == "IN_POSITION":
            # If position is now closed
            if position is None or position.quantity == 0:
                self.log_message(f"Position closed @ ${price:.2f}.")
                self._reset_to_idle()
