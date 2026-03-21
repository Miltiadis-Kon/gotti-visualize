"""
Multi-Timeframe Key Levels Strategy (v2)

Combines Fibonacci retracement analysis with S/R key levels:
- Recalculates every 15 minutes via iteration counter (3 × 5min sleeptime)
- Uses analyze() from strategies.key_levels for multi-resolution analysis
- Fibonacci trade setups drive primary entry/exit (entry_price, stop_loss, take_profit)
- S/R level entries as secondary signals (existing LONG at support / SHORT at resistance)
- Noise reduction: Fibonacci signal prioritized when overlapping with S/R

Data source: analyze(ticker, resolutions=['1D', '4H', '15m'])
"""

import pandas as pd
import sys
import os
import io
import contextlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_key_levels_strategy import BaseKeyLevelsStrategy
from key_levels.analyzer import analyze

load_dotenv()


class MultiTimeframeKeyLevelsStrategy(BaseKeyLevelsStrategy):
    """
    Multi-Timeframe Key Levels Trading Strategy (v2).

    Primary signals: Fibonacci trade setups (entry at 0.618 retracement, SL at 0.786,
    TP at swing extreme). Secondary signals: S/R level entries with R:R filtering.

    Every 15 minutes (3 iterations × 5min sleeptime), calls analyze() to refresh:
    - Merged S/R key levels across 1D, 4H, 15m
    - Fibonacci trade setups (latest pattern per resolution)

    Noise reduction: if Fibonacci and S/R entries are within FIB_SR_PROXIMITY of
    each other, only the Fibonacci signal is used. S/R signals near inactive Fibonacci
    entry zones are also suppressed.
    """

    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RISK_PERCENT": 0.02,          # Risk 2% per trade
        "MIN_IMPORTANCE": 3,           # Minimum importance for S/R levels
        "TIMEFRAMES": ['1d', '4h', '1h', '15m', '5m'],  # kept for base compat
        "PRICE_THRESHOLD": 0.5,        # Threshold for merging S/R levels
        "RECALC_FREQUENCY": "daily",   # kept for base compat (overridden by counter)

        # Entry/Exit thresholds
        "ENTRY_THRESHOLD": 0.005,      # 0.5% tolerance for level matching
        "EXIT_THRESHOLD": 0.01,        # 1% tolerance for TP/SL matching

        # S/R-specific thresholds
        "TP_THRESHOLD": 0.02,          # Take profit 2% before resistance
        "SL_THRESHOLD": 0.05,          # Stop loss 5% below support
        "MIN_RISK_REWARD": 1.5,        # Minimum R:R ratio

        # Analyzer configuration
        "ANALYSIS_RESOLUTIONS": ['1D', '4H', '15m'],  # Resolutions for analyze()
        "RECALC_ITERATIONS": 3,        # Recalc every N iterations (3 × 5min = 15min)
        "FIB_SR_PROXIMITY": 0.02,      # 2% proximity threshold for noise reduction
        "USE_ALPACA": False,           # Use Yahoo Finance by default
    }

    # Class-level analysis cache (persists across backtesting iterations)
    _analysis_cache = {}

    def get_strategy_name(self) -> str:
        return "MultiTFKeyLevels"

    def on_strategy_start(self):
        """Initialize strategy-specific state."""
        # S/R threshold parameters
        self.tp_threshold = self.parameters.get("TP_THRESHOLD", 0.02)
        self.sl_threshold = self.parameters.get("SL_THRESHOLD", 0.05)
        self.min_risk_reward = self.parameters.get("MIN_RISK_REWARD", 1.5)

        # Analyzer parameters
        self._recalc_every = self.parameters.get("RECALC_ITERATIONS", 3)
        self._iteration_count = 0
        self._analysis_loaded = False
        self._last_analysis_date = None
        self.fib_trade_setups = pd.DataFrame()
        self.fib_sr_proximity = self.parameters.get("FIB_SR_PROXIMITY", 0.02)
        self._analysis_resolutions = self.parameters.get(
            "ANALYSIS_RESOLUTIONS", ['1D', '4H', '15m']
        )
        self._use_alpaca = self.parameters.get("USE_ALPACA", False)

        # Ensure S/R levels start as empty DataFrames (not None)
        # so base class _handle_entry() doesn't bail early
        self.support_levels = pd.DataFrame()
        self.resistance_levels = pd.DataFrame()

        self.log_message(
            f"[{self.get_strategy_name()}] Initialized (FIB+SR, LONG+SHORT) - "
            f"Min R:R={self.min_risk_reward}, Recalc every {self._recalc_every} "
            f"iterations, Resolutions={self._analysis_resolutions}"
        )

    # ─────────────────── LIFECYCLE OVERRIDES ───────────────────

    def on_trading_iteration(self):
        """
        Main trading loop — overrides base class to use counter-based
        recalculation with analyze() instead of the old KeyLevels class.
        """
        ticker = self.parameters["Ticker"]

        # Get current price
        current_price = self.get_last_price(ticker)
        if current_price is None:
            return

        # Counter-based recalculation (every N iterations = 15 min)
        self._iteration_count += 1
        if self._iteration_count >= self._recalc_every or not self._analysis_loaded:
            self._iteration_count = 0
            self._run_analysis()

        # Need at least one successful analysis to proceed
        if not self._analysis_loaded:
            return

        # Get current position
        position = self.get_position(ticker)
        has_position = position is not None and position.quantity != 0

        # Check for entry (calls get_entry_signal via base _handle_entry)
        self._handle_entry(current_price)

        # If position exists, check for exit
        if has_position:
            self._handle_exit(current_price, position)

    # ─────────────────── ANALYSIS ENGINE ───────────────────

    def _run_analysis(self):
        """
        Run analyze() to refresh S/R levels and Fibonacci trade setups.

        Uses a date-based cache: if the same ticker+date was already analyzed
        (common during backtesting where intraday data doesn't change), the
        cached result is reused instantly.
        """
        ticker_symbol = self.parameters["Ticker"].symbol
        current_dt = self.get_datetime()
        current_date = current_dt.date()

        # Cache key: same ticker + same date = same levels
        cache_key = f"{ticker_symbol}_{current_date}"

        if cache_key in MultiTimeframeKeyLevelsStrategy._analysis_cache:
            cached = MultiTimeframeKeyLevelsStrategy._analysis_cache[cache_key]
            self.merged_levels = cached['merged']
            self.support_levels = cached['support']
            self.resistance_levels = cached['resistance']
            self.fib_trade_setups = cached['fib_setups']
            self._analysis_loaded = True
            # Only log on first load of each day
            if current_date != self._last_analysis_date:
                self.log_message(
                    f"[{self.get_strategy_name()}] Loaded cached analysis for {current_date}: "
                    f"{len(self.support_levels)} supports, "
                    f"{len(self.resistance_levels)} resistances, "
                    f"{len(self.fib_trade_setups)} fib setups"
                )
                self._last_analysis_date = current_date
            return

        # Fetch fresh analysis (suppress analyze() print output)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(
                    ticker_symbol,
                    resolutions=self._analysis_resolutions,
                    use_alpaca=self._use_alpaca,
                    as_of_date=current_dt
                )

            # Populate S/R levels
            if not result.merged_levels.empty:
                self.merged_levels = result.merged_levels
                filtered = result.merged_levels[
                    result.merged_levels['importance'] >= self.min_importance
                ]
                self.support_levels = filtered[
                    filtered['type'] == 'support'
                ].copy()
                self.resistance_levels = filtered[
                    filtered['type'] == 'resistance'
                ].copy()
            else:
                self.merged_levels = pd.DataFrame()
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()

            # Store Fibonacci trade setups
            self.fib_trade_setups = (
                result.trade_setups
                if not result.trade_setups.empty
                else pd.DataFrame()
            )

            self._analysis_loaded = True
            self._last_analysis_date = current_date

            # Cache the result
            MultiTimeframeKeyLevelsStrategy._analysis_cache[cache_key] = {
                'merged': self.merged_levels,
                'support': self.support_levels,
                'resistance': self.resistance_levels,
                'fib_setups': self.fib_trade_setups,
            }

            # Log levels history (for base class after_market_closes persistence)
            if not self.support_levels.empty or not self.resistance_levels.empty:
                self._log_levels_to_history(current_date)

            self.log_message(
                f"[{self.get_strategy_name()}] Analysis refreshed ({current_date}): "
                f"{len(self.support_levels)} supports, "
                f"{len(self.resistance_levels)} resistances, "
                f"{len(self.fib_trade_setups)} fib setups"
            )

        except Exception as e:
            self.log_message(f"[{self.get_strategy_name()}] Analysis error: {e}")
            # Keep previous levels if available
            if not self._analysis_loaded:
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()
                self.fib_trade_setups = pd.DataFrame()

    # ─────────────────── ENTRY SIGNALS ───────────────────

    def get_entry_signal(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Combined entry logic: Fibonacci primary, S/R secondary.

        Priority order:
        1. Fibonacci trade setups (uptrend → BUY, downtrend → SELL)
        2. S/R level entries (existing LONG at support / SHORT at resistance)

        Noise reduction:
        - If both Fibonacci and S/R signal → use Fibonacci
        - If only S/R signal but it's near a Fibonacci entry zone → suppress it
        """
        # Check Fibonacci entry signals
        fib_signal = self._check_fib_entry(current_price)

        # Check S/R entry signals (existing logic)
        sr_signal = self._check_sr_entry(current_price, support_levels, resistance_levels)

        # ── Priority & noise reduction ──

        # Case 1: Fibonacci signal exists → always use it
        if fib_signal:
            if sr_signal:
                self.log_message(
                    f"[{self.get_strategy_name()}] Fib + S/R both signal — "
                    f"using Fibonacci (noise reduction)"
                )
            return fib_signal

        # Case 2: Only S/R signal — check if it's near a Fibonacci zone
        if sr_signal:
            if self._is_near_fib_zone(sr_signal['entry_price']):
                self.log_message(
                    f"[{self.get_strategy_name()}] S/R signal suppressed "
                    f"(near Fibonacci entry zone — waiting for exact fib entry)"
                )
                return None
            return sr_signal

        return None

    def _check_fib_entry(self, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Check if current price matches any Fibonacci trade setup entry.

        For uptrend patterns: BUY at 0.618 retracement, TP at swing high, SL at 0.786
        For downtrend patterns: SELL at 0.618 retracement, TP at swing low, SL at 0.786
        """
        if self.fib_trade_setups.empty:
            return None

        for _, setup in self.fib_trade_setups.iterrows():
            entry_price = setup['entry_price']
            tolerance = entry_price * self.entry_threshold

            # Check if price is at the fibonacci entry level
            if not (entry_price - tolerance <= current_price <= entry_price + tolerance):
                continue

            # Determine trade direction from fibonacci trend
            trend = setup['trend']
            trade_type = 'BUY' if trend == 'uptrend' else 'SELL'

            take_profit = setup['take_profit']
            stop_loss = setup['stop_loss']

            # Validate R:R ratio
            if trade_type == 'BUY':
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:
                risk = stop_loss - current_price
                reward = current_price - take_profit

            risk_reward = reward / risk if risk > 0 else 0

            if risk_reward < self.min_risk_reward:
                continue

            resolution = setup.get('resolution', 'N/A')
            pattern_id = setup.get('pattern_id', 'N/A')
            self.log_message(
                f"[{self.get_strategy_name()}] FIB {trade_type} signal "
                f"({resolution}, {pattern_id}): "
                f"Entry ${entry_price:.2f}, TP ${take_profit:.2f}, "
                f"SL ${stop_loss:.2f}, R:R={risk_reward:.2f}"
            )

            # Map swing extremes to support/resistance for trade tracking
            support_level = setup.get('low_price', stop_loss)
            resistance_level = setup.get('high_price', take_profit)

            return {
                'trade_type': trade_type,
                'entry_price': current_price,
                'take_profit': round(take_profit, 2),
                'stop_loss': round(stop_loss, 2),
                'support_level': support_level,
                'resistance_level': resistance_level,
            }

        return None

    def _check_sr_entry(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Check for entry at S/R levels (existing LONG + SHORT logic).
        Returns the first valid signal or None.
        """
        if support_levels.empty or resistance_levels.empty:
            return None

        # Try LONG at support first
        long_signal = self._check_long_entry(
            current_price, support_levels, resistance_levels
        )
        if long_signal:
            return long_signal

        # Try SHORT at resistance
        short_signal = self._check_short_entry(
            current_price, support_levels, resistance_levels
        )
        if short_signal:
            return short_signal

        return None

    def _is_near_fib_zone(self, price: float) -> bool:
        """
        Check if a price is near any Fibonacci entry zone.
        Used to suppress S/R signals that overlap with upcoming Fibonacci entries.
        """
        if self.fib_trade_setups.empty:
            return False

        for _, setup in self.fib_trade_setups.iterrows():
            fib_entry = setup['entry_price']
            proximity = abs(fib_entry - price) / max(fib_entry, price)
            if proximity <= self.fib_sr_proximity:
                return True

        return False

    # ─────────────────── S/R ENTRY LOGIC (preserved) ───────────────────

    def _check_long_entry(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Check for LONG entry at support level."""
        supports = support_levels.copy()
        supports['distance'] = abs(supports['level_price'] - current_price)
        supports = supports.sort_values(
            ['importance', 'distance'], ascending=[False, True]
        )

        resistances = resistance_levels.sort_values('level_price', ascending=True)

        for _, support in supports.iterrows():
            support_price = support['level_price']
            support_importance = support['importance']

            # Check if price is at this support
            tolerance = support_price * self.entry_threshold
            if not (support_price - tolerance <= current_price <= support_price + tolerance):
                continue

            entry_price = current_price
            stop_loss = self._calculate_long_sl(support_price)
            risk_per_share = entry_price - stop_loss

            if risk_per_share <= 0:
                continue

            # Find resistance that meets R:R
            resistances_above = resistances[resistances['level_price'] > entry_price]

            for _, resistance in resistances_above.iterrows():
                target_resistance = resistance['level_price']
                take_profit = self._calculate_long_tp(target_resistance)
                reward_per_share = take_profit - entry_price

                risk_reward = reward_per_share / risk_per_share

                if risk_reward >= self.min_risk_reward:
                    self.log_message(
                        f"[{self.get_strategy_name()}] S/R LONG signal: "
                        f"Support ${support_price:.2f} (imp={support_importance}), "
                        f"Resistance ${target_resistance:.2f}, R:R={risk_reward:.2f}"
                    )

                    return {
                        'trade_type': 'BUY',
                        'entry_price': entry_price,
                        'take_profit': round(take_profit, 2),
                        'stop_loss': round(stop_loss, 2),
                        'support_level': support_price,
                        'resistance_level': target_resistance,
                    }

            # Fallback with fixed R:R
            if not resistances_above.empty:
                fallback_reward = risk_per_share * self.min_risk_reward
                fallback_tp = entry_price + fallback_reward
                nearest_resistance = resistances_above.iloc[0]['level_price']

                return {
                    'trade_type': 'BUY',
                    'entry_price': entry_price,
                    'take_profit': round(fallback_tp, 2),
                    'stop_loss': round(stop_loss, 2),
                    'support_level': support_price,
                    'resistance_level': nearest_resistance,
                }

        return None

    def _check_short_entry(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Check for SHORT entry at resistance level."""
        resistances = resistance_levels.copy()
        resistances['distance'] = abs(resistances['level_price'] - current_price)
        resistances = resistances.sort_values(
            ['importance', 'distance'], ascending=[False, True]
        )

        supports = support_levels.sort_values('level_price', ascending=False)

        for _, resistance in resistances.iterrows():
            resistance_price = resistance['level_price']
            resistance_importance = resistance['importance']

            # Check if price is at this resistance
            tolerance = resistance_price * self.entry_threshold
            if not (resistance_price - tolerance <= current_price <= resistance_price + tolerance):
                continue

            entry_price = current_price
            stop_loss = self._calculate_short_sl(resistance_price)
            risk_per_share = stop_loss - entry_price

            if risk_per_share <= 0:
                continue

            # Find support that meets R:R
            supports_below = supports[supports['level_price'] < entry_price]

            for _, support in supports_below.iterrows():
                target_support = support['level_price']
                take_profit = self._calculate_short_tp(target_support)
                reward_per_share = entry_price - take_profit

                risk_reward = reward_per_share / risk_per_share

                if risk_reward >= self.min_risk_reward:
                    self.log_message(
                        f"[{self.get_strategy_name()}] S/R SHORT signal: "
                        f"Resistance ${resistance_price:.2f} "
                        f"(imp={resistance_importance}), "
                        f"Support ${target_support:.2f}, R:R={risk_reward:.2f}"
                    )

                    return {
                        'trade_type': 'SELL',
                        'entry_price': entry_price,
                        'take_profit': round(take_profit, 2),
                        'stop_loss': round(stop_loss, 2),
                        'support_level': target_support,
                        'resistance_level': resistance_price,
                    }

            # Fallback with fixed R:R
            if not supports_below.empty:
                fallback_reward = risk_per_share * self.min_risk_reward
                fallback_tp = entry_price - fallback_reward
                nearest_support = supports_below.iloc[0]['level_price']

                return {
                    'trade_type': 'SELL',
                    'entry_price': entry_price,
                    'take_profit': round(fallback_tp, 2),
                    'stop_loss': round(stop_loss, 2),
                    'support_level': nearest_support,
                    'resistance_level': resistance_price,
                }

        return None

    # ─────────────────── EXIT SIGNALS ───────────────────

    def get_exit_signal(
        self,
        current_price: float,
        position,
        entry_support: float,
        target_resistance: float
    ) -> Optional[str]:
        """
        Manual exit check (backup for bracket orders).
        Uses exit_threshold for tolerance matching.
        Works for both Fibonacci and S/R originated trades.
        """
        trade = self.trade_tracker.get_trade(self.current_trade_id)
        if trade is None:
            return None

        # Calculate tolerance zones
        tp_tolerance = trade.take_profit * self.exit_threshold
        sl_tolerance = trade.stop_loss * self.exit_threshold

        if trade.trade_type == "BUY":
            # LONG position — TP is above, SL is below
            if current_price >= trade.take_profit - tp_tolerance:
                return "TP"
            if current_price <= trade.stop_loss + sl_tolerance:
                return "SL"
        else:
            # SHORT position — TP is below, SL is above
            if current_price <= trade.take_profit + tp_tolerance:
                return "TP"
            if current_price >= trade.stop_loss - sl_tolerance:
                return "SL"

        return None

    # ─────────────────── TP/SL HELPERS (for S/R entries) ───────────────────

    def _calculate_long_tp(self, resistance_price: float) -> float:
        """Calculate LONG TP price (threshold% before resistance)."""
        return resistance_price * (1 - self.tp_threshold)

    def _calculate_long_sl(self, support_price: float) -> float:
        """Calculate LONG SL price (threshold% below support)."""
        return support_price * (1 - self.sl_threshold)

    def _calculate_short_tp(self, support_price: float) -> float:
        """Calculate SHORT TP price (threshold% above support)."""
        return support_price * (1 + self.tp_threshold)

    def _calculate_short_sl(self, resistance_price: float) -> float:
        """Calculate SHORT SL price (threshold% above resistance)."""
        return resistance_price * (1 + self.sl_threshold)

    def _handle_entry(self, current_price: float):
        """
        Override to enforce correct Risk:Reward via Limit Orders and accurately
        match Lumibot orders to Trade ID to fix mismatch logging bugs.
        """
        if self.support_levels is None or self.resistance_levels is None:
            return
        
        signal = self.get_entry_signal(
            current_price,
            self.support_levels,
            self.resistance_levels
        )
        if signal is None:
            return
            
        trade_type = signal.get('trade_type', 'BUY')
        entry_price = signal.get('entry_price', current_price)
        take_profit = signal['take_profit']
        stop_loss = signal['stop_loss']
        support_level = signal['support_level']
        resistance_level = signal['resistance_level']
        
        entry_level = support_level if trade_type == 'BUY' else resistance_level
        if self._is_level_already_entered(entry_level, trade_type):
            return
            
        quantity = signal.get('quantity')
        if quantity is None:
            quantity = self.get_position_sizing(entry_price, stop_loss)
            
        if quantity <= 0:
            return
            
        self._mark_level_entered(entry_level, trade_type)
        self.entry_support = support_level
        self.target_resistance = resistance_level
        
        trade_id = self.trade_tracker.open_trade(
            date=self.get_datetime(),
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            support_level=support_level,
            resistance_level=resistance_level,
            trade_type=trade_type
        )
        self.current_trade_id = trade_id
        
        side = "buy" if trade_type == "BUY" else "sell"
        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=quantity,
            side=side,
            limit_price=entry_price,
            secondary_limit_price=take_profit,
            secondary_stop_price=stop_loss,
            order_class="bracket",
        )
        self.submit_order(order)
        
        if not hasattr(self, 'active_trade_map'):
            self.active_trade_map = {}
        self.active_trade_map[order.identifier] = trade_id
        if getattr(order, 'child_orders', None):
            for child in order.child_orders:
                self.active_trade_map[child.identifier] = trade_id
    def on_filled_order(self, position, order, price, quantity, multiplier):
        """
        Handle filled orders with perfect logging for Entry/Exit, TP, SL, and Fibonacci levels.
        Fixes base class bug with SHORT trades missing/closing immediately.
        """
        if not hasattr(self, 'active_trade_map'):
            self.active_trade_map = {}

        # Safe extraction of the correct trade ID
        mapped_trade_id = self.active_trade_map.get(order.identifier, self.current_trade_id)
        trade = self.trade_tracker.get_trade(mapped_trade_id)
        if not trade:
            return

        is_entry = False
        is_exit = False
        
        if trade.trade_type == "BUY":
            if order.side == "buy":
                is_entry = True
            elif order.side == "sell":
                is_exit = True
        elif trade.trade_type == "SELL":
            if order.side == "sell":
                is_entry = True
            elif order.side == "buy":
                is_exit = True

        current_time = self.get_datetime().strftime("%Y-%m-%d %H:%M")

        if is_entry:
            msg = (
                f"\n{'='*60}\n"
                f"🚀 [ENTRY {mapped_trade_id} FILLED] {current_time} | {trade.trade_type}\n"
                f"   Asset:        {self.parameters['Ticker'].symbol}\n"
                f"   Quantity:     {quantity} @ ${price:.2f}\n"
                f"   Take Profit:  ${trade.take_profit:.2f}\n"
                f"   Stop Loss:    ${trade.stop_loss:.2f}\n"
                f"   Fib Levels:   Low ${trade.support_level:.2f} - High ${trade.resistance_level:.2f}\n"
                f"{'='*60}\n"
            )
            print(msg)
            self.log_message(msg)

        elif is_exit:
            exit_reason = self._determine_exit_reason(price)
            closed_trade = self.trade_tracker.close_trade(
                trade_id=mapped_trade_id,
                date=self.get_datetime(),
                exit_price=price,
                exit_reason=exit_reason
            )
            
            pnl_str = f"+${closed_trade.pnl:.2f}" if closed_trade.pnl >= 0 else f"-${abs(closed_trade.pnl):.2f}"
            icon = "✅" if closed_trade.pnl > 0 else "❌"
            
            msg = (
                f"\n{'='*60}\n"
                f"{icon} [EXIT OF ENTRY {mapped_trade_id} FILLED] {current_time} | {trade.trade_type}\n"
                f"   Reason:       {exit_reason}\n"
                f"   Quantity:     {quantity} @ ${price:.2f}\n"
                f"   Realized P&L: {pnl_str}\n"
                f"   Take Profit:  ${trade.take_profit:.2f}\n"
                f"   Stop Loss:    ${trade.stop_loss:.2f}\n"
                f"{'='*60}\n"
            )
            print(msg)
            self.log_message(msg)
            
            if mapped_trade_id == self.current_trade_id:
                self.current_trade_id = None
            self.entry_support = None
            self.target_resistance = None

    def _determine_exit_reason(self, exit_price: float) -> str:
        """Determine exit reason accurately for both LONG and SHORT."""
        trade = self.trade_tracker.get_trade(self.current_trade_id)
        if trade:
            if trade.trade_type == "BUY":
                if exit_price >= trade.take_profit * 0.99:
                    return "TP"
                elif exit_price <= trade.stop_loss * 1.01:
                    return "SL"
            else: # SELL
                if exit_price <= trade.take_profit * 1.01:
                    return "TP"
                elif exit_price >= trade.stop_loss * 0.99:
                    return "SL"
        return "MANUAL"


def run_backtest(
    ticker: str = "NVDA",
    start_date: datetime = None,
    end_date: datetime = None,
    budget: float = 10000,
    min_importance: int = 3,
    min_risk_reward: float = 1.5,
    save_files: bool = False
):
    """Run backtest of Multi-Timeframe Key Levels Strategy."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now()

    print(f"Running Multi-TF Key Levels (v2) backtest for {ticker}")
    print(f"Period: {start_date.date()} to {end_date.date()}")

    MultiTimeframeKeyLevelsStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        budget=budget,
        parameters={
            "Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK),
            "MIN_IMPORTANCE": min_importance,
            "MIN_RISK_REWARD": min_risk_reward,
            "SAVE_FILES": save_files,
        },
        save_logfile=save_files,
        save_tearsheet=save_files,
        show_plot=save_files,
        show_tearsheet=save_files,
        save_stats_file=save_files
    )


if __name__ == "__main__":
    run_backtest(
        ticker="PLTR",
        start_date=datetime(2026, 3, 1),
        end_date=datetime(2026, 3, 21),
        budget=10000,
        min_importance=2,
        min_risk_reward=1.5,
        save_files=False
    )
