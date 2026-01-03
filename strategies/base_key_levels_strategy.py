"""
Base Key Levels Strategy

Abstract base class for key levels trading strategies.
Handles:
- Trade tracking and logging via TradeTracker
- Position sizing based on risk percentage
- Key levels detection and caching
- Common lifecycle methods

Child strategies should implement:
- get_entry_signal() - defines when to enter a trade
- get_exit_signal() - defines when to exit a trade
- Strategy-specific parameters
"""

import pandas as pd
import sys
import os
import json
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from lumibot.strategies import Strategy
from lumibot.entities import Asset

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from key_levels import (
    KeyLevels,
    TIMEFRAME_LOOKBACK,
    TIMEFRAME_IMPORTANCE,
    PRICE_THRESHOLD
)
from trade_tracker import TradeTracker


class BaseKeyLevelsStrategy(Strategy):
    """
    Base class for key levels trading strategies.
    
    Provides common functionality:
    - Trade tracking with TradeTracker
    - Position sizing based on risk percentage
    - Key levels loading and caching
    - Automatic trade saving at end of each day
    
    Child classes must implement:
    - get_entry_signal(): Returns entry details when conditions are met
    - get_exit_signal(): Returns exit signal when position should close
    
    Optional overrides:
    - get_strategy_name(): Returns strategy name for logging
    - on_strategy_start(): Called once at strategy start
    """
    
    # Default parameters - child classes can override
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RISK_PERCENT": 0.02,      # Risk 2% of portfolio per trade
        "MIN_IMPORTANCE": 1,       # Minimum level importance to consider
        "TIMEFRAMES": ['1d', '4h', '1h', '15m', '5m'],  # Timeframes to analyze
        "PRICE_THRESHOLD": 0.5,    # Price threshold for merging levels
        "RECALC_FREQUENCY": "daily",  # How often to recalculate: 'daily', 'weekly', 'once'
        "ENTRY_THRESHOLD": 0.05,   # 5% tolerance for entry (price within 5% of level)
        "EXIT_THRESHOLD": 0.1,    # 2% tolerance for TP/SL exits
    }
    
    # Class-level cache for key levels (persists across backtesting iterations)
    _cached_levels = {}
    _last_trade_tracker = None
    
    ##### CORE LIFECYCLE METHODS #####
    
    def initialize(self):
        """
        Initialize the strategy.
        Called once at the start of the strategy.
        Sets up TradeTracker and common variables.
        """
        self.sleeptime = "5M"  # Execute strategy every 5 minutes
        
        # Load parameters
        self.risk_percent = self.parameters.get("RISK_PERCENT", 0.02)
        self.min_importance = self.parameters.get("MIN_IMPORTANCE", 1)
        self.timeframes = self.parameters.get("TIMEFRAMES", ['1d', '4h', '1h', '15m', '5m'])
        self.price_threshold = self.parameters.get("PRICE_THRESHOLD", 0.5)
        self.recalc_frequency = self.parameters.get("RECALC_FREQUENCY", "daily")
        self.entry_threshold = self.parameters.get("ENTRY_THRESHOLD", 0.05)  # 5%
        self.exit_threshold = self.parameters.get("EXIT_THRESHOLD", 0.02)    # 2%
        
        # Key levels storage
        self.merged_levels = None
        self.support_levels = None
        self.resistance_levels = None
        
        # Current trade info
        self.entry_support = None
        self.target_resistance = None
        self.last_levels_date = None
        
        # Trade tracker for recording trades
        ticker = self.parameters["Ticker"].symbol
        self.trade_tracker = TradeTracker(ticker=ticker)
        self.current_trade_id = None
        
        # Generate output filename once at start (uses real system datetime, not simulation)
        # Format: {DDMM}_{HHMM}_{TICKER}_{STRATEGY}.json
        # Example: 2412_1738_MARA_multitfkeylevels.json
        strategy_name = self.get_strategy_name().lower()
        real_datetime = datetime.now()
        date_str = real_datetime.strftime('%d%m')  # DDMM
        time_str = real_datetime.strftime('%H%M')  # HHMM
        self._output_filename = f"{date_str}_{time_str}_{ticker}_{strategy_name}"
        
        # Levels history - stores calculated levels for each day
        self._levels_history = []
        
        # Track entered levels to prevent duplicate entries
        # Store as set of (level_price, level_type) tuples
        self._entered_levels = set()
        
        # Call child's custom initialization
        self.on_strategy_start()
    
    def on_strategy_start(self):
        """
        Called once at strategy start.
        Override in child class for custom initialization.
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Return strategy name for logging. Override in child class."""
        return self.__class__.__name__
    
    def on_trading_iteration(self):
        """
        Main trading loop - called on each iteration.
        Handles level loading and delegates to child's entry/exit logic.
        """
        ticker = self.parameters["Ticker"]
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()
        
        # Check if we need to reload levels
        if self._should_reload_levels(current_date):
            self._load_key_levels(current_datetime)
            self.last_levels_date = current_date
        
        # Get current price
        current_price = self.get_last_price(ticker)
        if current_price is None:
            return
        
        # Get current position
        position = self.get_position(ticker)
        has_position = position is not None and position.quantity != 0
        
        # Always check for entry at new levels (level tracking prevents duplicates)
        self._handle_entry(current_price)
        
        # If we have a position, also check for exit
        if has_position:
            exit_triggered = self._handle_exit(current_price, position)
            if exit_triggered:
                # Clear the specific level that just closed
                pass  # Levels remain tracked until daily reset
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        """
        Handle filled orders - tracks trades automatically.
        """
        ticker = self.parameters["Ticker"].symbol
        
        if order.side == "buy":
            self.log_message(f"[{self.get_strategy_name()}] BUY filled: {quantity} @ ${price:.2f}")
            if self.entry_support and self.target_resistance:
                self.log_message(f"  Support: ${self.entry_support:.2f}, Target: ${self.target_resistance:.2f}")
        
        elif order.side == "sell":
            self.log_message(f"[{self.get_strategy_name()}] SELL filled: {quantity} @ ${price:.2f}")
            
            # Close trade in tracker
            if self.current_trade_id:
                exit_reason = self._determine_exit_reason(price)
                
                closed_trade = self.trade_tracker.close_trade(
                    trade_id=self.current_trade_id,
                    date=self.get_datetime(),
                    exit_price=price,
                    exit_reason=exit_reason
                )
                
                if closed_trade:
                    pnl_str = f"+${closed_trade.pnl:.2f}" if closed_trade.pnl >= 0 else f"-${abs(closed_trade.pnl):.2f}"
                    self.log_message(f"  Trade closed ({exit_reason}): PnL = {pnl_str}")
                
                self.current_trade_id = None
            
            # Reset trade tracking
            self.entry_support = None
            self.target_resistance = None
    
    def after_market_closes(self):
        """
        Called at end of each trading day.
        Saves trades to file automatically.
        Uses filename generated at initialization (real system datetime).
        """
        # Save trade tracker to class-level
        BaseKeyLevelsStrategy._last_trade_tracker = self.trade_tracker
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Use filename generated at initialize() - same file for entire simulation
        json_path = os.path.join(output_dir, f"{self._output_filename}.json")
        self.trade_tracker.save_to_json(json_path)
        
        csv_path = os.path.join(output_dir, f"{self._output_filename}.csv")
        self.trade_tracker.save_to_csv(csv_path)
        
        # Save levels history
        levels_path = os.path.join(output_dir, f"{self._output_filename}_levels.json")
        self._save_levels_history(levels_path)
    
    def on_abrupt_closing(self):
        """Called on crash or manual stop."""
        self.after_market_closes()
        if self.trade_tracker.trades:
            self.trade_tracker.print_summary()
    
    ##### ABSTRACT METHODS - CHILD MUST IMPLEMENT #####
    
    @abstractmethod
    def get_entry_signal(
        self, 
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if we should enter a trade.
        
        Parameters:
        -----------
        current_price : float
            Current market price
        support_levels : pd.DataFrame
            DataFrame with support levels (level_price, importance, touch_count)
        resistance_levels : pd.DataFrame
            DataFrame with resistance levels
            
        Returns:
        --------
        Optional[Dict] with keys:
            - entry_price: float
            - take_profit: float
            - stop_loss: float
            - support_level: float (the support that triggered entry)
            - resistance_level: float (target resistance)
            - quantity: int (optional - if not provided, uses position sizing)
        
        Return None if no entry signal.
        """
        pass
    
    @abstractmethod
    def get_exit_signal(
        self,
        current_price: float,
        position,
        entry_support: float,
        target_resistance: float
    ) -> Optional[str]:
        """
        Determine if we should exit an existing position.
        
        Parameters:
        -----------
        current_price : float
            Current market price
        position : Position
            Current position object
        entry_support : float
            The support level where we entered
        target_resistance : float
            The resistance level we're targeting
            
        Returns:
        --------
        Optional[str]: Exit reason ("TP", "SL", "MANUAL") or None to hold
        """
        pass
    
    ##### HELPER METHODS #####
    
    def _handle_entry(self, current_price: float):
        """Handle entry logic - calls child's get_entry_signal."""
        if self.support_levels is None or self.resistance_levels is None:
            return
        
        signal = self.get_entry_signal(
            current_price,
            self.support_levels,
            self.resistance_levels
        )
        
        if signal is None:
            return
        
        # Extract signal details
        trade_type = signal.get('trade_type', 'BUY')  # Default to BUY for backwards compatibility
        entry_price = signal.get('entry_price', current_price)
        take_profit = signal['take_profit']
        stop_loss = signal['stop_loss']
        support_level = signal['support_level']
        resistance_level = signal['resistance_level']
        
        # Check if we already entered at this level
        entry_level = support_level if trade_type == 'BUY' else resistance_level
        if self._is_level_already_entered(entry_level, trade_type):
            return  # Skip - already entered at this level
        
        # Calculate position size if not provided
        quantity = signal.get('quantity')
        if quantity is None:
            quantity = self.get_position_sizing(entry_price, stop_loss, trade_type)
        
        if quantity <= 0:
            self.log_message("Insufficient funds for trade")
            return
        
        # Mark this level as entered
        self._mark_level_entered(entry_level, trade_type)
        
        # Store trade info
        self.entry_support = support_level
        self.target_resistance = resistance_level
        
        # Record trade in tracker
        self.current_trade_id = self.trade_tracker.open_trade(
            date=self.get_datetime(),
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            support_level=support_level,
            resistance_level=resistance_level,
            trade_type=trade_type
        )
        
        # Log and submit order
        side = "buy" if trade_type == "BUY" else "sell"
        self.log_message(
            f"[{self.get_strategy_name()}] ENTRY: {trade_type} {quantity} @ ${entry_price:.2f}, "
            f"TP: ${take_profit:.2f}, SL: ${stop_loss:.2f}"
        )
        
        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=quantity,
            side=side,
            take_profit_price=take_profit,
            stop_loss_price=stop_loss,
        )
        
        self.submit_order(order)
    
    def _is_level_already_entered(self, level_price: float, trade_type: str) -> bool:
        """Check if we've already entered at this level (within threshold)."""
        for entered_price, entered_type in self._entered_levels:
            threshold = level_price * self.entry_threshold
            if abs(entered_price - level_price) <= threshold and entered_type == trade_type:
                return True
        return False
    
    def _mark_level_entered(self, level_price: float, trade_type: str):
        """Mark a level as entered."""
        self._entered_levels.add((level_price, trade_type))
    
    def _clear_entered_level(self):
        """Clear entered level when position closes (allows re-entry later)."""
        self._entered_levels.clear()
    
    def _handle_exit(self, current_price: float, position):
        """Handle exit logic - calls child's get_exit_signal. Returns True if exit triggered."""
        if self.entry_support is None or self.target_resistance is None:
            return False
        
        exit_reason = self.get_exit_signal(
            current_price,
            position,
            self.entry_support,
            self.target_resistance
        )
        
        if exit_reason:
            self.log_message(f"[{self.get_strategy_name()}] Manual exit trigger: {exit_reason}")
            # Close position properly (works for both LONG and SHORT)
            self._close_all_positions()
            return True
        
        return False
    
    def _close_all_positions(self):
        """Close all positions - works for both LONG and SHORT."""
        ticker = self.parameters["Ticker"]
        position = self.get_position(ticker)
        
        if position is None or position.quantity == 0:
            return
        
        if position.quantity > 0:
            # LONG position - sell to close
            self.sell_all()
        else:
            # SHORT position - buy to cover
            order = self.create_order(
                asset=ticker,
                quantity=abs(position.quantity),
                side="buy"
            )
            self.submit_order(order)
    
    def _determine_exit_reason(self, exit_price: float) -> str:
        """Determine exit reason based on price vs TP/SL."""
        if self.target_resistance and self.entry_support:
            # Simple heuristic based on price relative to entry
            trade = self.trade_tracker.get_trade(self.current_trade_id)
            if trade:
                if exit_price >= trade.take_profit * 0.99:
                    return "TP"
                elif exit_price <= trade.stop_loss * 1.01:
                    return "SL"
        return "MANUAL"
    
    def get_position_sizing(self, entry_price: float, stop_loss: float, trade_type: str = "BUY") -> int:
        """
        Calculate position size based on risk percentage.
        Works for both LONG (BUY) and SHORT (SELL) positions.
        
        Risk = |Entry - Stop Loss| * Quantity
        Quantity = (Portfolio Value * Risk%) / |Entry - Stop Loss|
        """
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_percent
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        quantity = int(risk_amount / risk_per_share)
        
        # Ensure we can afford the position
        max_affordable = int(portfolio_value * 0.95 / entry_price)
        quantity = min(quantity, max_affordable)
        
        return max(0, quantity)
    
    def _should_reload_levels(self, current_date) -> bool:
        """Check if we need to reload key levels."""
        if self.last_levels_date is None:
            return True
        
        freq = self.recalc_frequency
        
        if freq == "daily":
            return current_date != self.last_levels_date
        elif freq == "weekly":
            days_since = (current_date - self.last_levels_date).days
            return days_since >= 7 or current_date.weekday() == 0
        else:  # "once"
            return False
    
    def _load_key_levels(self, as_of_datetime: datetime):
        """Load key levels using KeyLevels class."""
        ticker = self.parameters["Ticker"].symbol
        
        # Create cache key
        cache_key = f"{ticker}_{as_of_datetime.date()}_{self.recalc_frequency}"
        
        # Check cache
        if cache_key in BaseKeyLevelsStrategy._cached_levels:
            cached = BaseKeyLevelsStrategy._cached_levels[cache_key]
            self.merged_levels = cached['merged']
            self.support_levels = cached['support']
            self.resistance_levels = cached['resistance']
            return
        
        # Fetch fresh levels
        try:
            kl = KeyLevels(
                ticker=ticker,
                use_alpaca=False,
                as_of_date=as_of_datetime
            )
            
            kl.find_all_key_levels(timeframes=self.timeframes)
            self.merged_levels = kl.get_merged_levels(price_threshold=self.price_threshold)
            
            if self.merged_levels is not None and not self.merged_levels.empty:
                # Filter by importance
                filtered = self.merged_levels[
                    self.merged_levels['importance'] >= self.min_importance
                ]
                
                self.support_levels = filtered[filtered['type'] == 'support'].copy()
                self.resistance_levels = filtered[filtered['type'] == 'resistance'].copy()
                
                # Cache the results
                BaseKeyLevelsStrategy._cached_levels[cache_key] = {
                    'merged': self.merged_levels,
                    'support': self.support_levels,
                    'resistance': self.resistance_levels
                }
                
                # Log to levels history
                self._log_levels_to_history(as_of_datetime.date())
                
            else:
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()
                
        except Exception as e:
            self.log_message(f"Error loading key levels: {e}")
            self.support_levels = pd.DataFrame()
            self.resistance_levels = pd.DataFrame()
    
    def _log_levels_to_history(self, date):
        """Log current levels to history for visualization."""
        supports = []
        resistances = []
        
        if self.support_levels is not None and not self.support_levels.empty:
            for _, row in self.support_levels.iterrows():
                supports.append({
                    'price': round(float(row['level_price']), 2),
                    'importance': int(row['importance']),
                    'touch_count': int(row.get('touch_count', 1))
                })
        
        if self.resistance_levels is not None and not self.resistance_levels.empty:
            for _, row in self.resistance_levels.iterrows():
                resistances.append({
                    'price': round(float(row['level_price']), 2),
                    'importance': int(row['importance']),
                    'touch_count': int(row.get('touch_count', 1))
                })
        
        self._levels_history.append({
            'date': str(date),
            'supports': supports,
            'resistances': resistances
        })
    
    def _save_levels_history(self, filepath: str):
        """Save levels history to JSON file."""
        if not self._levels_history:
            return
        
        ticker = self.parameters["Ticker"].symbol
        data = {
            'ticker': ticker,
            'strategy': self.get_strategy_name(),
            'levels_count': len(self._levels_history),
            'levels': self._levels_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log_message(f"Levels history saved: {len(self._levels_history)} days to {os.path.basename(filepath)}")

