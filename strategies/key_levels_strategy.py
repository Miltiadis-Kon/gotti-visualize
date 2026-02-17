"""
Key Levels Trading Strategy

Trades based on key support and resistance levels from multi-timeframe analysis.
- Buys when price hits a support level with importance > 3
- Sells when price almost reaches the closest resistance level with importance > 3
- Stop loss at support level - threshold%
- Take profit at resistance level - threshold%

Uses the KeyLevels class to fetch support/resistance levels and Fibonacci patterns.
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.entities import Asset
from lumibot.traders import Trader
from dotenv import load_dotenv

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import key levels analysis from same directory
from key_levels import (
    KeyLevels,
    run_merged_key_levels,
    run_fibonacci_analysis
)

# Import trade tracker for recording trades
from trade_tracker import TradeTracker

load_dotenv()

apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

ALPACA_CONFIG = {
    "API_KEY": apikey,
    "API_SECRET": apisecret,
    "PAPER": True,  # Set to True for paper trading, False for live trading
}


class KeyLevelsStrategy(Strategy):
    """
    Key Levels Trading Strategy
    
    Trades based on support and resistance levels with importance > MIN_IMPORTANCE.
    
    Entry conditions:
    1. Price touches a support level with importance > MIN_IMPORTANCE
    2. Support price must be lower than target resistance price
    3. Risk/Reward ratio must be >= MIN_RISK_REWARD
    
    Exit: Take profit when price reaches threshold% before the nearest resistance,
          or stop loss when price drops threshold% below entry support.
    
    Parameters:
    -----------
    Ticker : Asset
        The stock/asset to trade
    MIN_IMPORTANCE : int
        Minimum importance level for support/resistance (default: 3)
    TP_THRESHOLD : float
        Take profit threshold as decimal (default: 0.02 = 2% before resistance)
    SL_THRESHOLD : float
        Stop loss threshold as decimal (default: 0.05 = 5% below support)
    MIN_RISK_REWARD : float
        Minimum risk/reward ratio for a trade (default: 1.5)
    SUPPORT_HIT_TOLERANCE : float
        Tolerance for considering price "at" a support level (default: 0.005 = 0.5%)
    """
    
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "MIN_IMPORTANCE": 3,       # Minimum importance for levels to trade on
        "TP_THRESHOLD": 0.02,      # Take profit at 2% before resistance (adjust as needed)
        "SL_THRESHOLD": 0.05,      # Stop loss at 5% below support (adjust as needed)
        "MIN_RISK_REWARD": 1.5,    # Minimum risk/reward ratio for a trade
        "SUPPORT_HIT_TOLERANCE": 0.005,  # 0.5% tolerance for support touch
        "RISK_PERCENT": 0.02,      # Risk 2% of portfolio per trade
        "TIMEFRAMES_LONGTERM": ['1d', '4h', '1h'],  # Long-term levels (calculated once per day)
        "TIMEFRAMES_SHORTTERM": ['15m', '5m'],      # Short-term levels (updated every iteration)
        "PRICE_THRESHOLD": 0.5,    # Price threshold for merging levels
        "RECALC_FREQUENCY": "weekly",  # How often to recalculate levels: 'daily', 'weekly', 'once'
        "Plot": False,
    }
    
    # CLASS-LEVEL CACHE for key levels (persists across backtesting iterations)
    _cached_levels = {}
    
    # CLASS-LEVEL storage for trade tracker (accessible after backtest)
    _last_trade_tracker = None
    
    ##### CORE FUNCTIONS #####
    
    def initialize(self):
        """Initialize the strategy."""
        self.sleeptime = "5M"  # Execute strategy every 5 minutes
        self.risk_percent = self.parameters["RISK_PERCENT"]
        self.min_importance = self.parameters["MIN_IMPORTANCE"]
        self.tp_threshold = self.parameters["TP_THRESHOLD"]
        self.sl_threshold = self.parameters["SL_THRESHOLD"]
        self.min_risk_reward = self.parameters["MIN_RISK_REWARD"]
        self.support_tolerance = self.parameters["SUPPORT_HIT_TOLERANCE"]
        
        # Storage for levels (will be loaded from cache or fetched)
        self.merged_levels = None
        self.support_levels = None
        self.resistance_levels = None
        self.fib_levels = None
        
        # Storage for short-term levels (updated every iteration)
        self.shortterm_support_levels = None
        self.shortterm_resistance_levels = None
        
        # Current trade info
        self.entry_support = None  # The support level where we entered
        self.target_resistance = None  # The resistance level to target
        
        # Track the last date we loaded levels for
        self.last_levels_date = None
        self.recalc_frequency = self.parameters.get("RECALC_FREQUENCY", "weekly")
        
        # Trade tracker for recording trades
        self.trade_tracker = TradeTracker(ticker=self.parameters["Ticker"].symbol)
        self.current_trade_id = None  # Track active trade ID
        
    def _get_cache_period(self, current_date):
        """Get the cache period based on RECALC_FREQUENCY."""
        freq = self.recalc_frequency
        
        if freq == "daily":
            # Recalculate every day
            return str(current_date)
        elif freq == "weekly":
            # Recalculate once per week (use ISO week number)
            year, week, _ = current_date.isocalendar()
            return f"{year}-W{week}"
        elif freq == "once":
            # Only calculate once (use static key)
            return "static"
        else:
            # Default to weekly
            year, week, _ = current_date.isocalendar()
            return f"{year}-W{week}"
        
    def _load_key_levels(self, as_of_date=None):
        """
        Fetch key levels using data available up to the specified date.
        
        This prevents look-ahead bias in backtesting by only using historical
        data that would have been available at that point in time.
        
        Parameters:
        -----------
        as_of_date : datetime
            Calculate levels as of this date. If None, uses current simulation date.
        """
        ticker_symbol = self.parameters["Ticker"].symbol
        
        # Get current simulation date if not provided
        if as_of_date is None:
            as_of_date = self.get_datetime()
        
        # Convert to date only for caching
        current_date = as_of_date.date() if hasattr(as_of_date, 'date') else as_of_date
        
        # Get cache period based on recalculation frequency
        cache_period = self._get_cache_period(current_date)
        cache_key = f"{ticker_symbol}_{self.min_importance}_{cache_period}"
        
        if cache_key in KeyLevelsStrategy._cached_levels:
            cached = KeyLevelsStrategy._cached_levels[cache_key]
            self.merged_levels = cached['merged_levels']
            self.support_levels = cached['support_levels']
            self.resistance_levels = cached['resistance_levels']
            self.fib_levels = cached['fib_levels']
            # Only log once per period change
            if self.last_levels_date != cache_period:
                self.log_message(f"Loaded cached long-term key levels for {ticker_symbol} (period: {cache_period})")
                self.last_levels_date = cache_period
            return
        
        self.log_message(f"Fetching long-term key levels for {ticker_symbol} as of {current_date}...")
        
        # Fetch merged key levels and Fibonacci patterns
        try:
            # Convert to datetime if it's a date
            if hasattr(as_of_date, 'date'):
                as_of_datetime = as_of_date
            else:
                from datetime import datetime as dt
                as_of_datetime = dt.combine(as_of_date, dt.min.time())
            
            kl = KeyLevels(
                ticker=ticker_symbol,
                tolerance_pct=0.5,
                pivot_lookback=10,
                use_alpaca=False,  # Use Yahoo for backtesting (faster)
                as_of_date=as_of_datetime  # Pass the simulation date!
            )
            
            # Get long-term levels only
            kl.find_all_key_levels(timeframes=self.parameters["TIMEFRAMES_LONGTERM"])
            
            # Get merged levels with importance
            self.merged_levels = kl.get_merged_levels(
                price_threshold=self.parameters["PRICE_THRESHOLD"]
            )
            
            # Calculate Fibonacci levels
            self.fib_levels = kl.calculate_fibonacci_levels(merged_df=self.merged_levels)
            
            if self.merged_levels is not None and not self.merged_levels.empty:
                # Filter by minimum importance (> MIN_IMPORTANCE, not >=)
                important_levels = self.merged_levels[
                    self.merged_levels['importance'] > self.min_importance
                ]
                
                # Separate support and resistance levels
                self.support_levels = important_levels[
                    important_levels['type'] == 'support'
                ].sort_values('level_price', ascending=False).reset_index(drop=True)
                
                self.resistance_levels = important_levels[
                    important_levels['type'] == 'resistance'
                ].sort_values('level_price', ascending=True).reset_index(drop=True)
                
                self.log_message(f"Found {len(self.support_levels)} support levels "
                               f"and {len(self.resistance_levels)} resistance levels "
                               f"with importance > {self.min_importance}")
                
                # Log the levels
                if not self.support_levels.empty:
                    self.log_message(f"Support levels: {self.support_levels['level_price'].tolist()}")
                if not self.resistance_levels.empty:
                    self.log_message(f"Resistance levels: {self.resistance_levels['level_price'].tolist()}")
            else:
                self.log_message("No key levels found!")
                self.support_levels = pd.DataFrame()
                self.resistance_levels = pd.DataFrame()
                
        except Exception as e:
            self.log_message(f"Error fetching key levels: {e}")
            import traceback
            traceback.print_exc()
            self.support_levels = pd.DataFrame()
            self.resistance_levels = pd.DataFrame()
        
        # Save to class-level cache for this specific period
        KeyLevelsStrategy._cached_levels[cache_key] = {
            'merged_levels': self.merged_levels,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'fib_levels': self.fib_levels
        }
        self.last_levels_date = cache_period
        self.log_message(f"Cached long-term key levels for {ticker_symbol} (period: {cache_period})")
        
    def _update_shortterm_levels(self):
        """
        Update short-term (5m, 15m) levels every iteration.
        
        This method fetches fresh levels for importance 1 and 2 timeframes (5m, 15m)
        to get the most recent support/resistance levels.
        """
        ticker_symbol = self.parameters["Ticker"].symbol
        current_date = self.get_datetime()
        
        try:
            # Convert to datetime if needed
            if hasattr(current_date, 'date'):
                as_of_datetime = current_date
            else:
                from datetime import datetime as dt
                as_of_datetime = dt.combine(current_date, dt.min.time())
            
            kl = KeyLevels(
                ticker=ticker_symbol,
                tolerance_pct=0.5,
                pivot_lookback=10,
                use_alpaca=False,
                as_of_date=as_of_datetime
            )
            
            # Get short-term levels only (importance 1 and 2: 5m and 15m)
            kl.find_all_key_levels(timeframes=self.parameters["TIMEFRAMES_SHORTTERM"])
            
            # Get merged levels
            merged_short = kl.get_merged_levels(
                price_threshold=self.parameters["PRICE_THRESHOLD"]
            )
            
            if merged_short is not None and not merged_short.empty:
                # Filter to importance 1, 2 (5m and 15m levels)
                shortterm_levels = merged_short[
                    merged_short['importance'].isin([1, 2])
                ]
                
                # Separate support and resistance
                self.shortterm_support_levels = shortterm_levels[
                    shortterm_levels['type'] == 'support'
                ].sort_values('level_price', ascending=False).reset_index(drop=True)
                
                self.shortterm_resistance_levels = shortterm_levels[
                    shortterm_levels['type'] == 'resistance'
                ].sort_values('level_price', ascending=True).reset_index(drop=True)
                
        except Exception as e:
            self.log_message(f"Error updating short-term levels: {e}")
            self.shortterm_support_levels = pd.DataFrame()
            self.shortterm_resistance_levels = pd.DataFrame()
        
    def before_market_opens(self):
        """Load long-term key levels for the current simulation date."""
        self._load_key_levels()
    
    def on_trading_iteration(self):
        """Main trading logic executed every sleeptime (5 minutes)."""
        # Update short-term levels (5m, 15m) every iteration
        self._update_shortterm_levels()
        
        # Combine long-term and short-term support/resistance levels
        all_support_levels = pd.concat([
            self.support_levels if self.support_levels is not None else pd.DataFrame(),
            self.shortterm_support_levels if self.shortterm_support_levels is not None else pd.DataFrame()
        ], ignore_index=True).drop_duplicates(subset=['level_price']).sort_values('level_price', ascending=False)
        
        all_resistance_levels = pd.concat([
            self.resistance_levels if self.resistance_levels is not None else pd.DataFrame(),
            self.shortterm_resistance_levels if self.shortterm_resistance_levels is not None else pd.DataFrame()
        ], ignore_index=True).drop_duplicates(subset=['level_price']).sort_values('level_price', ascending=True)
        
        # Check if we have valid levels
        if all_support_levels.empty or all_resistance_levels.empty:
            return
        
        # Get current price
        current_price = self.get_last_price(self.parameters["Ticker"])
        if current_price is None:
            self.log_message("Could not get current price")
            return
        
        # Check if we already have a position
        position = self.get_position(self.parameters["Ticker"])
        
        if position is None or position.quantity == 0:
            # No position - look for entry (use combined levels)
            self._check_for_entry(current_price, all_support_levels, all_resistance_levels)
        else:
            # Have position - check for exit
            self._check_for_exit(current_price, position)
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Handle filled orders."""
        if order.side == "buy":
            self.log_message(f"BUY order filled at ${price:.2f} - Quantity: {quantity}")
            if self.entry_support and self.target_resistance:
                self.log_message(f"Entry support: ${self.entry_support:.2f}, "
                               f"Target resistance: ${self.target_resistance:.2f}")
        elif order.side == "sell":
            self.log_message(f"SELL order filled at ${price:.2f} - Quantity: {quantity}")
            
            # Close trade in tracker
            if self.current_trade_id:
                # Determine exit reason based on price vs TP/SL
                if self.target_resistance and self.entry_support:
                    take_profit = self.get_take_profit(self.target_resistance)
                    stop_loss = self.get_stop_loss(self.entry_support)
                    
                    if price >= take_profit * 0.99:  # Within 1% of TP
                        exit_reason = "TP"
                    elif price <= stop_loss * 1.01:  # Within 1% of SL
                        exit_reason = "SL"
                    else:
                        exit_reason = "MANUAL"
                else:
                    exit_reason = "MANUAL"
                
                closed_trade = self.trade_tracker.close_trade(
                    trade_id=self.current_trade_id,
                    date=self.get_datetime(),
                    exit_price=price,
                    exit_reason=exit_reason
                )
                
                if closed_trade:
                    pnl_str = f"+${closed_trade.pnl:.2f}" if closed_trade.pnl >= 0 else f"-${abs(closed_trade.pnl):.2f}"
                    self.log_message(f"Trade closed ({exit_reason}): PnL = {pnl_str}")
                
                self.current_trade_id = None
            
            # Reset trade tracking
            self.entry_support = None
            self.target_resistance = None
    
    def after_market_closes(self):
        """
        Called at the end of each trading day.
        Saves trades to file so the final backtest state is always persisted.
        """
        # Save trade tracker to class-level for access after backtest
        KeyLevelsStrategy._last_trade_tracker = self.trade_tracker
        
        # Save trades to file (overwrites each day, final day has all trades)
        ticker = self.parameters["Ticker"].symbol
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Always save current state (will be overwritten each day)
        json_path = os.path.join(output_dir, f"{ticker}_trades.json")
        self.trade_tracker.save_to_json(json_path)
        
        csv_path = os.path.join(output_dir, f"{ticker}_trades.csv")
        self.trade_tracker.save_to_csv(csv_path)
        
        # Print summary on days with trades
        if self.trade_tracker.trades:
            current_date = self.get_datetime().strftime('%Y-%m-%d')
            self.log_message(f"[{current_date}] Trades saved to: {json_path}")
    
    def on_abrupt_closing(self):
        """Called when the strategy is stopped (crash or manual stop)."""
        # Final save on any abnormal termination
        self.after_market_closes()
        
        # Print final summary
        if self.trade_tracker.trades:
            self.trade_tracker.print_summary()
    
    ##### TRADING FUNCTIONS #####
    
    # Debug counter for periodic logging
    _debug_counter = 0
    
    def _check_for_entry(self, current_price: float, support_levels: pd.DataFrame, resistance_levels: pd.DataFrame):
        """
        Check if current price is at a support level and enter a trade.
        
        Entry conditions:
        1. Price is within tolerance of a support level
        2. Support price must be lower than target resistance price
        3. Try multiple resistance levels until R:R >= MIN_RISK_REWARD
        4. If no resistance meets R:R, use fixed 1.5x from entry for TP
        """
        # Periodic debug logging (every 100 iterations)
        KeyLevelsStrategy._debug_counter += 1
        if KeyLevelsStrategy._debug_counter % 100 == 1:
            if not support_levels.empty:
                closest_support = min(support_levels['level_price'], key=lambda x: abs(x - current_price))
                distance_pct = abs(current_price - closest_support) / closest_support * 100
                self.log_message(f"DEBUG: Price=${current_price:.2f}, Closest support=${closest_support:.2f}, "
                               f"Distance={distance_pct:.2f}%, Tolerance={self.support_tolerance*100:.2f}%")
        
        # Find if we're at any support level
        for _, support in support_levels.iterrows():
            support_price = support['level_price']
            tolerance = support_price * self.support_tolerance
            
            # Check if current price is at or slightly above support
            if abs(current_price - support_price) <= tolerance:
                # We're at a support level!
                self.log_message(f"Price ${current_price:.2f} hit support at ${support_price:.2f}")
                
                entry_price = current_price
                stop_loss = self.get_stop_loss(support_price)
                
                # Validate SL makes sense
                if stop_loss >= entry_price:
                    self.log_message(f"Invalid trade: Stop loss ${stop_loss:.2f} >= Entry ${entry_price:.2f}")
                    continue
                
                risk = entry_price - stop_loss
                if risk <= 0:
                    self.log_message(f"Invalid risk: ${risk:.2f}")
                    continue
                
                # Try to find a resistance level that meets R:R >= MIN_RISK_REWARD
                resistances_above = resistance_levels[
                    resistance_levels['level_price'] > current_price
                ].sort_values('level_price', ascending=True)  # Closest first
                
                target_resistance = None
                take_profit = None
                best_rr = 0
                
                for _, resistance in resistances_above.iterrows():
                    resistance_price = resistance['level_price']
                    
                    # SAFETY CHECK: Ensure support < resistance
                    if support_price >= resistance_price:
                        continue
                    
                    # Calculate TP for this resistance
                    tp_candidate = self.get_take_profit(resistance_price)
                    
                    # Validate TP > entry
                    if tp_candidate <= entry_price:
                        continue
                    
                    # Calculate R:R for this resistance
                    reward = tp_candidate - entry_price
                    rr = reward / risk
                    
                    if rr > best_rr:
                        best_rr = rr
                    
                    # Found a resistance that meets our criteria!
                    if rr >= self.min_risk_reward:
                        target_resistance = resistance_price
                        take_profit = tp_candidate
                        self.log_message(f"Found suitable resistance ${target_resistance:.2f} with R:R {rr:.2f}")
                        break
                
                # If no resistance met R:R criteria, use fixed 1.5x multiplier from entry
                if target_resistance is None:
                    self.log_message(f"No resistance met R:R {self.min_risk_reward}. Best was {best_rr:.2f}. "
                                   f"Using fixed 1.5x multiplier for TP.")
                    take_profit = entry_price + (risk * self.min_risk_reward)
                    target_resistance = take_profit / (1 - self.tp_threshold)  # Reverse calculate
                
                risk_reward_ratio = (take_profit - entry_price) / risk
                self.log_message(f"Risk/Reward ratio: {risk_reward_ratio:.2f} (min: {self.min_risk_reward})")
                
                # Calculate position size
                quantity = self.get_position_sizing(entry_price, stop_loss)
                
                if quantity <= 0:
                    self.log_message("Insufficient funds for trade")
                    return
                
                # Store trade info
                self.entry_support = support_price
                self.target_resistance = target_resistance
                
                # Record trade in tracker
                self.current_trade_id = self.trade_tracker.open_trade(
                    date=self.get_datetime(),
                    entry_price=entry_price,
                    quantity=quantity,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    support_level=support_price,
                    resistance_level=target_resistance,
                    trade_type="BUY"
                )
                
                # Create and submit buy order with bracket (SL/TP)
                self.log_message(f"Entering trade: BUY {quantity} @ ${entry_price:.2f}, "
                               f"SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}, R:R: {risk_reward_ratio:.2f}")
                
                order = self.create_order(
                    asset=self.parameters["Ticker"],
                    quantity=quantity,
                    side="buy",
                    take_profit_price=take_profit,
                    stop_loss_price=stop_loss,
                )
                
                self.submit_order(order)
                return  # Only enter one trade
    
    def _check_for_exit(self, current_price: float, position):
        """
        Check if we should exit an existing position.
        
        This is a backup check - the bracket order should handle exits,
        but we can manually exit if needed.
        """
        if self.entry_support is None or self.target_resistance is None:
            return
        
        take_profit = self.get_take_profit(self.target_resistance)
        stop_loss = self.get_stop_loss(self.entry_support)
        
        # Manual exit check (in case bracket order didn't trigger)
        if current_price >= take_profit:
            self.log_message(f"Manual TP trigger: Price ${current_price:.2f} >= TP ${take_profit:.2f}")
            self.sell_all()
        elif current_price <= stop_loss:
            self.log_message(f"Manual SL trigger: Price ${current_price:.2f} <= SL ${stop_loss:.2f}")
            self.sell_all()
    
    def get_stop_loss(self, entry_support: float) -> float:
        """
        Calculate stop loss price.
        
        Stop loss = support price - (support price * SL_THRESHOLD)
        
        Example: If support is $169.69 and SL_THRESHOLD is 0.1 (10%):
                Stop loss = 169.69 - (169.69 * 0.1) = 169.69 - 16.969 = $152.72
        
        Parameters:
        -----------
        entry_support : float
            The support level where we entered the trade
            
        Returns:
        --------
        float
            Stop loss price
        """
        stop_loss = entry_support * (1 - self.sl_threshold)
        return round(stop_loss, 2)
    
    def get_take_profit(self, target_resistance: float) -> float:
        """
        Calculate take profit price.
        
        Take profit = resistance - (TP_THRESHOLD * resistance)
        
        We exit "almost at" resistance, leaving a buffer before the level.
        
        Example: If resistance is $184.19 and TP_THRESHOLD is 0.02 (2%):
                Take profit = 184.19 * (1 - 0.02) = 184.19 * 0.98 = $180.51
                
        Note: Default TP_THRESHOLD of 0.1 (10%) may be too large for most trades.
        Consider using 0.02 (2%) for tighter exits near resistance.
        
        Parameters:
        -----------
        target_resistance : float
            The resistance level we're targeting
            
        Returns:
        --------
        float
            Take profit price (resistance minus threshold buffer)
        """
        take_profit = target_resistance * (1 - self.tp_threshold)
        return round(take_profit, 2)
    
    def get_position_sizing(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on risk percentage.
        
        Risk per share = entry_price - stop_loss
        Max loss = portfolio_value * risk_percent
        Position size = Max loss / Risk per share
        
        Parameters:
        -----------
        entry_price : float
            Expected entry price
        stop_loss : float
            Stop loss price
            
        Returns:
        --------
        int
            Number of shares to buy
        """
        portfolio_value = self.portfolio_value
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return 0
        
        max_loss = portfolio_value * self.risk_percent
        position_size = int(max_loss / risk_per_share)
        
        # Make sure we have enough cash
        max_affordable = int(self.cash / entry_price)
        position_size = min(position_size, max_affordable)
        
        return max(position_size, 0)


def run_live():
    """Run the strategy live with paper trading."""
    trader = Trader()
    broker = Alpaca(ALPACA_CONFIG)
    strategy = KeyLevelsStrategy(
        broker=broker,
        parameters={
            "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
            "MIN_IMPORTANCE": 3,
            "TP_THRESHOLD": 0.02,
            "SL_THRESHOLD": 0.05,
        }
    )
    
    trader.add_strategy(strategy)
    trader.run_all()


def run_backtest(ticker: str = "NVDA",
                 start_date: datetime = None,
                 end_date: datetime = None,
                 budget: float = 10000,
                 min_importance: int = 1,
                 tp_threshold: float = 0.04,
                 sl_threshold: float = 0.02,
                 min_risk_reward: float = 1.5):
    """
    Run a backtest of the Key Levels strategy.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to trade (default: "NVDA")
    start_date : datetime
        Backtest start date (default: 6 months ago)
    end_date : datetime
        Backtest end date (default: today)
    budget : float
        Starting capital (default: $10,000)
    min_importance : int
        Minimum importance for levels (default: 3)
    tp_threshold : float
        Take profit threshold (default: 0.02 = 2%)
    sl_threshold : float
        Stop loss threshold (default: 0.05 = 5%)
    min_risk_reward : float
        Minimum risk/reward ratio (default: 1.5)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=180)
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Running backtest for {ticker} from {start_date.date()} to {end_date.date()}")
    
    KeyLevelsStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        budget=budget,
        parameters={
            "Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK),
            "MIN_IMPORTANCE": min_importance,
            "TP_THRESHOLD": tp_threshold,
            "SL_THRESHOLD": sl_threshold,
            "MIN_RISK_REWARD": min_risk_reward,
        }
    )


if __name__ == "__main__":
    # Run backtest by default
    run_backtest(
        ticker="PLTR",
        start_date=datetime(2026, 1, 10),
        end_date=datetime(2026, 1, 20),
        budget=10000,
        min_importance=2,
        tp_threshold=0.02,
        sl_threshold=0.05,
        min_risk_reward=1.5
    )
    # Uncomment to run live:
    # run_live()
