"""
Signal-Based Key Levels Strategy (Multi-Ticker / Dynamic)

Trades based on historical signals from MySQL database combined with key levels.
- Fetches ALL signals for the current trading day from historical_signals table
- Supports both static tickers list AND dynamic tickers from database
- BUY signal: Buy at support level
- SELL signal: Sell (short) at resistance level
- Uses 1.5 risk-reward ratio
- Allows multiple trades at different levels per ticker

Database Connection:
- Host: localhost
- Port: 3306
- Database: gotti
- Table: historical_signals

Usage:
1. Static tickers: Set "Tickers" parameter to a list of Asset objects
2. Dynamic tickers: Set "DYNAMIC_TICKERS" to True - will trade any ticker with signals
3. Mixed: Set both - will trade static tickers + any additional from database
"""

import pandas as pd
import sys
import os
import mysql.connector
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set

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


# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "database": "gotti",
    "user": "root",
    "password": "1234"
}


class SignalKeyLevelsStrategy(Strategy):
    """
    Signal-based key levels trading strategy with multi-ticker support.
    
    Combines historical signals from database with key level detection.
    Supports both static ticker lists and dynamic tickers from database signals.
    
    Entry Logic:
    - BUY signal: Enter long when price hits a support level
    - SELL signal: Enter short when price hits a resistance level
    
    Exit Logic:
    - Take Profit: 1.5x the risk distance
    - Stop Loss: Below support (for longs) or above resistance (for shorts)
    
    Parameters:
    -----------
    Tickers : List[Asset]
        Static list of tickers to trade (optional if DYNAMIC_TICKERS=True)
    DYNAMIC_TICKERS : bool
        If True, automatically trade any ticker with signals in database
    RISK_REWARD : float
        Risk-reward ratio for trades (default: 1.5)
    MIN_IMPORTANCE : int
        Minimum importance level for key levels (default: 1 = include 5m levels)
    ENTRY_THRESHOLD : float
        Tolerance for considering price "at" a level (default: 0.01 = 1%)
    """
    
    parameters = {
        # Can be a single Asset or list of Assets or list of strings
        "Tickers": [],  # Empty = use only dynamic tickers from DB
        "DYNAMIC_TICKERS": True,         # Auto-trade tickers from database signals
        "RISK_PERCENT": 0.02,            # Risk 2% of portfolio per trade
        "RISK_REWARD": 1.5,              # Risk-reward ratio
        "MIN_IMPORTANCE": 1,             # Minimum level importance (1 = include 5m)
        "TIMEFRAMES": ['1d', '4h', '1h', '15m', '5m'],  # All timeframes
        "PRICE_THRESHOLD": 0.5,          # Price threshold for merging levels
        "RECALC_FREQUENCY": "daily",     # Recalculate levels daily
        "ENTRY_THRESHOLD": 0.01,         # 1% tolerance for entry
        "SL_BUFFER": 0.005,              # 0.5% buffer for stop loss
    }
    
    # Class-level cache for key levels
    _cached_levels = {}
    _last_trade_tracker = None
    
    def initialize(self):
        """Initialize the strategy."""
        self.sleeptime = "5M"  # Execute every 5 minutes
        
        # Load parameters
        self.risk_percent = self.parameters.get("RISK_PERCENT", 0.02)
        self.risk_reward = self.parameters.get("RISK_REWARD", 1.5)
        self.min_importance = self.parameters.get("MIN_IMPORTANCE", 1)
        self.timeframes = self.parameters.get("TIMEFRAMES", ['1d', '4h', '1h', '15m', '5m'])
        self.price_threshold = self.parameters.get("PRICE_THRESHOLD", 0.5)
        self.recalc_frequency = self.parameters.get("RECALC_FREQUENCY", "daily")
        self.entry_threshold = self.parameters.get("ENTRY_THRESHOLD", 0.01)
        self.sl_buffer = self.parameters.get("SL_BUFFER", 0.005)
        self.dynamic_tickers = self.parameters.get("DYNAMIC_TICKERS", True)
        
        # Database connection
        self._db_connection = None
        
        # Current day's signals from database: {ticker: signal_position}
        self._current_signals: Dict[str, str] = {}
        self._last_signal_date = None
        
        # Per-ticker data storage
        self._ticker_data: Dict[str, Dict] = {}  # {ticker: {support_levels, resistance_levels, entered_levels}}
        
        # Active tickers for today (from signals)
        self._active_tickers: Set[str] = set()
        
        # Trade tracker
        self.trade_tracker = TradeTracker(ticker="MULTI")
        self._current_trade_ids: Dict[str, int] = {}  # {ticker: trade_id}
        
        # Output filename
        real_datetime = datetime.now()
        date_str = real_datetime.strftime('%d%m')
        time_str = real_datetime.strftime('%H%M')
        self._output_filename = f"{date_str}_{time_str}_signalkeylevels"
        
        self.log_message(f"[SignalKeyLevels] Initialized - R:R={self.risk_reward}, Dynamic={self.dynamic_tickers}")
    
    def _get_db_connection(self):
        """Get or create database connection."""
        try:
            if self._db_connection is None or not self._db_connection.is_connected():
                self._db_connection = mysql.connector.connect(**DB_CONFIG)
            return self._db_connection
        except Exception as e:
            self.log_message(f"Database connection error: {e}")
            return None
    
    def _fetch_signals_for_date(self, date) -> Dict[str, str]:
        """
        Fetch all signals from database for a specific date.
        
        Returns dict: {ticker: signal_position}
        """
        conn = self._get_db_connection()
        if conn is None:
            return {}
        
        try:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT ticker, signal_position
                FROM historical_signals
                WHERE signal_date = %s
            """
            cursor.execute(query, (date,))
            rows = cursor.fetchall()
            cursor.close()
            
            signals = {row['ticker'].upper(): row['signal_position'] for row in rows}
            self.log_message(f"[SignalKeyLevels] Fetched {len(signals)} signals for {date}")
            
            for ticker, signal in signals.items():
                self.log_message(f"  {ticker}: {signal}")
            
            return signals
            
        except Exception as e:
            self.log_message(f"Error fetching signals: {e}")
            return {}
    
    def _get_static_tickers(self) -> List[str]:
        """Get list of static ticker symbols from parameters."""
        tickers_param = self.parameters.get("Tickers", [])
        
        if not tickers_param:
            return []
        
        # Handle single Asset
        if isinstance(tickers_param, Asset):
            return [tickers_param.symbol]
        
        # Handle list
        result = []
        for t in tickers_param:
            if isinstance(t, Asset):
                result.append(t.symbol)
            elif isinstance(t, str):
                result.append(t.upper())
        
        return result
    
    def _update_active_tickers(self, current_date):
        """Update active tickers based on signals and static list."""
        # Fetch signals for today
        self._current_signals = self._fetch_signals_for_date(current_date)
        
        # Start with static tickers
        static_tickers = self._get_static_tickers()
        
        # Build active tickers set
        self._active_tickers = set()
        
        if self.dynamic_tickers:
            # Add all tickers that have signals
            self._active_tickers.update(self._current_signals.keys())
        
        # Add static tickers (they may or may not have signals)
        self._active_tickers.update(static_tickers)
        
        # Initialize data for new tickers
        for ticker in self._active_tickers:
            if ticker not in self._ticker_data:
                self._ticker_data[ticker] = {
                    'support_levels': None,
                    'resistance_levels': None,
                    'entered_levels': set(),
                    'last_levels_date': None
                }
        
        self.log_message(f"[SignalKeyLevels] Active tickers: {sorted(self._active_tickers)}")
    
    def _load_key_levels_for_ticker(self, ticker: str, as_of_datetime: datetime):
        """Load key levels for a specific ticker."""
        cache_key = f"{ticker}_{as_of_datetime.date()}_{self.recalc_frequency}"
        
        # Check cache
        if cache_key in SignalKeyLevelsStrategy._cached_levels:
            cached = SignalKeyLevelsStrategy._cached_levels[cache_key]
            self._ticker_data[ticker]['support_levels'] = cached['support']
            self._ticker_data[ticker]['resistance_levels'] = cached['resistance']
            return
        
        try:
            kl = KeyLevels(
                ticker=ticker,
                use_alpaca=False,
                as_of_date=as_of_datetime
            )
            
            kl.find_all_key_levels(timeframes=self.timeframes)
            merged_levels = kl.get_merged_levels(price_threshold=self.price_threshold)
            
            if merged_levels is not None and not merged_levels.empty:
                filtered = merged_levels[merged_levels['importance'] >= self.min_importance]
                support_levels = filtered[filtered['type'] == 'support'].copy()
                resistance_levels = filtered[filtered['type'] == 'resistance'].copy()
                
                self._ticker_data[ticker]['support_levels'] = support_levels
                self._ticker_data[ticker]['resistance_levels'] = resistance_levels
                
                # Cache
                SignalKeyLevelsStrategy._cached_levels[cache_key] = {
                    'support': support_levels,
                    'resistance': resistance_levels
                }
                
                self.log_message(
                    f"[SignalKeyLevels] {ticker}: {len(support_levels)} supports, "
                    f"{len(resistance_levels)} resistances"
                )
            else:
                self._ticker_data[ticker]['support_levels'] = pd.DataFrame()
                self._ticker_data[ticker]['resistance_levels'] = pd.DataFrame()
                
        except Exception as e:
            self.log_message(f"Error loading key levels for {ticker}: {e}")
            self._ticker_data[ticker]['support_levels'] = pd.DataFrame()
            self._ticker_data[ticker]['resistance_levels'] = pd.DataFrame()
    
    def on_trading_iteration(self):
        """Main trading loop - processes all active tickers."""
        current_datetime = self.get_datetime()
        current_date = current_datetime.date()
        
        # Update signals and active tickers daily
        if self._last_signal_date != current_date:
            self._update_active_tickers(current_date)
            self._last_signal_date = current_date
            
            # Reset entered levels for new day
            for ticker in self._ticker_data:
                self._ticker_data[ticker]['entered_levels'] = set()
        
        # Process each active ticker
        for ticker in self._active_tickers:
            self._process_ticker(ticker, current_datetime)
    
    def _process_ticker(self, ticker: str, current_datetime: datetime):
        """Process trading logic for a single ticker."""
        current_date = current_datetime.date()
        
        # Get or create ticker data
        if ticker not in self._ticker_data:
            self._ticker_data[ticker] = {
                'support_levels': None,
                'resistance_levels': None,
                'entered_levels': set(),
                'last_levels_date': None
            }
        
        ticker_data = self._ticker_data[ticker]
        
        # Load key levels if needed
        if ticker_data['last_levels_date'] != current_date:
            self._load_key_levels_for_ticker(ticker, current_datetime)
            ticker_data['last_levels_date'] = current_date
        
        # Get signal for this ticker
        signal = self._current_signals.get(ticker)
        if signal is None:
            return  # No signal for this ticker today
        
        # Get current price
        asset = Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK)
        current_price = self.get_last_price(asset)
        if current_price is None:
            return
        
        # Check for entry
        self._check_entry(ticker, asset, current_price, signal, ticker_data)
    
    def _check_entry(self, ticker: str, asset: Asset, current_price: float, 
                     signal: str, ticker_data: Dict):
        """Check for entry conditions and execute trade."""
        support_levels = ticker_data['support_levels']
        resistance_levels = ticker_data['resistance_levels']
        
        if support_levels is None or support_levels.empty:
            return
        if resistance_levels is None or resistance_levels.empty:
            return
        
        if signal == "BUY":
            entry = self._check_long_entry(ticker, current_price, support_levels, 
                                           resistance_levels, ticker_data)
        elif signal == "SELL":
            entry = self._check_short_entry(ticker, current_price, support_levels,
                                            resistance_levels, ticker_data)
        else:
            return
        
        if entry is None:
            return
        
        # Execute the trade
        self._execute_trade(ticker, asset, entry)
    
    def _check_long_entry(self, ticker: str, current_price: float,
                          support_levels: pd.DataFrame, resistance_levels: pd.DataFrame,
                          ticker_data: Dict) -> Optional[Dict]:
        """Check for LONG entry at support level."""
        for _, support in support_levels.iterrows():
            support_price = support['level_price']
            
            # Check if price is at support
            threshold = current_price * self.entry_threshold
            if abs(current_price - support_price) > threshold:
                continue
            
            # Check if already entered at this level
            level_key = (round(support_price, 2), 'BUY')
            if level_key in ticker_data['entered_levels']:
                continue
            
            # Find closest resistance above
            resistances_above = resistance_levels[
                resistance_levels['level_price'] > current_price
            ].sort_values('level_price')
            
            if resistances_above.empty:
                continue
            
            target_resistance = resistances_above.iloc[0]['level_price']
            
            # Calculate SL and TP
            stop_loss = support_price * (1 - self.sl_buffer)
            risk = current_price - stop_loss
            
            if risk <= 0:
                continue
            
            take_profit = current_price + (risk * self.risk_reward)
            
            # Mark level as entered
            ticker_data['entered_levels'].add(level_key)
            
            return {
                'trade_type': 'BUY',
                'entry_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'support_level': support_price,
                'resistance_level': target_resistance
            }
        
        return None
    
    def _check_short_entry(self, ticker: str, current_price: float,
                           support_levels: pd.DataFrame, resistance_levels: pd.DataFrame,
                           ticker_data: Dict) -> Optional[Dict]:
        """Check for SHORT entry at resistance level."""
        for _, resistance in resistance_levels.iterrows():
            resistance_price = resistance['level_price']
            
            # Check if price is at resistance
            threshold = current_price * self.entry_threshold
            if abs(current_price - resistance_price) > threshold:
                continue
            
            # Check if already entered at this level
            level_key = (round(resistance_price, 2), 'SELL')
            if level_key in ticker_data['entered_levels']:
                continue
            
            # Find closest support below
            supports_below = support_levels[
                support_levels['level_price'] < current_price
            ].sort_values('level_price', ascending=False)
            
            if supports_below.empty:
                continue
            
            target_support = supports_below.iloc[0]['level_price']
            
            # Calculate SL and TP
            stop_loss = resistance_price * (1 + self.sl_buffer)
            risk = stop_loss - current_price
            
            if risk <= 0:
                continue
            
            take_profit = current_price - (risk * self.risk_reward)
            
            # Mark level as entered
            ticker_data['entered_levels'].add(level_key)
            
            return {
                'trade_type': 'SELL',
                'entry_price': current_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'support_level': target_support,
                'resistance_level': resistance_price
            }
        
        return None
    
    def _execute_trade(self, ticker: str, asset: Asset, entry: Dict):
        """Execute a trade based on entry signal."""
        trade_type = entry['trade_type']
        entry_price = entry['entry_price']
        take_profit = entry['take_profit']
        stop_loss = entry['stop_loss']
        
        # Calculate position size
        quantity = self._get_position_sizing(entry_price, stop_loss, trade_type)
        
        if quantity <= 0:
            self.log_message(f"[SignalKeyLevels] {ticker}: Insufficient funds")
            return
        
        # Record in trade tracker
        trade_id = self.trade_tracker.open_trade(
            date=self.get_datetime(),
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            support_level=entry['support_level'],
            resistance_level=entry['resistance_level'],
            trade_type=trade_type
        )
        self._current_trade_ids[ticker] = trade_id
        
        # Create and submit order
        side = "buy" if trade_type == "BUY" else "sell"
        
        self.log_message(
            f"[SignalKeyLevels] {ticker} {trade_type}: {quantity} @ ${entry_price:.2f}, "
            f"TP: ${take_profit:.2f}, SL: ${stop_loss:.2f}"
        )
        
        order = self.create_order(
            asset=asset,
            quantity=quantity,
            side=side,
            take_profit_price=take_profit,
            stop_loss_price=stop_loss,
        )
        self.submit_order(order)
    
    def _get_position_sizing(self, entry_price: float, stop_loss: float, 
                             trade_type: str = "BUY") -> int:
        """Calculate position size based on risk percentage."""
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_percent
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        quantity = int(risk_amount / risk_per_share)
        
        # Ensure we can afford it
        max_affordable = int(portfolio_value * 0.95 / entry_price)
        quantity = min(quantity, max_affordable)
        
        return max(0, quantity)
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Handle filled orders."""
        ticker = order.asset.symbol
        
        if order.side == "buy":
            self.log_message(f"[SignalKeyLevels] {ticker} BUY filled: {quantity} @ ${price:.2f}")
        elif order.side == "sell":
            self.log_message(f"[SignalKeyLevels] {ticker} SELL filled: {quantity} @ ${price:.2f}")
            
            # Close trade in tracker if it's an exit
            if ticker in self._current_trade_ids:
                trade_id = self._current_trade_ids[ticker]
                self.trade_tracker.close_trade(
                    trade_id=trade_id,
                    date=self.get_datetime(),
                    exit_price=price,
                    exit_reason="FILLED"
                )
                del self._current_trade_ids[ticker]
    
    def after_market_closes(self):
        """Save trades at end of day."""
        SignalKeyLevelsStrategy._last_trade_tracker = self.trade_tracker
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, f"{self._output_filename}.json")
        self.trade_tracker.save_to_json(json_path)
        
        csv_path = os.path.join(output_dir, f"{self._output_filename}.csv")
        self.trade_tracker.save_to_csv(csv_path)
    
    def on_abrupt_closing(self):
        """Clean up on strategy end."""
        if self._db_connection and self._db_connection.is_connected():
            self._db_connection.close()
        
        self.after_market_closes()
        
        if self.trade_tracker.trades:
            self.trade_tracker.print_summary()


# ==================== BACKTEST RUNNER ====================

def run_backtest(
    tickers: List[str] = None,
    start_date: datetime = None,
    end_date: datetime = None,
    budget: float = 100000,
    dynamic_tickers: bool = True
):
    """
    Run backtest for the Signal Key Levels Strategy.
    
    Parameters:
    -----------
    tickers : List[str]
        Static list of ticker symbols to trade (optional if dynamic_tickers=True)
    start_date : datetime
        Backtest start date
    end_date : datetime
        Backtest end date
    budget : float
        Initial portfolio value
    dynamic_tickers : bool
        If True, automatically trade any ticker with signals in database
    
    Examples:
    ---------
    # Trade only signals from database (fully dynamic)
    run_backtest(dynamic_tickers=True)
    
    # Trade specific tickers only
    run_backtest(tickers=['AAPL', 'NVDA'], dynamic_tickers=False)
    
    # Trade specific tickers + any additional from database
    run_backtest(tickers=['AAPL'], dynamic_tickers=True)
    """
    from lumibot.backtesting import YahooDataBacktesting
    
    if start_date is None:
        start_date = datetime(2022, 1, 1)
    if end_date is None:
        end_date = datetime(2024, 12, 31)
    
    # Convert ticker strings to Asset objects
    ticker_assets = []
    if tickers:
        ticker_assets = [Asset(symbol=t, asset_type=Asset.AssetType.STOCK) for t in tickers]
    
    # Configure strategy parameters
    SignalKeyLevelsStrategy.parameters["Tickers"] = ticker_assets
    SignalKeyLevelsStrategy.parameters["DYNAMIC_TICKERS"] = dynamic_tickers
    
    ticker_str = ", ".join(tickers) if tickers else "Dynamic from DB"
    
    print(f"\n{'='*60}")
    print(f"SIGNAL KEY LEVELS BACKTEST (Multi-Ticker)")
    print(f"Tickers: {ticker_str}")
    print(f"Dynamic Tickers: {dynamic_tickers}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Budget: ${budget:,.2f}")
    print(f"{'='*60}\n")
    
    # Run backtest
    results = SignalKeyLevelsStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        budget=budget,
        show_plot=True,
        show_tearsheet=True,
        save_tearsheet=True,
    )
    
    # Print trade summary if available
    if SignalKeyLevelsStrategy._last_trade_tracker:
        print("\n" + "="*60)
        print("TRADE SUMMARY")
        print("="*60)
        SignalKeyLevelsStrategy._last_trade_tracker.print_summary()
    
    return results


def run_live(tickers: List[str] = None, dynamic_tickers: bool = True):
    """
    Run live trading with the Signal Key Levels Strategy.
    
    Parameters:
    -----------
    tickers : List[str]
        Static list of ticker symbols (optional if dynamic_tickers=True)
    dynamic_tickers : bool
        If True, automatically trade any ticker with signals in database
    """
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Alpaca configuration
    ALPACA_CONFIG = {
        "API_KEY": os.getenv("APCA_API_KEY_PAPER"),
        "API_SECRET": os.getenv("APCA_API_SECRET_KEY_PAPER"),
        "PAPER": True,
    }
    
    # Convert ticker strings to Asset objects
    ticker_assets = []
    if tickers:
        ticker_assets = [Asset(symbol=t, asset_type=Asset.AssetType.STOCK) for t in tickers]
    
    # Configure strategy
    SignalKeyLevelsStrategy.parameters["Tickers"] = ticker_assets
    SignalKeyLevelsStrategy.parameters["DYNAMIC_TICKERS"] = dynamic_tickers
    
    # Create broker and trader
    broker = Alpaca(ALPACA_CONFIG)
    strategy = SignalKeyLevelsStrategy(broker=broker)
    trader = Trader()
    trader.add_strategy(strategy)
    
    ticker_str = ", ".join(tickers) if tickers else "Dynamic from DB"
    
    print(f"\n{'='*60}")
    print(f"SIGNAL KEY LEVELS LIVE TRADING (Multi-Ticker)")
    print(f"Tickers: {ticker_str}")
    print(f"Dynamic Tickers: {dynamic_tickers}")
    print(f"{'='*60}\n")
    
    trader.run_all()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Key Levels Strategy (Multi-Ticker)")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--tickers", nargs="+", default=None, help="List of ticker symbols")
    parser.add_argument("--dynamic", action="store_true", default=True,
                        help="Trade tickers dynamically from database signals")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false",
                        help="Only trade specified tickers")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--budget", type=float, default=100000)
    
    args = parser.parse_args()
    
    if args.mode == "backtest":
        start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else datetime(2022, 1, 1)
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime(2024, 12, 31)
        run_backtest(args.tickers, start, end, args.budget, args.dynamic)
    else:
        run_live(args.tickers, args.dynamic)
