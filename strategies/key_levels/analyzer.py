"""
Key Levels Analyzer - Data Fetcher & Orchestrator

Fetches candlestick data from Yahoo Finance or Alpaca, then runs both
key levels and Fibonacci analysis.

Usage:
    from strategies.key_levels import analyze
    
    # Simple usage
    result = analyze("NVDA")
    
    # Access results
    result.key_levels      # DataFrame of support/resistance levels
    result.fibonacci       # DataFrame of Fibonacci levels
    result.trade_setups    # DataFrame of trade setups
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass

from .key_levels import find_key_levels, merge_key_levels, KeyLevelDetector
from .fibonacci_levels import find_fibonacci_levels, get_fibonacci_trade_setups, FibonacciDetector


# Resolution to interval mapping for data fetching
RESOLUTION_CONFIG = {
    '1D': {'interval': '1d', 'days': 150, 'label': 'Daily'},
    '4H': {'interval': '4h', 'days': 90, 'label': '4 Hour'},
    '1H': {'interval': '1h', 'days': 30, 'label': '1 Hour'},
    '15m': {'interval': '15m', 'days': 7, 'label': '15 Min'},
    '5m': {'interval': '5m', 'days': 2, 'label': '5 Min'}
}


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    ticker: str
    key_levels: pd.DataFrame
    merged_levels: pd.DataFrame
    fibonacci: pd.DataFrame
    trade_setups: pd.DataFrame
    candle_data: Dict[str, pd.DataFrame]  # {resolution: df}


class DataFetcher:
    """
    Fetches candlestick data from Yahoo Finance or Alpaca.
    """
    
    def __init__(self, use_alpaca: bool = True, as_of_date: datetime = None):
        """
        Initialize data fetcher.
        
        Parameters:
        -----------
        use_alpaca : bool
            Try Alpaca first (falls back to Yahoo if unavailable)
        as_of_date : datetime
            Fetch data up to this date (for backtesting). None = current date.
        """
        self.use_alpaca = use_alpaca
        self.as_of_date = as_of_date
        self.alpaca_client = None
        
        if use_alpaca:
            self._init_alpaca()
    
    def _init_alpaca(self):
        """Initialize Alpaca client if available."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            api_key = os.getenv("APCA_API_KEY_PAPER")
            api_secret = os.getenv("APCA_API_SECRET_KEY_PAPER")
            if api_key and api_secret:
                self.alpaca_client = StockHistoricalDataClient(api_key, api_secret)
        except ImportError:
            self.alpaca_client = None
    
    def fetch(self, ticker: str, interval: str = '1d', 
              days_back: int = 150) -> pd.DataFrame:
        """
        Fetch candlestick data for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        interval : str
            Data interval: '1d', '4h', '1h', '15m', '5m'
        days_back : int
            Number of days to fetch
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data with columns: date, open, high, low, close, volume
        """
        end_date = self.as_of_date if self.as_of_date else datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Try Alpaca first
        if self.alpaca_client:
            df = self._fetch_alpaca(ticker, interval, start_date, end_date)
            if not df.empty:
                return df
        
        # Fall back to Yahoo
        return self._fetch_yahoo(ticker, interval, start_date, end_date)
    
    def _fetch_alpaca(self, ticker: str, interval: str, 
                      start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch from Alpaca API."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            from alpaca.data.enums import DataFeed
            
            # Map interval to Alpaca TimeFrame
            timeframe_map = {
                '1d': TimeFrame.Day,
                '4h': TimeFrame(4, TimeFrameUnit.Hour),
                '1h': TimeFrame.Hour,
                '15m': TimeFrame(15, TimeFrameUnit.Minute),
                '5m': TimeFrame(5, TimeFrameUnit.Minute)
            }
            
            timeframe = timeframe_map.get(interval, TimeFrame.Day)
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX
            )
            
            bars = self.alpaca_client.get_stock_bars(request)
            
            if not bars.data or ticker not in bars.data:
                return pd.DataFrame()
            
            df = bars.df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'date'})
            
            if 'volume' in df.columns:
                df = df[df['volume'] != 0]
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"Alpaca error for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_yahoo(self, ticker: str, interval: str,
                     start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch from Yahoo Finance."""
        try:
            import yfinance as yf
            
            # Adjust interval for Yahoo
            yf_interval = interval
            
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                prepost=True  # Include extended hours
            )
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            df = df[df['volume'] != 0]
            df = df.reset_index()
            
            # Rename datetime column
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            elif 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'date'})
            
            return df
            
        except Exception as e:
            print(f"Yahoo error for {ticker}: {e}")
            return pd.DataFrame()


class KeyLevelAnalyzer:
    """
    Main analyzer that combines data fetching with key level and Fibonacci analysis.
    """
    
    def __init__(self, 
                 use_alpaca: bool = True,
                 as_of_date: datetime = None,
                 tolerance_pct: float = 0.5,
                 pivot_lookback: int = 10,
                 price_threshold: float = 0.5):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        use_alpaca : bool
            Use Alpaca API (falls back to Yahoo)
        as_of_date : datetime
            Analyze as of this date (for backtesting)
        tolerance_pct : float
            Tolerance for clustering key levels
        pivot_lookback : int
            Bars before/after for pivot detection
        price_threshold : float
            Price threshold for merging levels ($)
        """
        self.data_fetcher = DataFetcher(use_alpaca=use_alpaca, as_of_date=as_of_date)
        self.key_level_detector = KeyLevelDetector(
            tolerance_pct=tolerance_pct,
            pivot_lookback=pivot_lookback
        )
        self.fibonacci_detector = FibonacciDetector()
        self.price_threshold = price_threshold
    
    def analyze(self, ticker: str,
                resolutions: List[str] = None,
                days_back: Dict[str, int] = None) -> AnalysisResult:
        """
        Run full analysis on a ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol to analyze
        resolutions : List[str]
            List of resolutions to analyze (default: ['1D', '4H', '1H', '15m', '5m'])
        days_back : Dict[str, int]
            Override days_back for each resolution
            
        Returns:
        --------
        AnalysisResult
            Container with key_levels, fibonacci, trade_setups, and candle_data
        """
        if resolutions is None:
            resolutions = ['1D', '4H', '1H', '15m', '5m']
        
        print(f"\n{'='*60}")
        print(f"ANALYZING {ticker}")
        print(f"{'='*60}")
        
        all_key_levels = []
        all_fibonacci = []
        candle_data = {}
        
        for resolution in resolutions:
            config = RESOLUTION_CONFIG.get(resolution, RESOLUTION_CONFIG['1D'])
            interval = config['interval']
            default_days = config['days']
            
            # Get days_back for this resolution
            res_days_back = default_days
            if days_back and resolution in days_back:
                res_days_back = days_back[resolution]
            
            print(f"\n--- {config['label']} ({resolution}) ---")
            
            # Fetch data
            df = self.data_fetcher.fetch(ticker, interval=interval, days_back=res_days_back)
            
            if df.empty:
                print(f"  No data available")
                continue
            
            print(f"  Fetched {len(df)} candles")
            candle_data[resolution] = df
            
            # Find key levels
            levels = self.key_level_detector.find_key_levels(
                df, resolution=resolution, days_back=res_days_back
            )
            if not levels.empty:
                all_key_levels.append(levels)
                print(f"  Found {len(levels)} key levels")
            
            # Find Fibonacci levels for every resolution
            fib = self.fibonacci_detector.find_fibonacci_levels(
                df, resolution=resolution, days_back=res_days_back
            )
            if not fib.empty:
                all_fibonacci.append(fib)
                patterns = fib['pattern_id'].nunique()
                print(f"  Found {patterns} Fibonacci patterns")
        
        # Combine and merge key levels
        if all_key_levels:
            combined_levels = pd.concat(all_key_levels, ignore_index=True)
            merged_levels = merge_key_levels(combined_levels, self.price_threshold)
        else:
            combined_levels = pd.DataFrame()
            merged_levels = pd.DataFrame()
        
        # Combine Fibonacci
        if all_fibonacci:
            combined_fib = pd.concat(all_fibonacci, ignore_index=True)
        else:
            combined_fib = pd.DataFrame()
        
        # Get trade setups from Fibonacci – latest pattern per resolution
        all_trade_setups = []
        for res_key in resolutions:
            if res_key in candle_data and not candle_data[res_key].empty:
                res_setups = self.fibonacci_detector.get_trade_setups(
                    candle_data[res_key], resolution=res_key
                )
                if not res_setups.empty:
                    # Keep only the latest (most recent) pattern per resolution
                    latest = res_setups.iloc[[-1]]
                    all_trade_setups.append(latest)
        
        trade_setups = (pd.concat(all_trade_setups, ignore_index=True)
                        if all_trade_setups else pd.DataFrame())
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {ticker}")
        print(f"{'='*60}")
        print(f"Total Key Levels: {len(combined_levels)}")
        print(f"Merged Key Levels: {len(merged_levels)}")
        print(f"Fibonacci Patterns: {combined_fib['pattern_id'].nunique() if not combined_fib.empty else 0}")
        print(f"Trade Setups: {len(trade_setups)}")
        
        return AnalysisResult(
            ticker=ticker,
            key_levels=combined_levels,
            merged_levels=merged_levels,
            fibonacci=combined_fib,
            trade_setups=trade_setups,
            candle_data=candle_data
        )
    
    def analyze_single_resolution(self, ticker: str,
                                  resolution: str = '1D',
                                  days_back: int = None) -> AnalysisResult:
        """
        Analyze a single resolution.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        resolution : str
            Resolution to analyze (e.g., '1D', '4H')
        days_back : int
            Days to look back (uses default if None)
        """
        days_override = {resolution: days_back} if days_back else None
        return self.analyze(ticker, resolutions=[resolution], days_back=days_override)


def analyze(ticker: str,
            resolutions: List[str] = None,
            use_alpaca: bool = True,
            as_of_date: datetime = None,
            days_back: Dict[str, int] = None) -> AnalysisResult:
    """
    Convenience function to run full analysis on a ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to analyze
    resolutions : List[str]
        Resolutions to analyze (default: all)
    use_alpaca : bool
        Use Alpaca API (default True)
    as_of_date : datetime
        Analyze as of this date (for backtesting)
    days_back : Dict[str, int]
        Override days_back for each resolution
        
    Returns:
    --------
    AnalysisResult
        Container with key_levels, fibonacci, trade_setups
        
    Example:
    --------
    >>> result = analyze("NVDA")
    >>> print(result.merged_levels)
    >>> print(result.trade_setups)
    """
    analyzer = KeyLevelAnalyzer(use_alpaca=use_alpaca, as_of_date=as_of_date)
    return analyzer.analyze(ticker, resolutions=resolutions, days_back=days_back)


def quick_analyze(ticker: str, resolution: str = '1D', 
                  days_back: int = 150) -> AnalysisResult:
    """
    Quick analysis for a single resolution.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol
    resolution : str
        Single resolution (e.g., '1D', '4H', '1H')
    days_back : int
        Days to look back
        
    Returns:
    --------
    AnalysisResult
    """
    analyzer = KeyLevelAnalyzer()
    return analyzer.analyze_single_resolution(ticker, resolution, days_back)


if __name__ == "__main__":
    # Example usage
    result = analyze("NVDA", resolutions=['1D', '4H', '15m'])
    
    print("\n--- MERGED KEY LEVELS ---")
    print(result.merged_levels)
    
    print("\n--- TRADE SETUPS ---")
    print(result.trade_setups)
