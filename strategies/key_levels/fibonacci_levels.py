"""
Fibonacci Levels Detection Module

Calculates Fibonacci retracement levels using swing detection algorithm 
from the Fibonacci Strategy notebook.

Algorithm:
    1. Find swing highs and lows within a lookback period
    2. Determine trend direction (min before max = uptrend)
    3. Calculate Fibonacci retracement levels for entry, stop loss, and take profit

For uptrend (retracement from high):
    entry = max_price - 0.62 * price_diff  
    stop_loss = max_price - 0.78 * price_diff
    take_profit = max_price (0% retracement)
    
For downtrend (retracement from low):
    entry = min_price + 0.62 * price_diff
    stop_loss = min_price + 0.78 * price_diff  
    take_profit = min_price (0% retracement)

Input:
    - df: DataFrame with OHLCV candle data (columns: open, high, low, close, volume)
    - resolution: Timeframe label (e.g., '1D', '4H', '1H', '15m', '5m')
    - days_back: Number of days to look back for swing detection
    
Output:
    - DataFrame with Fibonacci levels and trade setup info
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# Fibonacci configuration
FIBONACCI_LEVELS = [0.0, 0.236, 0.382, 0.50, 0.618, 0.786, 1.0]
FIBONACCI_ENTRY_LEVEL = 0.618     # Entry level
FIBONACCI_SL_LEVEL = 0.786        # Stop loss level

# Column names for each fib level in trade setups
FIB_LEVEL_COLUMNS = {
    0.0: 'fib_0',
    0.236: 'fib_236',
    0.382: 'fib_382',
    0.50: 'fib_500',
    0.618: 'fib_618',
    0.786: 'fib_786',
    1.0: 'fib_1000'
}
DEFAULT_BACKCANDLES = 40          # Lookback period for swing detection
DEFAULT_GAP_CANDLES = 5           # Gap before current candle
DEFAULT_PRICE_DIFF_PCT = 0.01     # Minimum 1% price difference for valid swing


@dataclass
class FibonacciPattern:
    """Represents a Fibonacci retracement pattern."""
    low_price: float
    high_price: float
    trend: str  # 'uptrend' or 'downtrend'
    entry_price: float
    stop_loss: float
    take_profit: float
    range_pct: float
    bar_index: int


class FibonacciDetector:
    """
    Detects Fibonacci retracement levels using swing detection.
    
    Parameters:
    -----------
    backcandles : int
        Number of candles to look back for swing detection (default 40)
    gap_candles : int
        Gap before current candle (default 5)
    min_price_diff_pct : float
        Minimum price difference percentage for valid swings (default 0.01 = 1%)
    """
    
    def __init__(self, 
                 backcandles: int = DEFAULT_BACKCANDLES,
                 gap_candles: int = DEFAULT_GAP_CANDLES,
                 min_price_diff_pct: float = DEFAULT_PRICE_DIFF_PCT):
        self.backcandles = backcandles
        self.gap_candles = gap_candles
        self.min_price_diff_pct = min_price_diff_pct
    
    def _find_swings(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find swing high/low patterns from price data (notebook-style).
        
        Looks at a lookback period and identifies the max and min prices,
        determining trend direction based on which came first.
        
        Returns:
        --------
        List of swing dictionaries with: max_price, min_price, index_max, 
        index_min, price_diff, trend, bar_index
        """
        swings = []
        
        for l in range(self.backcandles, len(df)):
            # Get the range excluding recent gap candles
            high_range = df['high'].iloc[l - self.backcandles:l - self.gap_candles]
            low_range = df['low'].iloc[l - self.backcandles:l - self.gap_candles]
            
            if len(high_range) == 0 or len(low_range) == 0:
                continue
            
            max_price = high_range.max()
            min_price = low_range.min()
            index_max = high_range.idxmax()
            index_min = low_range.idxmin()
            price_diff = max_price - min_price
            
            # Determine trend direction based on which extreme came first
            # min before max = uptrend (price went up)
            # max before min = downtrend (price went down)
            trend = 'uptrend' if index_min < index_max else 'downtrend'
            
            swings.append({
                'max_price': max_price,
                'min_price': min_price,
                'index_max': index_max,
                'index_min': index_min,
                'price_diff': price_diff,
                'trend': trend,
                'bar_index': l
            })
        
        return swings
    
    def _calculate_trade_levels(self, swing: Dict, trend: str) -> Dict:
        """
        Calculate entry, stop loss, take profit, and all Fibonacci price levels for a swing.
        
        Parameters:
        -----------
        swing : Dict
            Swing data with max_price, min_price, price_diff
        trend : str
            'uptrend' or 'downtrend'
            
        Returns:
        --------
        Dict with entry_price, stop_loss, take_profit, and fib_0..fib_1000
        """
        max_price = swing['max_price']
        min_price = swing['min_price']
        price_diff = swing['price_diff']
        
        if trend == 'uptrend':
            # Uptrend: retracement from the high
            entry_price = round(max_price - FIBONACCI_ENTRY_LEVEL * price_diff, 4)
            stop_loss = round(max_price - FIBONACCI_SL_LEVEL * price_diff, 4)
            take_profit = round(max_price, 4)  # 0% retracement = the high
        else:
            # Downtrend: retracement from the low
            entry_price = round(min_price + FIBONACCI_ENTRY_LEVEL * price_diff, 4)
            stop_loss = round(min_price + FIBONACCI_SL_LEVEL * price_diff, 4)
            take_profit = round(min_price, 4)  # 0% retracement = the low
        
        result = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        # Add all Fibonacci level prices as individual columns
        for fib_ratio in FIBONACCI_LEVELS:
            col_name = FIB_LEVEL_COLUMNS[fib_ratio]
            if trend == 'uptrend':
                result[col_name] = round(max_price - fib_ratio * price_diff, 4)
            else:
                result[col_name] = round(min_price + fib_ratio * price_diff, 4)
        
        return result
    
    def find_fibonacci_levels(self, df: pd.DataFrame, 
                              resolution: str = '1D',
                              days_back: int = None) -> pd.DataFrame:
        """
        Find Fibonacci retracement levels from price data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV candle data with columns: open, high, low, close, volume
        resolution : str
            Timeframe label (e.g., '1D', '4H', '1H', '15m', '5m')
        days_back : int
            Number of days to look back (None = use all data)
            
        Returns:
        --------
        pd.DataFrame
            Fibonacci levels with columns: fib_price, fib_level, trend, 
            entry_price, stop_loss, take_profit, pattern_id, etc.
        """
        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Filter by days_back if specified
        if days_back is not None and days_back > 0:
            date_col = None
            for col in ['date', 'datetime', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                date_col = df.columns[0]
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                cutoff = df[date_col].max() - pd.Timedelta(days=days_back)
                df = df[df[date_col] >= cutoff]
        
        if df.empty:
            return pd.DataFrame(columns=['fib_price', 'fib_level', 'fib_ratio',
                                         'low_price', 'high_price', 'trend',
                                         'level_type', 'entry_price', 'stop_loss', 
                                         'take_profit', 'range_pct', 'pattern_id',
                                         'resolution'])
        
        df = df.reset_index(drop=True)
        
        # Find swings
        swings = self._find_swings(df)
        
        # Process swings and generate Fibonacci levels
        fibonacci_levels = []
        used_pairs = set()
        
        for swing in swings:
            max_price = swing['max_price']
            min_price = swing['min_price']
            price_diff = swing['price_diff']
            trend = swing['trend']
            
            # Check minimum price difference threshold
            pct_diff = price_diff / min_price if min_price > 0 else 0
            if pct_diff < self.min_price_diff_pct:
                continue
            
            # Create unique pair key to avoid duplicates
            pair_key = (round(min_price, 2), round(max_price, 2))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            
            # Pattern ID and scoring
            pattern_id = f"${min_price:.2f}-${max_price:.2f}"            
            # Calculate trade levels
            trade_levels = self._calculate_trade_levels(swing, trend)
            
            # Generate all Fibonacci levels
            for fib_ratio in FIBONACCI_LEVELS:
                if trend == 'uptrend':
                    fib_price = round(max_price - fib_ratio * price_diff, 4)
                else:
                    fib_price = round(min_price + fib_ratio * price_diff, 4)
                
                # Determine level type
                if abs(fib_ratio - FIBONACCI_ENTRY_LEVEL) < 0.01:
                    level_type = 'entry'
                elif abs(fib_ratio - FIBONACCI_SL_LEVEL) < 0.01 or fib_ratio == FIBONACCI_SL_LEVEL:
                    level_type = 'stop_loss'
                elif fib_ratio == 0.0:
                    level_type = 'take_profit'
                else:
                    level_type = 'retracement'
                
                row_data = {
                    'fib_price': fib_price,
                    'fib_level': f"{fib_ratio * 100:.1f}%",
                    'fib_ratio': fib_ratio,
                    'low_price': min_price,
                    'high_price': max_price,
                    'trend': trend,
                    'level_type': level_type,
                    'entry_price': trade_levels['entry_price'],
                    'stop_loss': trade_levels['stop_loss'],
                    'take_profit': trade_levels['take_profit'],
                    'range_pct': round(pct_diff * 100, 2),
                    'pattern_id': pattern_id,
                    'resolution': resolution
                }
                # Add individual fib level columns
                for col_name in FIB_LEVEL_COLUMNS.values():
                    row_data[col_name] = trade_levels[col_name]
                
                fibonacci_levels.append(row_data)
        
        if not fibonacci_levels:
            return pd.DataFrame(columns=['fib_price', 'fib_level', 'fib_ratio',
                                         'low_price', 'high_price', 'trend',
                                         'level_type', 'entry_price', 'stop_loss',
                                         'take_profit', 'range_pct', 'pattern_id',
                                          'resolution'])
        
        fib_df = pd.DataFrame(fibonacci_levels)
        
        # Sort by pattern score (descending) then fib price
        fib_df = fib_df.sort_values(
            ['range_pct', 'fib_price'],
            ascending=[False, False]
        ).reset_index(drop=True)
        
        # Rank patterns
        unique_patterns = fib_df['pattern_id'].unique()
        rank_map = {pid: i + 1 for i, pid in enumerate(unique_patterns)}
        fib_df['pattern_rank'] = fib_df['pattern_id'].map(rank_map)
        
        return fib_df
    
    def get_trade_setups(self, df: pd.DataFrame,
                         resolution: str = '1D',
                         days_back: int = None,
                         trend_filter: str = None) -> pd.DataFrame:
        """
        Get simplified Fibonacci trade setups (entry, SL, TP) for each pattern.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV candle data
        resolution : str
            Timeframe label
        days_back : int
            Number of days to look back
        trend_filter : str
            Filter by trend: 'uptrend', 'downtrend', or None for all
            
        Returns:
        --------
        pd.DataFrame
            Trade setups with entry, SL, TP, risk-reward for each pattern
        """
        fib_df = self.find_fibonacci_levels(df, resolution=resolution, days_back=days_back)
        
        if fib_df.empty:
            return pd.DataFrame(columns=['pattern_id', 'trend', 'entry_price',
                                         'stop_loss', 'take_profit', 'risk_reward'])
        
        # Get unique patterns
        trade_setups = fib_df.groupby('pattern_id').first().reset_index()
        
        # Select columns (include fib level columns)
        fib_cols = list(FIB_LEVEL_COLUMNS.values())
        cols = ['pattern_id', 'trend', 'low_price', 'high_price',
                'entry_price', 'stop_loss', 'take_profit', 'range_pct',
                'resolution'] + fib_cols
        trade_setups = trade_setups[[c for c in cols if c in trade_setups.columns]]
        
        # Calculate risk-reward ratio
        def calc_rr(row):
            if row['trend'] == 'uptrend':
                risk = row['entry_price'] - row['stop_loss']
                reward = row['take_profit'] - row['entry_price']
            else:
                risk = row['stop_loss'] - row['entry_price']
                reward = row['entry_price'] - row['take_profit']
            return round(reward / risk, 2) if risk > 0 else 0
        
        trade_setups['risk_reward'] = trade_setups.apply(calc_rr, axis=1)
        
        # Apply trend filter
        if trend_filter:
            trade_setups = trade_setups[trade_setups['trend'] == trend_filter]
        
        return trade_setups.sort_values('range_pct', ascending=False).reset_index(drop=True)


def find_fibonacci_levels(df: pd.DataFrame,
                          resolution: str = '1D',
                          days_back: int = None,
                          backcandles: int = DEFAULT_BACKCANDLES,
                          gap_candles: int = DEFAULT_GAP_CANDLES,
                          min_price_diff_pct: float = DEFAULT_PRICE_DIFF_PCT) -> pd.DataFrame:
    """
    Convenience function to find Fibonacci levels from candle data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV candle data (columns: open, high, low, close, volume)
    resolution : str
        Timeframe label (e.g., '1D', '4H', '1H', '15m', '5m')
    days_back : int
        Number of days to look back (None = use all data)
    backcandles : int
        Lookback period for swing detection (default 40)
    gap_candles : int
        Gap before current candle (default 5)
    min_price_diff_pct : float
        Minimum price difference for valid swings (default 0.01 = 1%)
        
    Returns:
    --------
    pd.DataFrame
        Fibonacci levels with all retracement info
    """
    detector = FibonacciDetector(
        backcandles=backcandles,
        gap_candles=gap_candles,
        min_price_diff_pct=min_price_diff_pct
    )
    return detector.find_fibonacci_levels(df, resolution=resolution, days_back=days_back)


def get_fibonacci_trade_setups(df: pd.DataFrame,
                               resolution: str = '1D',
                               days_back: int = None,
                               trend_filter: str = None,
                               backcandles: int = DEFAULT_BACKCANDLES,
                               gap_candles: int = DEFAULT_GAP_CANDLES,
                               min_price_diff_pct: float = DEFAULT_PRICE_DIFF_PCT) -> pd.DataFrame:
    """
    Convenience function to get Fibonacci trade setups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV candle data
    resolution : str
        Timeframe label
    days_back : int
        Number of days to look back
    trend_filter : str
        Filter by trend: 'uptrend', 'downtrend', or None
    backcandles : int
        Lookback period for swing detection
    gap_candles : int
        Gap before current candle
    min_price_diff_pct : float
        Minimum price difference for valid swings
        
    Returns:
    --------
    pd.DataFrame
        Trade setups with entry, SL, TP, risk-reward
    """
    detector = FibonacciDetector(
        backcandles=backcandles,
        gap_candles=gap_candles,
        min_price_diff_pct=min_price_diff_pct
    )
    return detector.get_trade_setups(
        df, resolution=resolution, days_back=days_back, trend_filter=trend_filter
    )
