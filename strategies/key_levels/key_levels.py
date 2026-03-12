"""
Key Levels Detection Module

Identifies significant support and resistance levels based on pivot highs and lows.
Clusters similar price levels and counts how many times each level has been touched.

Input:
    - df: DataFrame with OHLCV candle data (columns: open, high, low, close, volume)
    - resolution: Timeframe label (e.g., '1D', '4H', '1H', '15m', '5m')
    - days_back: Number of days to look back for level detection
    
Output:
    - DataFrame with columns: level_price, type (support/resistance), touch_count, resolution
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Importance scores for each resolution (higher = more important)
RESOLUTION_IMPORTANCE = {
    '1D': 5,
    '4H': 4,
    '1H': 3,
    '15m': 2,
    '5m': 1
}

# Default price threshold for merging similar levels ($)
DEFAULT_PRICE_THRESHOLD = 0.5


@dataclass
class KeyLevel:
    """Represents a single key price level."""
    price: float
    level_type: str  # 'support' or 'resistance'
    touch_count: int
    resolution: str
    importance: int


class KeyLevelDetector:
    """
    Detects key support and resistance levels from price data.
    
    Parameters:
    -----------
    tolerance_pct : float
        Percentage tolerance for clustering similar price levels (default 0.5%)
    pivot_lookback : int
        Number of bars before/after for pivot detection (default 10)
    """
    
    def __init__(self, tolerance_pct: float = 0.5, pivot_lookback: int = 10):
        self.tolerance_pct = tolerance_pct
        self.pivot_lookback = pivot_lookback
    
    def _detect_pivot(self, df: pd.DataFrame, idx: int, n1: int, n2: int) -> int:
        """
        Detect pivot points by comparing candle at index with surrounding bars.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        idx : int
            Index of the bar to check
        n1 : int
            Number of bars to look back
        n2 : int
            Number of bars to look forward
            
        Returns:
        --------
        int:
            0 = No pivot
            1 = Pivot Low (support)
            2 = Pivot High (resistance)
            3 = Both (rare)
        """
        if idx - n1 < 0 or idx + n2 >= len(df):
            return 0
        
        pivot_low = True
        pivot_high = True
        
        for i in range(idx - n1, idx + n2 + 1):
            if df['low'].iloc[idx] > df['low'].iloc[i]:
                pivot_low = False
            if df['high'].iloc[idx] < df['high'].iloc[i]:
                pivot_high = False
        
        if pivot_low and pivot_high:
            return 3
        elif pivot_low:
            return 1
        elif pivot_high:
            return 2
        return 0
    
    def _detect_all_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all pivot points in the data."""
        df = df.copy()
        n = self.pivot_lookback
        df['pivot'] = [self._detect_pivot(df, i, n, n) for i in range(len(df))]
        return df
    
    def _cluster_levels(self, prices: List[float]) -> List[Tuple[float, int]]:
        """
        Cluster similar prices within tolerance range.
        
        Returns:
        --------
        List of (average_price, touch_count) tuples
        """
        if not prices:
            return []
        
        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            cluster_avg = np.mean(current_cluster)
            tolerance = cluster_avg * (self.tolerance_pct / 100)
            
            if abs(price - cluster_avg) <= tolerance:
                current_cluster.append(price)
            else:
                clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [price]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
        
        return clusters
    
    def find_key_levels(self, df: pd.DataFrame, resolution: str = '1D',
                        days_back: int = None) -> pd.DataFrame:
        """
        Find key support and resistance levels from price data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV candle data with columns: open, high, low, close, volume
            Index should be datetime or have a 'date'/'datetime' column
        resolution : str
            Timeframe label (e.g., '1D', '4H', '1H', '15m', '5m')
        days_back : int
            Number of days to look back (None = use all data)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: level_price, type, touch_count, resolution, importance
        """
        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Filter by days_back if specified
        if days_back is not None and days_back > 0:
            # Find datetime column
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
            return pd.DataFrame(columns=['level_price', 'type', 'touch_count', 
                                         'resolution', 'importance'])
        
        # Reset index for pivot detection
        df = df.reset_index(drop=True)
        
        # Detect pivots
        df = self._detect_all_pivots(df)
        
        # Extract pivot highs and lows
        pivot_highs = df[df['pivot'].isin([2, 3])]['high'].tolist()
        pivot_lows = df[df['pivot'].isin([1, 3])]['low'].tolist()
        
        # Cluster similar levels
        high_clusters = self._cluster_levels(pivot_highs)
        low_clusters = self._cluster_levels(pivot_lows)
        
        # Get importance score
        importance = RESOLUTION_IMPORTANCE.get(resolution, 1)
        
        # Build results
        results = []
        
        for price, count in high_clusters:
            results.append({
                'level_price': round(price, 2),
                'type': 'resistance',
                'touch_count': count,
                'resolution': resolution,
                'importance': importance
            })
        
        for price, count in low_clusters:
            results.append({
                'level_price': round(price, 2),
                'type': 'support',
                'touch_count': count,
                'resolution': resolution,
                'importance': importance
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def merge_levels(levels_df: pd.DataFrame, 
                     price_threshold: float = DEFAULT_PRICE_THRESHOLD) -> pd.DataFrame:
        """
        Merge similar price levels from multiple resolutions.
        
        Levels within price_threshold dollars of each other are merged.
        The highest importance is kept, touch counts are summed.
        
        Parameters:
        -----------
        levels_df : pd.DataFrame
            Combined key levels from multiple resolutions
        price_threshold : float
            Maximum price difference to merge levels
            
        Returns:
        --------
        pd.DataFrame
            Merged levels with columns: level_price, type, touch_count, importance
        """
        if levels_df.empty:
            return pd.DataFrame(columns=['level_price', 'type', 'touch_count', 'importance'])
        
        df = levels_df.copy()
        df = df.sort_values('level_price', ascending=False).reset_index(drop=True)
        
        merged = []
        used = set()
        
        for i, row in df.iterrows():
            if i in used:
                continue
            
            # Find all levels within threshold of same type
            similar_mask = (
                (abs(df['level_price'] - row['level_price']) <= price_threshold) &
                (df['type'] == row['type']) &
                (~df.index.isin(used))
            )
            similar = df[similar_mask]
            used.update(similar.index.tolist())
            
            # Merge: weighted average price, sum touches, max importance
            merged.append({
                'level_price': round(similar['level_price'].mean(), 2),
                'type': row['type'],
                'touch_count': similar['touch_count'].sum(),
                'importance': similar['importance'].max()
            })
        
        result = pd.DataFrame(merged)
        return result.sort_values('level_price', ascending=False).reset_index(drop=True)


def find_key_levels(df: pd.DataFrame, 
                    resolution: str = '1D',
                    days_back: int = None,
                    tolerance_pct: float = 0.5,
                    pivot_lookback: int = 10) -> pd.DataFrame:
    """
    Convenience function to find key levels from candle data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV candle data (columns: open, high, low, close, volume)
    resolution : str
        Timeframe label (e.g., '1D', '4H', '1H', '15m', '5m')
    days_back : int
        Number of days to look back (None = use all data)
    tolerance_pct : float
        Percentage tolerance for clustering (default 0.5%)
    pivot_lookback : int
        Bars before/after for pivot detection (default 10)
        
    Returns:
    --------
    pd.DataFrame
        Key levels with columns: level_price, type, touch_count, resolution, importance
    """
    detector = KeyLevelDetector(tolerance_pct=tolerance_pct, pivot_lookback=pivot_lookback)
    return detector.find_key_levels(df, resolution=resolution, days_back=days_back)


def merge_key_levels(levels_df: pd.DataFrame,
                     price_threshold: float = DEFAULT_PRICE_THRESHOLD) -> pd.DataFrame:
    """
    Convenience function to merge key levels from multiple resolutions.
    
    Parameters:
    -----------
    levels_df : pd.DataFrame
        Combined key levels from multiple resolutions
    price_threshold : float
        Maximum price difference to merge levels
        
    Returns:
    --------
    pd.DataFrame
        Merged levels with columns: level_price, type, touch_count, importance
    """
    return KeyLevelDetector.merge_levels(levels_df, price_threshold=price_threshold)
