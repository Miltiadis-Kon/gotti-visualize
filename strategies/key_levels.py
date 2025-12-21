"""
Key Levels Detection Strategy - Multi-Timeframe

Identifies significant support and resistance levels based on pivot highs and lows
across multiple timeframes (Daily, 4H, 1H, 15M).
Clusters similar price levels and counts how many times each level has been touched.

Data Source: Alpaca API (with Yahoo Finance fallback)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to path for plots module import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import Alpaca, fall back to Yahoo if not available
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.live import StockDataStream
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: alpaca-py not installed. Using Yahoo Finance as fallback.")

# Yahoo Finance as fallback
try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False


# Color scheme for different timeframes
TIMEFRAME_COLORS = {
    '1d': 'lime',      # Daily - green
    '4h': 'cyan',      # 4 Hour - cyan
    '1h': 'yellow',    # 1 Hour - yellow
    '15m': 'magenta',  # 15 Min - magenta
    '5m': 'orange'     # 5 Min - orange
}

# Lookback periods for each timeframe
TIMEFRAME_LOOKBACK = {
    '1d': {'days': 150, 'label': '1D'},   # 5 months
    '4h': {'days': 90, 'label': '4H'},    # 3 months
    '1h': {'days': 30, 'label': '1H'},    # 1 month
    '15m': {'days': 7, 'label': '15m'},   # 1 week
    '5m': {'days': 2, 'label': '5m'}      # Current + past day
}

# Importance scores for each timeframe (higher = more important)
TIMEFRAME_IMPORTANCE = {
    '1D': 5,    # Daily - highest importance
    '4H': 4,    # 4 Hour
    '1H': 3,    # 1 Hour
    '15m': 2,   # 15 Min
    '5m': 1     # 5 Min - lowest importance
}

# Price threshold for merging similar levels ($)
PRICE_THRESHOLD = 0.5

# Fibonacci retracement configuration
FIBONACCI_LEVELS = [0.382, 0.50, 0.618]  # 38.2%, 50%, 61.8%
FIBONACCI_THRESHOLD = 0.20  # Only calculate if levels are > 20% apart
FIBONACCI_IMPORTANCE = 3    # Minimum importance to calculate Fibonacci

# Alpaca timeframe mapping
ALPACA_TIMEFRAME_MAP = {
    '1d': TimeFrame.Day,
    '4h': TimeFrame(4, TimeFrameUnit.Hour),
    '1h': TimeFrame.Hour,
    '15m': TimeFrame(15, TimeFrameUnit.Minute),
    '5m': TimeFrame(5, TimeFrameUnit.Minute)
} if ALPACA_AVAILABLE else {}


class KeyLevels:
    """
    Detects and visualizes key price levels (support/resistance) based on pivot points
    across multiple timeframes.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to analyze
    tolerance_pct : float
        Percentage tolerance for clustering similar price levels (default 0.5%)
    pivot_lookback : int
        Number of bars before/after for pivot detection (default 10)
    use_alpaca : bool
        Whether to use Alpaca API (default True, falls back to Yahoo if unavailable)
    """
    
    def __init__(self, ticker: str = "NVDA", 
                 tolerance_pct: float = 0.5, 
                 pivot_lookback: int = 10,
                 use_alpaca: bool = True):
        self.ticker = ticker
        self.tolerance_pct = tolerance_pct
        self.pivot_lookback = pivot_lookback
        self.use_alpaca = use_alpaca and ALPACA_AVAILABLE
        self.timeframe_data = {}  # Store data for each timeframe
        self.all_levels_df = None
        
        # Initialize Alpaca client if available
        if self.use_alpaca:
            api_key = os.getenv("APCA_API_KEY_PAPER")
            api_secret = os.getenv("APCA_API_SECRET_KEY_PAPER")
            if api_key and api_secret:
                self.alpaca_client = StockHistoricalDataClient(api_key, api_secret)
            else:
                print("Warning: Alpaca API keys not found. Falling back to Yahoo Finance.")
                self.use_alpaca = False
                self.alpaca_client = None
        else:
            self.alpaca_client = None
        
    def fetch_data(self, interval: str = '1d', lookback_days: int = 150) -> pd.DataFrame:
        """
        Fetch historical data from Alpaca API (or Yahoo Finance as fallback).
        
        Parameters:
        -----------
        interval : str
            Data interval: '1d', '4h', '1h', '15m'
        lookback_days : int
            Number of days to look back
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        print(f"Fetching {self.ticker} {interval} data from {start_date.date()} to {end_date.date()}...")
        
        if self.use_alpaca and self.alpaca_client:
            return self._fetch_alpaca(interval, start_date, end_date)
        else:
            return self._fetch_yahoo(interval, start_date, end_date)
    
    def _fetch_alpaca(self, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Alpaca API using IEX feed (free tier)."""
        try:
            from alpaca.data.enums import DataFeed
            
            timeframe = ALPACA_TIMEFRAME_MAP.get(interval, TimeFrame.Day)
            
            request = StockBarsRequest(
                symbol_or_symbols=self.ticker,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX  # Use IEX for free tier
            )
            
            bars = self.alpaca_client.get_stock_bars(request)
            
            if not bars.data or self.ticker not in bars.data:
                print(f"  No Alpaca data returned, falling back to Yahoo...")
                return self._fetch_yahoo(interval, start_date, end_date)
            
            # Convert to DataFrame
            df = bars.df
            
            # Reset index to get timestamp as column
            df = df.reset_index()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Rename 'timestamp' to 'Date' or 'Datetime'
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'Datetime'})
            
            # Filter out rows with zero volume
            if 'volume' in df.columns:
                df = df[df['volume'] != 0]
            
            df.reset_index(drop=True, inplace=True)
            
            print(f"  Fetched {len(df)} bars from Alpaca")
            return df
            
        except Exception as e:
            print(f"  Alpaca error: {e}. Falling back to Yahoo...")
            return self._fetch_yahoo(interval, start_date, end_date)
    
    def _fetch_yahoo(self, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from Yahoo Finance with extended hours (fallback)."""
        if not YAHOO_AVAILABLE:
            print("  Error: Neither Alpaca nor Yahoo Finance available!")
            return pd.DataFrame()
        
        try:
            import yfinance as yf
            stock = yf.Ticker(self.ticker)
            
            # Use prepost=True for extended hours (pre-market + after-hours)
            df = stock.history(
                start=start_date, 
                end=end_date, 
                interval=interval,
                prepost=True  # Include pre-market and after-hours data
            )
            
            # Rename columns to lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            
            # Filter out rows with zero volume
            df = df[df['volume'] != 0]
            df.reset_index(inplace=True)
            
            print(f"  Fetched {len(df)} bars from Yahoo (with extended hours)")
            return df
        except Exception as e:
            print(f"  Yahoo error: {e}")
            return pd.DataFrame()
    
    def _pivotid(self, df: pd.DataFrame, l: int, n1: int, n2: int) -> int:
        """
        Detect pivot points by comparing candle at index l with n1 bars before and n2 bars after.
        
        Returns:
        --------
        0: No pivot
        1: Pivot Low (support)
        2: Pivot High (resistance)
        3: Both pivot high and low (rare)
        """
        if l - n1 < 0 or l + n2 >= len(df):
            return 0
        
        pivot_low = True
        pivot_high = True
        
        for i in range(l - n1, l + n2 + 1):
            if df['low'].iloc[l] > df['low'].iloc[i]:
                pivot_low = False
            if df['high'].iloc[l] < df['high'].iloc[i]:
                pivot_high = False
                
        if pivot_low and pivot_high:
            return 3
        elif pivot_low:
            return 1
        elif pivot_high:
            return 2
        else:
            return 0
    
    def detect_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all pivot points in the data."""
        n = self.pivot_lookback
        df['pivot'] = df.apply(lambda x: self._pivotid(df, x.name, n, n), axis=1)
        return df
    
    def _cluster_levels(self, prices: list, tolerance_pct: float) -> list:
        """
        Cluster similar prices within a tolerance range.
        Returns list of (average_price, count) tuples.
        """
        if not prices:
            return []
        
        prices = sorted(prices)
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            # Check if price is within tolerance of current cluster average
            cluster_avg = np.mean(current_cluster)
            tolerance = cluster_avg * (tolerance_pct / 100)
            
            if abs(price - cluster_avg) <= tolerance:
                current_cluster.append(price)
            else:
                # Save current cluster and start new one
                clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [price]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
        
        return clusters
    
    def find_key_levels_for_timeframe(self, interval: str) -> pd.DataFrame:
        """
        Find key support and resistance levels for a specific timeframe.
        
        Parameters:
        -----------
        interval : str
            Timeframe interval: '1d', '4h', '1h', '15m'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: level_price, type, touch_count, resolution
        """
        lookback_config = TIMEFRAME_LOOKBACK.get(interval, TIMEFRAME_LOOKBACK['1d'])
        
        # Fetch data
        df = self.fetch_data(interval=interval, lookback_days=lookback_config['days'])
        
        if df.empty:
            print(f"  Warning: No data for {interval}")
            return pd.DataFrame()
        
        # Detect pivots
        df = self.detect_pivots(df)
        
        # Store the data for plotting
        self.timeframe_data[interval] = df
        
        # Extract pivot highs and lows
        pivot_highs = df[df['pivot'].isin([2, 3])]['high'].tolist()
        pivot_lows = df[df['pivot'].isin([1, 3])]['low'].tolist()
        
        pivot_high_count = len(pivot_highs)
        pivot_low_count = len(pivot_lows)
        print(f"  Detected {pivot_high_count} pivot highs and {pivot_low_count} pivot lows")
        
        # Cluster similar levels
        high_clusters = self._cluster_levels(pivot_highs, self.tolerance_pct)
        low_clusters = self._cluster_levels(pivot_lows, self.tolerance_pct)
        
        # Build results DataFrame
        results = []
        
        for price, count in high_clusters:
            results.append({
                'level_price': round(price, 2),
                'type': 'resistance',
                'touch_count': count,
                'resolution': lookback_config['label']
            })
            
        for price, count in low_clusters:
            results.append({
                'level_price': round(price, 2),
                'type': 'support',
                'touch_count': count,
                'resolution': lookback_config['label']
            })
        
        return pd.DataFrame(results)
    
    def find_all_key_levels(self, timeframes: list = None) -> pd.DataFrame:
        """
        Find key levels across all specified timeframes.
        
        Parameters:
        -----------
        timeframes : list
            List of timeframes to analyze. Default: ['1d', '4h', '1h', '15m', '5m']
            
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all key levels and their resolutions
        """
        if timeframes is None:
            timeframes = ['1d', '4h', '1h', '15m', '5m']
        
        all_levels = []
        
        print("\n" + "=" * 60)
        print(f"MULTI-TIMEFRAME KEY LEVELS ANALYSIS - {self.ticker}")
        print("=" * 60)
        
        for tf in timeframes:
            print(f"\n--- {TIMEFRAME_LOOKBACK[tf]['label']} Timeframe ---")
            levels_df = self.find_key_levels_for_timeframe(tf)
            if not levels_df.empty:
                all_levels.append(levels_df)
        
        if all_levels:
            self.all_levels_df = pd.concat(all_levels, ignore_index=True)
            # Sort by price descending
            self.all_levels_df = self.all_levels_df.sort_values('level_price', ascending=False)
            self.all_levels_df.reset_index(drop=True, inplace=True)
        else:
            self.all_levels_df = pd.DataFrame(columns=['level_price', 'type', 'touch_count', 'resolution'])
        
        print("\n" + "=" * 60)
        print("ALL KEY LEVELS (sorted by price)")
        print("=" * 60)
        print(self.all_levels_df.to_string())
        
        return self.all_levels_df
    
    def get_merged_levels(self, price_threshold: float = None) -> pd.DataFrame:
        """
        Merge similar price levels within a threshold and add importance scores.
        
        Levels within PRICE_THRESHOLD dollars of each other are merged.
        The highest importance (from the highest timeframe) is kept.
        Touch counts are summed.
        
        Parameters:
        -----------
        price_threshold : float
            Maximum price difference to merge levels (default: PRICE_THRESHOLD constant)
            
        Returns:
        --------
        pd.DataFrame
            Merged DataFrame with columns: level_price, type, touch_count, importance
        """
        if self.all_levels_df is None or self.all_levels_df.empty:
            self.find_all_key_levels()
        
        if self.all_levels_df.empty:
            return pd.DataFrame(columns=['level_price', 'type', 'touch_count', 'importance'])
        
        threshold = price_threshold if price_threshold is not None else PRICE_THRESHOLD
        
        # Add importance column based on resolution
        df = self.all_levels_df.copy()
        df['importance'] = df['resolution'].map(TIMEFRAME_IMPORTANCE)
        
        # Sort by price descending
        df = df.sort_values('level_price', ascending=False).reset_index(drop=True)
        
        # Merge similar levels
        merged_levels = []
        used_indices = set()
        
        for i, row in df.iterrows():
            if i in used_indices:
                continue
            
            # Find all levels within threshold
            similar_mask = (
                (abs(df['level_price'] - row['level_price']) <= threshold) &
                (df['type'] == row['type']) &
                (~df.index.isin(used_indices))
            )
            similar_levels = df[similar_mask]
            
            # Mark these as used
            used_indices.update(similar_levels.index.tolist())
            
            # Merge: use weighted average price, sum touch_count, max importance
            merged_level = {
                'level_price': round(similar_levels['level_price'].mean(), 2),
                'type': row['type'],
                'touch_count': similar_levels['touch_count'].sum(),
                'importance': similar_levels['importance'].max()
            }
            merged_levels.append(merged_level)
        
        merged_df = pd.DataFrame(merged_levels)
        
        # Sort by price descending
        merged_df = merged_df.sort_values('level_price', ascending=False).reset_index(drop=True)
        
        print("\n" + "=" * 60)
        print(f"MERGED KEY LEVELS (threshold: ${threshold})")
        print("=" * 60)
        print(merged_df.to_string())
        print(f"\nMerged {len(self.all_levels_df)} levels into {len(merged_df)} unique levels")
        
        return merged_df
    
    def calculate_fibonacci_levels(self, merged_df: pd.DataFrame = None,
                                    fib_threshold: float = None,
                                    min_importance: int = None) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels between key support/resistance levels.
        
        Fibonacci levels (38.2%, 50%, 61.8%) are calculated between pairs of levels
        where the price difference exceeds the threshold and both levels have
        sufficient importance.
        
        Parameters:
        -----------
        merged_df : pd.DataFrame
            Merged key levels DataFrame. If None, will calculate from scratch.
        fib_threshold : float
            Minimum % difference between levels (default: FIBONACCI_THRESHOLD = 20%)
        min_importance : int
            Minimum importance for both levels (default: FIBONACCI_IMPORTANCE = 3)
            
        Returns:
        --------
        pd.DataFrame
            Fibonacci levels with columns: fib_price, fib_level, low_price, high_price, importance
        """
        if merged_df is None:
            merged_df = self.get_merged_levels()
        
        if merged_df.empty:
            return pd.DataFrame(columns=['fib_price', 'fib_level', 'low_price', 'high_price', 'importance'])
        
        threshold = fib_threshold if fib_threshold is not None else FIBONACCI_THRESHOLD
        importance = min_importance if min_importance is not None else FIBONACCI_IMPORTANCE
        
        # Get all levels sorted by price
        levels = merged_df.sort_values('level_price').reset_index(drop=True)
        
        fibonacci_levels = []
        used_pairs = set()
        
        # Iterate through all pairs of levels
        for i, low_level in levels.iterrows():
            for j, high_level in levels.iterrows():
                if i >= j:
                    continue
                
                low_price = low_level['level_price']
                high_price = high_level['level_price']
                low_importance = low_level['importance']
                high_importance = high_level['importance']
                
                # Check importance threshold
                if low_importance < importance and high_importance < importance:
                    continue
                
                # Calculate percentage difference
                pct_diff = (high_price - low_price) / low_price
                
                # Check if exceeds threshold
                if pct_diff < threshold:
                    continue
                
                # Create unique pair key to avoid duplicates
                pair_key = (round(low_price, 2), round(high_price, 2))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)
                
                # Calculate Fibonacci retracement levels
                # From low to high: fib_price = low + (high - low) * fib_ratio
                range_val = high_price - low_price
                pair_importance = max(low_importance, high_importance)
                
                # Create a readable ID for the pattern
                pattern_id = f"${low_price:.0f}-${high_price:.0f}"
                
                # Calculate Ranking Score:
                # Based on range percentage (bigger moves are more significant) AND importance of levels
                # Range score: 20% diff = 20 points
                # Importance score: Max 10 points (5+5) -> scaled to be significant
                range_score = pct_diff * 100 
                importance_score = (low_importance + high_importance) * 2 
                total_score = range_score + importance_score
                
                for fib_ratio in FIBONACCI_LEVELS:
                    fib_price = round(low_price + range_val * fib_ratio, 2)
                    
                    fibonacci_levels.append({
                        'fib_price': fib_price,
                        'fib_level': f"{fib_ratio*100:.1f}%",
                        'fib_ratio': fib_ratio,
                        'low_price': low_price,
                        'high_price': high_price,
                        'importance': pair_importance,
                        'range_pct': round(pct_diff * 100, 1),
                        'pattern_id': pattern_id,
                        'pattern_score': round(total_score, 1)
                    })
        
        fib_df = pd.DataFrame(fibonacci_levels)
        
        if not fib_df.empty:
            # Sort by pattern score (descending) then fib price
            fib_df = fib_df.sort_values(
                ['pattern_score', 'fib_price'], 
                ascending=[False, False]
            ).reset_index(drop=True)
            
            # Rank patterns
            unique_patterns = fib_df['pattern_id'].unique()
            # Map pattern_id to rank (1, 2, 3...)
            # Because we sorted by score, the first unique pattern is Rank 1
            rank_map = {pid: i+1 for i, pid in enumerate(unique_patterns)}
            fib_df['pattern_rank'] = fib_df['pattern_id'].map(rank_map)
            
            print(f"\nCalculated {len(fib_df)} Fibonacci levels from {len(used_pairs)} solid patterns")
        else:
            print("\nNo Fibonacci levels found")
            
        return fib_df
    
    def plot_key_levels(self, show: bool = True, save_html: bool = False) -> go.Figure:
        """
        Plot candlestick chart with key levels as horizontal lines.
        Features:
        - Dropdown to filter by resolution (All, 1D, 4H, 1H, 15m)
        - Candlestick chart updates to match selected timeframe
        - Price labels on each level
        - Touch count annotations
        - Different colors per timeframe
        """
        if self.all_levels_df is None:
            self.find_all_key_levels()
        
        if not self.timeframe_data:
            print("No data available for plotting")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Resolution to interval mapping
        resolution_to_interval = {'1D': '1d', '4H': '4h', '1H': '1h', '15m': '15m', '5m': '5m'}
        interval_to_resolution = {'1d': '1D', '4h': '4H', '1h': '1H', '15m': '15m', '5m': '5m'}
        
        # Track trace indices for visibility control
        trace_info = []  # List of dicts with 'type' and 'resolution'
        
        # Add candlestick traces for each timeframe
        for tf_interval in ['1d', '4h', '1h', '15m', '5m']:
            if tf_interval not in self.timeframe_data:
                continue
                
            chart_df = self.timeframe_data[tf_interval]
            tf_label = TIMEFRAME_LOOKBACK[tf_interval]['label']
            
            # Determine x-axis column
            x_col = 'Date' if 'Date' in chart_df.columns else 'Datetime' if 'Datetime' in chart_df.columns else None
            if x_col:
                x_data = chart_df[x_col]
            else:
                x_data = chart_df.index
            
            # Only show 1D candlestick by default
            visible = (tf_interval == '1d')
            
            fig.add_trace(go.Candlestick(
                x=x_data,
                open=chart_df['open'],
                high=chart_df['high'],
                low=chart_df['low'],
                close=chart_df['close'],
                name=f"{self.ticker} ({tf_label})",
                increasing_line_color='green',
                decreasing_line_color='red',
                visible=visible
            ))
            trace_info.append({'type': 'candlestick', 'resolution': tf_label})
        
        # Get x-axis range from daily data (for level lines)
        if '1d' in self.timeframe_data:
            ref_df = self.timeframe_data['1d']
        else:
            ref_df = list(self.timeframe_data.values())[0]
        
        x_col = 'Date' if 'Date' in ref_df.columns else 'Datetime' if 'Datetime' in ref_df.columns else None
        if x_col:
            x_ref = ref_df[x_col]
        else:
            x_ref = ref_df.index
        x_start = x_ref.iloc[0] if hasattr(x_ref, 'iloc') else x_ref[0]
        x_end = x_ref.iloc[-1] if hasattr(x_ref, 'iloc') else x_ref[-1]
        
        # Add horizontal lines for each key level
        for idx, row in self.all_levels_df.iterrows():
            interval = resolution_to_interval.get(row['resolution'], '1d')
            color = TIMEFRAME_COLORS.get(interval, 'lime')
            
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[row['level_price'], row['level_price']],
                mode='lines',
                line=dict(color=color, width=1.5, dash='solid'),
                name=f"${row['level_price']:.2f} ({row['resolution']})",
                legendgroup=row['resolution'],
                showlegend=True,
                hovertemplate=f"${row['level_price']:.2f}<br>{row['type']}<br>{row['touch_count']} touches<br>{row['resolution']}<extra></extra>",
                visible=True
            ))
            trace_info.append({'type': 'level', 'resolution': row['resolution']})
        
        # Calculate and add Fibonacci levels
        merged_df = self.get_merged_levels()
        fib_df = self.calculate_fibonacci_levels(merged_df=merged_df)
        
        # Fibonacci color scheme
        FIB_COLORS = {
            0.382: 'rgba(255, 215, 0, 0.8)',   # Gold for 38.2%
            0.50: 'rgba(255, 165, 0, 0.8)',    # Orange for 50%
            0.618: 'rgba(255, 69, 0, 0.8)'     # Red-Orange for 61.8%
        }
        
        if not fib_df.empty:
            # Group Fibonacci levels to avoid duplicates at same price
            for _, fib_row in fib_df.drop_duplicates(subset=['fib_price']).iterrows():
                fib_color = FIB_COLORS.get(fib_row['fib_ratio'], 'rgba(255, 215, 0, 0.8)')
                
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[fib_row['fib_price'], fib_row['fib_price']],
                    mode='lines',
                    line=dict(color=fib_color, width=1, dash='dash'),
                    name=f"Fib ${fib_row['fib_price']:.2f} ({fib_row['fib_level']})",
                    legendgroup='Fibonacci',
                    showlegend=True,
                    hovertemplate=f"Fib ${fib_row['fib_price']:.2f}<br>{fib_row['fib_level']}<br>Range: ${fib_row['low_price']:.2f} - ${fib_row['high_price']:.2f}<extra></extra>",
                    visible=True
                ))
                trace_info.append({'type': 'fibonacci', 'resolution': 'Fib'})
        
        # Create dropdown buttons for filtering by resolution
        all_resolutions = ['All', '1D', '4H', '1H', '15m', '5m', 'Fib']
        buttons = []
        
        for res in all_resolutions:
            visibility = []
            
            for info in trace_info:
                if info['type'] == 'candlestick':
                    if res == 'All' or res == 'Fib':
                        # In "All" or "Fib" view, show only 1D candlestick
                        visibility.append(info['resolution'] == '1D')
                    else:
                        # Show only matching candlestick
                        visibility.append(info['resolution'] == res)
                elif info['type'] == 'fibonacci':
                    # Show Fibonacci in 'All' view and 'Fib' view
                    if res == 'All' or res == 'Fib':
                        visibility.append(True)
                    else:
                        visibility.append(False)
                else:  # level
                    if res == 'All':
                        visibility.append(True)
                    elif res == 'Fib':
                        visibility.append(False)  # Hide levels in Fib-only view
                    else:
                        visibility.append(info['resolution'] == res)
            
            title = f"{self.ticker} Key Levels - {res}" if res not in ['All', 'Fib'] else f"{self.ticker} Multi-Timeframe Key Levels (Daily Chart)"
            if res == 'Fib':
                title = f"{self.ticker} Fibonacci Retracement Levels"
            
            buttons.append(dict(
                label=res,
                method='update',
                args=[
                    {'visible': visibility},
                    {'title': title}
                ]
            ))
        
        # Add price annotations (always visible, will overlap but that's ok)
        annotations = [
            dict(
                text="Resolution:",
                x=0.94,
                xref="paper",
                y=1.12,
                yref="paper",
                showarrow=False,
                font=dict(color='white', size=12)
            )
        ]
        
        for idx, row in self.all_levels_df.iterrows():
            interval = resolution_to_interval.get(row['resolution'], '1d')
            color = TIMEFRAME_COLORS.get(interval, 'lime')
            
            annotations.append(dict(
                x=x_end,
                y=row['level_price'],
                text=f"${row['level_price']:.2f} ({row['touch_count']}x {row['resolution']})",
                showarrow=False,
                xanchor='left',
                xshift=10,
                font=dict(color=color, size=10),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor=color,
                borderwidth=1
            ))
        
        # Styling with dropdown menu
        fig.update_layout(
            title=f"{self.ticker} Multi-Timeframe Key Levels (Daily Chart)",
            xaxis_title="Date/Time",
            yaxis_title="Price ($)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                yanchor="top", 
                y=0.99, 
                xanchor="left", 
                x=0.01,
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='white',
                borderwidth=1,
                font=dict(size=9)
            ),
            height=900,
            plot_bgcolor='black',
            paper_bgcolor='#1a1a1a',
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.98,
                    xanchor="right",
                    y=1.12,
                    yanchor="top",
                    bgcolor='#333',
                    bordercolor='white',
                    font=dict(color='white')
                )
            ],
            annotations=annotations
        )
        
        if save_html:
            filename = f".\\logs\\charts\\{self.ticker}_multi_tf_key_levels.html"
            fig.write_html(filename)
            print(f"\nChart saved to {filename}")
        
        if show:
            fig.show()
        
        return fig
    
    def get_levels_dataframe(self) -> pd.DataFrame:
        """Return the key levels as a DataFrame."""
        if self.all_levels_df is None:
            self.find_all_key_levels()
        return self.all_levels_df
    
    def plot_lightweight(self, timeframe: str = '1d') -> None:
        """
        Plot candlestick chart with key levels using TradingView's lightweight-charts.
        
        More interactive and cleaner than Plotly for financial data visualization.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe to display: '1d', '4h', '1h', '15m', '5m'
        """
        try:
            import lightweight_charts as lc
        except ImportError:
            print("Error: lightweight_charts not installed. Install with: pip install lightweight-charts")
            return
        
        if self.all_levels_df is None:
            self.find_all_key_levels()
        
        if timeframe not in self.timeframe_data:
            print(f"Error: No data for timeframe {timeframe}")
            return
        
        # Get chart data
        chart_df = self.timeframe_data[timeframe].copy()
        
        # Prepare data for lightweight_charts (needs specific column names)
        if 'Date' in chart_df.columns:
            chart_df['time'] = chart_df['Date']
        elif 'Datetime' in chart_df.columns:
            chart_df['time'] = chart_df['Datetime']
        else:
            chart_df['time'] = chart_df.index
        
        # Create chart
        chart = lc.Chart(
            title=f"{self.ticker} Key Levels - {TIMEFRAME_LOOKBACK[timeframe]['label']}",
            toolbox=True,
            width=1200,
            height=800
        )
        
        # Set candlestick data
        chart.set(chart_df)
        
        # Get merged levels and Fibonacci
        merged_df = self.get_merged_levels()
        fib_df = self.calculate_fibonacci_levels(merged_df=merged_df)
        
        # Add horizontal lines for key levels
        for _, row in merged_df.iterrows():
            level_type = row['type']
            color = 'lime' if level_type == 'support' else 'red'
            
            chart.horizontal_line(
                price=row['level_price'],
                color=color,
                width=2,
                style='solid',
                text=f"${row['level_price']:.2f} ({row['touch_count']}x) Imp:{row['importance']}"
            )
        
        # Add Fibonacci levels (dashed)
        if not fib_df.empty:
            # Deduplicate by price
            for _, fib_row in fib_df.drop_duplicates(subset=['fib_price']).iterrows():
                chart.horizontal_line(
                    price=fib_row['fib_price'],
                    color='gold',
                    width=1,
                    style='dashed',
                    text=f"Fib {fib_row['fib_level']} ${fib_row['fib_price']:.2f}"
                )
        
        print(f"\n✅ Lightweight chart opened for {self.ticker}")
        print(f"   Key Levels: {len(merged_df)} | Fibonacci: {len(fib_df.drop_duplicates(subset=['fib_price'])) if not fib_df.empty else 0}")
        
        # Show chart (blocking)
        chart.show(block=True)


def run_key_levels_analysis(ticker: str = "NVDA", 
                            timeframes: list = None,
                            tolerance_pct: float = 0.5,
                            pivot_lookback: int = 10,
                            show_plot: bool = True,
                            save_html: bool = False,
                            use_alpaca: bool = True) -> pd.DataFrame:
    """
    Convenience function to run full multi-timeframe key levels analysis.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to analyze
    timeframes : list
        List of timeframes: ['1d', '4h', '1h', '15m'] (default: all)
    tolerance_pct : float
        Percentage tolerance for clustering (default 0.5%)
    pivot_lookback : int
        Bars before/after for pivot detection
    show_plot : bool
        Whether to display the interactive chart
    save_html : bool
        Whether to save chart as HTML file
    use_alpaca : bool
        Whether to use Alpaca API (default True, falls back to Yahoo)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with key levels: level_price, type, touch_count, resolution
    """
    if timeframes is None:
        timeframes = ['1d', '4h', '1h', '15m']
    
    kl = KeyLevels(
        ticker=ticker,
        tolerance_pct=tolerance_pct,
        pivot_lookback=pivot_lookback,
        use_alpaca=use_alpaca
    )
    
    levels_df = kl.find_all_key_levels(timeframes=timeframes)
    
    if show_plot or save_html:
        kl.plot_key_levels(show=show_plot, save_html=save_html)
    
    return levels_df


# Legacy function for single timeframe (backward compatibility)
def run_single_timeframe_analysis(ticker: str = "NVDA", 
                                   lookback_months: int = 5,
                                   tolerance_pct: float = 0.5,
                                   pivot_lookback: int = 10,
                                   show_plot: bool = True) -> pd.DataFrame:
    """Run analysis for a single timeframe (daily only)."""
    return run_key_levels_analysis(
        ticker=ticker,
        timeframes=['1d'],
        tolerance_pct=tolerance_pct,
        pivot_lookback=pivot_lookback,
        show_plot=show_plot
    )


def run_merged_key_levels(ticker: str = "NVDA",
                          timeframes: list = None,
                          price_threshold: float = None,
                          tolerance_pct: float = 0.5,
                          pivot_lookback: int = 10,
                          use_alpaca: bool = True,
                          show_plot: bool = False) -> pd.DataFrame:
    """
    Convenience function to get merged key levels with importance scores.
    
    Analyzes all timeframes (1D, 4H, 1H, 15m, 5m), merges similar price levels
    within the price_threshold, and returns a consolidated DataFrame with
    importance scores.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to analyze
    timeframes : list
        List of timeframes (default: ['1d', '4h', '1h', '15m', '5m'])
    price_threshold : float
        Maximum price difference to merge levels (default: PRICE_THRESHOLD = 0.5)
    tolerance_pct : float
        Percentage tolerance for clustering within timeframes
    pivot_lookback : int
        Bars before/after for pivot detection
    use_alpaca : bool
        Whether to use Alpaca API
    show_plot : bool
        Whether to display the chart
        
    Returns:
    --------
    pd.DataFrame
        Merged levels with columns: level_price, type, touch_count, importance
        
    Importance Scale:
        5 = Daily (1D) - highest importance
        4 = 4 Hour (4H)
        3 = 1 Hour (1H)
        2 = 15 Minutes (15m)
        1 = 5 Minutes (5m) - lowest importance
    """
    if timeframes is None:
        timeframes = ['1d', '4h', '1h', '15m', '5m']
    
    kl = KeyLevels(
        ticker=ticker,
        tolerance_pct=tolerance_pct,
        pivot_lookback=pivot_lookback,
        use_alpaca=use_alpaca
    )
    
    # First get all levels
    kl.find_all_key_levels(timeframes=timeframes)
    
    # Then merge them
    merged_df = kl.get_merged_levels(price_threshold=price_threshold)
    
    if show_plot:
        kl.plot_key_levels(show=True, save_html=False)
    
    return merged_df


def run_fibonacci_analysis(ticker: str = "NVDA",
                           timeframes: list = None,
                           price_threshold: float = None,
                           fib_threshold: float = None,
                           min_importance: int = None,
                           use_alpaca: bool = True,
                           show_plot: bool = False) -> tuple:
    """
    Convenience function to get merged key levels with Fibonacci retracements.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to analyze
    timeframes : list
        List of timeframes (default: ['1d', '4h', '1h', '15m', '5m'])
    price_threshold : float
        Maximum price difference to merge levels (default: PRICE_THRESHOLD = 0.5)
    fib_threshold : float
        Minimum % difference for Fibonacci (default: FIBONACCI_THRESHOLD = 20%)
    min_importance : int
        Minimum importance for Fibonacci (default: FIBONACCI_IMPORTANCE = 3)
    use_alpaca : bool
        Whether to use Alpaca API
    show_plot : bool
        Whether to display the chart
        
    Returns:
    --------
    tuple (merged_df, fib_df)
        merged_df: Merged key levels with importance
        fib_df: Fibonacci retracement levels
    """
    if timeframes is None:
        timeframes = ['1d', '4h', '1h', '15m', '5m']
    
    kl = KeyLevels(
        ticker=ticker,
        use_alpaca=use_alpaca
    )
    
    # Get all levels
    kl.find_all_key_levels(timeframes=timeframes)
    
    # Merge them
    merged_df = kl.get_merged_levels(price_threshold=price_threshold)
    
    # Calculate Fibonacci levels
    fib_df = kl.calculate_fibonacci_levels(
        merged_df=merged_df,
        fib_threshold=fib_threshold,
        min_importance=min_importance
    )
    
    if show_plot:
        # Use lightweight-charts from plots module
        try:
            from plots.key_levels_plot import plot_key_levels_lightweight
            chart_df = kl.timeframe_data.get('1d', list(kl.timeframe_data.values())[0])
            plot_key_levels_lightweight(
                ticker=ticker,
                chart_df=chart_df,
                merged_df=merged_df,
                fib_df=fib_df,
                timeframe='1d',
                timeframe_label='1D'
            )
        except ImportError:
            # Fallback to Plotly
            kl.plot_key_levels(show=True, save_html=False)
    
    return merged_df, fib_df


if __name__ == "__main__":
    # Run multi-timeframe analysis with Fibonacci and lightweight chart
    merged_df, fib_df = run_fibonacci_analysis(ticker="NVDA", show_plot=True)


# =============================================================================
# LIVE STREAMING SUPPORT
# =============================================================================

class LiveKeyLevels:
    """
    Live streaming key levels monitor using Alpaca's WebSocket API.
    
    Monitors real-time price action and alerts when price approaches or 
    breaches key support/resistance levels.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to monitor
    tolerance_pct : float
        Percentage tolerance for clustering levels
    alert_distance_pct : float
        Alert when price is within this % of a key level
    """
    
    def __init__(self, ticker: str = "NVDA", 
                 tolerance_pct: float = 0.5,
                 alert_distance_pct: float = 0.3):
        self.ticker = ticker
        self.tolerance_pct = tolerance_pct
        self.alert_distance_pct = alert_distance_pct
        self.key_levels_df = None
        self.stream = None
        self.last_price = None
        self.breached_levels = set()
        
        # Get API credentials
        self.api_key = os.getenv("APCA_API_KEY_PAPER")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY_PAPER")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
    
    def load_key_levels(self, timeframes: list = None):
        """Load key levels using historical data analysis."""
        print(f"\n{'='*60}")
        print(f"Loading key levels for {self.ticker}...")
        print(f"{'='*60}")
        
        kl = KeyLevels(
            ticker=self.ticker,
            tolerance_pct=self.tolerance_pct,
            use_alpaca=True
        )
        
        if timeframes is None:
            timeframes = ['1d', '4h', '1h', '15m']
        
        self.key_levels_df = kl.find_all_key_levels(timeframes=timeframes)
        
        print(f"\nLoaded {len(self.key_levels_df)} key levels for monitoring")
        return self.key_levels_df
    
    def _check_level_proximity(self, current_price: float):
        """Check if current price is near any key levels."""
        if self.key_levels_df is None or self.key_levels_df.empty:
            return
        
        for _, level in self.key_levels_df.iterrows():
            level_price = level['level_price']
            level_type = level['type']
            resolution = level['resolution']
            
            # Calculate distance percentage
            distance_pct = abs(current_price - level_price) / level_price * 100
            
            # Check if within alert distance
            if distance_pct <= self.alert_distance_pct:
                level_key = f"{level_price}_{level_type}"
                
                if level_key not in self.breached_levels:
                    direction = "above" if current_price > level_price else "below"
                    icon = "🔴" if level_type == "resistance" else "🟢"
                    
                    print(f"\n{icon} ALERT: Price ${current_price:.2f} is {distance_pct:.2f}% {direction} "
                          f"{level_type} at ${level_price:.2f} ({resolution})")
                    
                    # Check for breach
                    if (level_type == "resistance" and current_price > level_price) or \
                       (level_type == "support" and current_price < level_price):
                        print(f"   ⚡ LEVEL BREACHED!")
                        self.breached_levels.add(level_key)
    
    async def _on_bar(self, bar):
        """Callback for incoming bar data."""
        current_price = bar.close
        timestamp = bar.timestamp
        
        # Only print every few updates to avoid spam
        if self.last_price is None or abs(current_price - self.last_price) > 0.01:
            print(f"[{timestamp.strftime('%H:%M:%S')}] {self.ticker}: ${current_price:.2f}", end="")
            
            if self.last_price:
                change = current_price - self.last_price
                change_pct = change / self.last_price * 100
                icon = "📈" if change > 0 else "📉"
                print(f" {icon} {change:+.2f} ({change_pct:+.2f}%)")
            else:
                print()
            
            self.last_price = current_price
            self._check_level_proximity(current_price)
    
    async def _on_quote(self, quote):
        """Callback for incoming quote data."""
        bid = quote.bid_price
        ask = quote.ask_price
        mid = (bid + ask) / 2
        
        if self.last_price is None or abs(mid - self.last_price) > 0.05:
            print(f"[{quote.timestamp.strftime('%H:%M:%S')}] {self.ticker}: "
                  f"Bid ${bid:.2f} | Ask ${ask:.2f} | Mid ${mid:.2f}")
            
            self.last_price = mid
            self._check_level_proximity(mid)
    
    def start_streaming(self, use_bars: bool = True):
        """
        Start live streaming with level monitoring.
        
        Parameters:
        -----------
        use_bars : bool
            If True, stream bars (1-min candles). If False, stream quotes.
        """
        if not ALPACA_AVAILABLE:
            print("Error: Alpaca not available for live streaming")
            return
        
        # Load key levels first
        if self.key_levels_df is None:
            self.load_key_levels()
        
        print(f"\n{'='*60}")
        print(f"Starting live stream for {self.ticker}...")
        print(f"Monitoring {len(self.key_levels_df)} key levels")
        print(f"Alert distance: {self.alert_distance_pct}% from level")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop\n")
        
        # Initialize stream with IEX feed
        self.stream = StockDataStream(
            self.api_key, 
            self.api_secret,
            feed=DataFeed.IEX
        )
        
        # Subscribe to data
        if use_bars:
            self.stream.subscribe_bars(self._on_bar, self.ticker)
        else:
            self.stream.subscribe_quotes(self._on_quote, self.ticker)
        
        try:
            self.stream.run()
        except KeyboardInterrupt:
            print("\n\nStream stopped by user")
        finally:
            self.stream.stop()
    
    def stop_streaming(self):
        """Stop the live stream."""
        if self.stream:
            self.stream.stop()
            print("Stream stopped")


def run_live_monitor(ticker: str = "NVDA", 
                     alert_distance_pct: float = 0.3,
                     use_bars: bool = True):
    """
    Convenience function to start live key levels monitoring.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol to monitor
    alert_distance_pct : float
        Alert when price is within this % of a key level
    use_bars : bool
        If True, stream 1-min bars. If False, stream quotes.
    """
    live = LiveKeyLevels(
        ticker=ticker,
        alert_distance_pct=alert_distance_pct
    )
    
    live.start_streaming(use_bars=use_bars)

