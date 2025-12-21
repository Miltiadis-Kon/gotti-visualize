"""
Key Levels Plotting Module

Uses TradingView's lightweight-charts for a clean, interactive charting experience.
"""

import pandas as pd

# Try to import lightweight_charts
try:
    import lightweight_charts as lc
    LIGHTWEIGHT_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_AVAILABLE = False
    print("Warning: lightweight-charts not installed. Install with: pip install lightweight-charts")

# Import Plotly as fallback
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Global plot toggle
PLOT = True


def plot_key_levels_lightweight(ticker: str, 
                                 chart_df: pd.DataFrame,
                                 merged_df: pd.DataFrame,
                                 fib_df: pd.DataFrame,
                                 timeframe: str = '1d',
                                 timeframe_label: str = '1D') -> None:
    """
    Plot candlestick chart with key levels using TradingView's lightweight-charts.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol
    chart_df : pd.DataFrame
        OHLCV data for candlesticks
    merged_df : pd.DataFrame
        Merged key levels with importance
    fib_df : pd.DataFrame
        Fibonacci retracement levels
    timeframe : str
        Timeframe interval
    timeframe_label : str
        Human-readable timeframe label
    """
    global PLOT
    if not PLOT:
        print("Plotting disabled (PLOT=False)")
        return
    
    if not LIGHTWEIGHT_AVAILABLE:
        print("Error: lightweight_charts not installed. Install with: pip install lightweight-charts")
        return
    
    if chart_df is None or chart_df.empty:
        print("Error: No chart data available")
        return
    
    # Prepare data for lightweight_charts
    df = chart_df.copy()
    
    # Reset index to avoid Timestamp in index
    df = df.reset_index(drop=True)
    
    # Find the time column and convert to string
    if 'Date' in df.columns:
        df['time'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    elif 'Datetime' in df.columns:
        df['time'] = pd.to_datetime(df['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    elif 'datetime' in df.columns:
        df['time'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    elif 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Try to use original index before reset
        original_index = chart_df.index
        if hasattr(original_index, 'strftime'):
            df['time'] = pd.to_datetime(original_index).strftime('%Y-%m-%d')
        else:
            df['time'] = [str(x) for x in original_index]
    
    # Ensure OHLCV columns are float (not any special types)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Drop any remaining datetime columns that could cause issues
    cols_to_drop = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
    df_clean = df[['time', 'open', 'high', 'low', 'close']].copy()
    if 'volume' in df.columns:
        df_clean['volume'] = df['volume']
    
    # Create chart
    chart = lc.Chart(
        title=f"{ticker} Key Levels - {timeframe_label}",
        toolbox=True,
        width=1400,
        height=800
    )
    
    # Set candlestick data
    chart.set(df_clean)
    
    # Add horizontal lines for key levels
    if merged_df is not None and not merged_df.empty:
        for _, row in merged_df.iterrows():
            level_type = row['type']
            color = '#00FF00' if level_type == 'support' else '#FF4444'  # Green/Red
            
            chart.horizontal_line(
                price=row['level_price'],
                color=color,
                width=2,
                style='solid',
                text=f"${row['level_price']:.2f} ({row['touch_count']}x) Imp:{row['importance']}"
            )
    
    # Add Fibonacci levels (dashed, gold)
    if fib_df is not None and not fib_df.empty:
        # Deduplicate by price to avoid too many lines
        unique_fibs = fib_df.drop_duplicates(subset=['fib_price'])
        
        for _, fib_row in unique_fibs.iterrows():
            chart.horizontal_line(
                price=fib_row['fib_price'],
                color='#FFD700',  # Gold
                width=1,
                style='dashed',
                text=f"Fib {fib_row['fib_level']} ${fib_row['fib_price']:.2f}"
            )
    
    # Print summary
    fib_count = len(fib_df.drop_duplicates(subset=['fib_price'])) if fib_df is not None and not fib_df.empty else 0
    level_count = len(merged_df) if merged_df is not None else 0
    
    print(f"\n✅ Lightweight chart opened for {ticker}")
    print(f"   Timeframe: {timeframe_label}")
    print(f"   Key Levels: {level_count} | Fibonacci: {fib_count}")
    
    # Show chart (blocking)
    chart.show(block=True)


def plot_key_levels_plotly(ticker: str,
                           timeframe_data: dict,
                           all_levels_df: pd.DataFrame,
                           merged_df: pd.DataFrame = None,
                           fib_df: pd.DataFrame = None,
                           show: bool = True,
                           save_html: bool = False) -> 'go.Figure':
    """
    Plot candlestick chart with key levels using Plotly (fallback).
    
    This is the original Plotly implementation with dropdown filtering.
    """
    global PLOT
    if not PLOT:
        print("Plotting disabled (PLOT=False)")
        return None
    
    if not PLOTLY_AVAILABLE:
        print("Error: plotly not installed")
        return None
    
    # Import constants
    from strategies.key_levels import TIMEFRAME_COLORS, TIMEFRAME_LOOKBACK
    
    if not timeframe_data:
        print("No data available for plotting")
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Resolution to interval mapping
    resolution_to_interval = {'1D': '1d', '4H': '4h', '1H': '1h', '15m': '15m', '5m': '5m'}
    interval_to_resolution = {'1d': '1D', '4h': '4H', '1h': '1H', '15m': '15m', '5m': '5m'}
    
    trace_info = []
    
    # Add candlestick traces
    for tf_interval in ['1d', '4h', '1h', '15m', '5m']:
        if tf_interval not in timeframe_data:
            continue
        
        chart_df = timeframe_data[tf_interval]
        tf_label = TIMEFRAME_LOOKBACK[tf_interval]['label']
        
        x_col = 'Date' if 'Date' in chart_df.columns else 'Datetime' if 'Datetime' in chart_df.columns else None
        x_data = chart_df[x_col] if x_col else chart_df.index
        
        visible = (tf_interval == '1d')
        
        fig.add_trace(go.Candlestick(
            x=x_data,
            open=chart_df['open'],
            high=chart_df['high'],
            low=chart_df['low'],
            close=chart_df['close'],
            name=f"{ticker} ({tf_label})",
            increasing_line_color='green',
            decreasing_line_color='red',
            visible=visible
        ))
        trace_info.append({'type': 'candlestick', 'resolution': tf_label})
    
    # Get x-axis range
    ref_df = timeframe_data.get('1d', list(timeframe_data.values())[0])
    x_col = 'Date' if 'Date' in ref_df.columns else 'Datetime' if 'Datetime' in ref_df.columns else None
    x_ref = ref_df[x_col] if x_col else ref_df.index
    x_start = x_ref.iloc[0] if hasattr(x_ref, 'iloc') else x_ref[0]
    x_end = x_ref.iloc[-1] if hasattr(x_ref, 'iloc') else x_ref[-1]
    
    # Add key level lines
    for _, row in all_levels_df.iterrows():
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
            visible=True
        ))
        trace_info.append({'type': 'level', 'resolution': row['resolution']})
    
    # Add Fibonacci levels
    if fib_df is not None and not fib_df.empty:
        FIB_COLORS = {0.382: 'rgba(255,215,0,0.8)', 0.50: 'rgba(255,165,0,0.8)', 0.618: 'rgba(255,69,0,0.8)'}
        
        for _, fib_row in fib_df.drop_duplicates(subset=['fib_price']).iterrows():
            fib_color = FIB_COLORS.get(fib_row['fib_ratio'], 'gold')
            
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[fib_row['fib_price'], fib_row['fib_price']],
                mode='lines',
                line=dict(color=fib_color, width=1, dash='dash'),
                name=f"Fib ${fib_row['fib_price']:.2f} ({fib_row['fib_level']})",
                legendgroup='Fibonacci',
                showlegend=True,
                visible=True
            ))
            trace_info.append({'type': 'fibonacci', 'resolution': 'Fib'})
    
    # Layout
    fig.update_layout(
        title=f"{ticker} Multi-Timeframe Key Levels",
        xaxis_title="Date/Time",
        yaxis_title="Price ($)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=900
    )
    
    if save_html:
        filename = f"./logs/charts/{ticker}_key_levels.html"
        fig.write_html(filename)
        print(f"Chart saved to {filename}")
    
    if show:
        fig.show()
    
    return fig


def set_plot_enabled(enabled: bool):
    """Enable or disable plotting globally."""
    global PLOT
    PLOT = enabled
    print(f"Plotting {'enabled' if PLOT else 'disabled'}")
