"""
Chart Utilities Module

Reusable TradingView-style chart functions extracted from key_levels_dashboard.py.
Provides functions for creating candlestick charts with trade overlays.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime

# Try to import Trade from trade_tracker
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategies.trade_tracker import Trade
except ImportError:
    Trade = None  # Will work without trades


# TradingView-style color scheme
COLORS = {
    'background': '#131722',
    'paper': '#131722',
    'text': '#d1d4dc',
    'text_secondary': '#b2b5be',
    'grid': 'rgba(42, 46, 57, 0.6)',
    'candle_up': '#26a69a',
    'candle_down': '#ef5350',
    'support': 'rgba(38, 166, 154, 0.9)',
    'resistance': 'rgba(239, 83, 80, 0.9)',
    'tp_line': 'rgba(38, 166, 154, 0.8)',
    'sl_line': 'rgba(239, 83, 80, 0.8)',
    'trade_win': 'rgba(38, 166, 154, 0.15)',
    'trade_loss': 'rgba(239, 83, 80, 0.15)',
    'entry_marker': '#26a69a',
    'exit_marker': '#ef5350',
    'fib_382': 'rgba(255, 215, 0, 0.7)',
    'fib_50': 'rgba(255, 165, 0, 0.7)',
    'fib_618': 'rgba(255, 99, 71, 0.7)',
}


def create_candlestick_chart(
    ticker: str,
    chart_df: pd.DataFrame,
    timeframe: str = '1d',
) -> go.Figure:
    """
    Create a basic TradingView-style candlestick chart.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol
    chart_df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, and date_plot or Date
    timeframe : str
        Timeframe label for display
        
    Returns:
    --------
    go.Figure
        Plotly figure with candlestick chart
    """
    fig = go.Figure()
    
    # Prepare date column
    if 'date_plot' not in chart_df.columns:
        if 'Date' in chart_df.columns:
            chart_df = chart_df.copy()
            chart_df['date_plot'] = pd.to_datetime(chart_df['Date'])
        elif 'Datetime' in chart_df.columns:
            chart_df = chart_df.copy()
            chart_df['date_plot'] = pd.to_datetime(chart_df['Datetime'])
        else:
            chart_df = chart_df.copy()
            chart_df['date_plot'] = pd.to_datetime(chart_df.index)
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=chart_df['date_plot'],
        open=chart_df['open'],
        high=chart_df['high'],
        low=chart_df['low'],
        close=chart_df['close'],
        name=ticker,
        increasing_line_color=COLORS['candle_up'],
        decreasing_line_color=COLORS['candle_down'],
        increasing_fillcolor=COLORS['candle_up'],
        decreasing_fillcolor=COLORS['candle_down']
    ))
    
    return fig


def add_trade_markers(
    fig: go.Figure,
    trades: List[Any],  # List[Trade]
    chart_df: pd.DataFrame,
) -> go.Figure:
    """
    Add trade entry/exit markers, TP/SL lines, and trade zones to chart.
    
    Parameters:
    -----------
    fig : go.Figure
        Existing Plotly figure
    trades : List[Trade]
        List of Trade objects
    chart_df : pd.DataFrame
        Chart data for x-axis range
        
    Returns:
    --------
    go.Figure
        Figure with trade overlays added
    """
    if not trades:
        return fig
    
    # Get x-axis range
    if 'date_plot' in chart_df.columns:
        x_start = chart_df['date_plot'].iloc[0]
        x_end = chart_df['date_plot'].iloc[-1]
    else:
        x_start = chart_df.index[0]
        x_end = chart_df.index[-1]
    
    for trade in trades:
        entry_date = trade.date_executed
        exit_date = trade.date_completed if trade.date_completed else x_end
        
        # Trade zone shading (background rectangle)
        if trade.date_completed:
            zone_color = COLORS['trade_win'] if trade.is_winner else COLORS['trade_loss']
            fig.add_vrect(
                x0=entry_date,
                x1=exit_date,
                fillcolor=zone_color,
                layer="below",
                line_width=0,
            )
        
        # Take Profit line (horizontal from entry to exit)
        fig.add_trace(go.Scatter(
            x=[entry_date, exit_date],
            y=[trade.take_profit, trade.take_profit],
            mode='lines',
            line=dict(color=COLORS['tp_line'], width=1, dash='dash'),
            name=f'TP ${trade.take_profit:.2f}',
            showlegend=False,
            hovertemplate=f"<b>Take Profit</b><br>${trade.take_profit:.2f}<extra></extra>"
        ))
        
        # Stop Loss line (horizontal from entry to exit)
        fig.add_trace(go.Scatter(
            x=[entry_date, exit_date],
            y=[trade.stop_loss, trade.stop_loss],
            mode='lines',
            line=dict(color=COLORS['sl_line'], width=1, dash='dash'),
            name=f'SL ${trade.stop_loss:.2f}',
            showlegend=False,
            hovertemplate=f"<b>Stop Loss</b><br>${trade.stop_loss:.2f}<extra></extra>"
        ))
        
        # Entry marker (triangle up)
        fig.add_trace(go.Scatter(
            x=[entry_date],
            y=[trade.entry_price],
            mode='markers+text',
            marker=dict(
                symbol='triangle-up',
                size=14,
                color=COLORS['entry_marker'],
                line=dict(width=1, color='white')
            ),
            text=[f"BUY ${trade.entry_price:.2f}"],
            textposition="top center",
            textfont=dict(color=COLORS['entry_marker'], size=10),
            name='Entry',
            showlegend=False,
            hovertemplate=(
                f"<b>ENTRY</b><br>"
                f"Price: ${trade.entry_price:.2f}<br>"
                f"Qty: {trade.quantity}<br>"
                f"TP: ${trade.take_profit:.2f}<br>"
                f"SL: ${trade.stop_loss:.2f}"
                f"<extra></extra>"
            )
        ))
        
        # Exit marker (triangle down) - only if trade is closed
        if trade.date_completed and trade.exit_price:
            exit_color = COLORS['entry_marker'] if trade.is_winner else COLORS['exit_marker']
            pnl_text = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
            
            fig.add_trace(go.Scatter(
                x=[trade.date_completed],
                y=[trade.exit_price],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-down',
                    size=14,
                    color=exit_color,
                    line=dict(width=1, color='white')
                ),
                text=[f"{trade.exit_reason} {pnl_text}"],
                textposition="bottom center",
                textfont=dict(color=exit_color, size=10),
                name='Exit',
                showlegend=False,
                hovertemplate=(
                    f"<b>EXIT ({trade.exit_reason})</b><br>"
                    f"Price: ${trade.exit_price:.2f}<br>"
                    f"PnL: {pnl_text}<br>"
                    f"Duration: {trade.duration:.1f}h"
                    f"<extra></extra>"
                )
            ))
        
        # Connection line from entry to exit
        if trade.date_completed and trade.exit_price:
            fig.add_trace(go.Scatter(
                x=[entry_date, trade.date_completed],
                y=[trade.entry_price, trade.exit_price],
                mode='lines',
                line=dict(
                    color=COLORS['entry_marker'] if trade.is_winner else COLORS['exit_marker'],
                    width=1,
                    dash='dot'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    return fig


def add_key_levels(
    fig: go.Figure,
    levels_df: pd.DataFrame,
    chart_df: pd.DataFrame,
    min_importance: int = 1,
    show_support: bool = True,
    show_resistance: bool = True,
    proximity_filter: float = 0.25,
) -> go.Figure:
    """
    Add key support/resistance levels to chart.
    
    Parameters:
    -----------
    fig : go.Figure
        Existing Plotly figure
    levels_df : pd.DataFrame
        Merged levels DataFrame with columns: level_price, type, importance, touch_count
    chart_df : pd.DataFrame
        Chart data for x-axis range and proximity filtering
    min_importance : int
        Minimum importance level to show
    show_support : bool
        Whether to show support levels
    show_resistance : bool
        Whether to show resistance levels
    proximity_filter : float
        Only show levels within this percentage of current price (0.25 = 25%)
        
    Returns:
    --------
    go.Figure
        Figure with levels added
    """
    if levels_df is None or levels_df.empty:
        return fig
    
    # Get current price and filter range
    current_price = chart_df['close'].iloc[-1]
    min_price = current_price * (1 - proximity_filter)
    max_price = current_price * (1 + proximity_filter)
    
    # Get x-axis range
    if 'date_plot' in chart_df.columns:
        x_start = chart_df['date_plot'].iloc[0]
        x_end = chart_df['date_plot'].iloc[-1]
    else:
        x_start = chart_df.index[0]
        x_end = chart_df.index[-1]
    
    # Filter levels
    filtered_levels = levels_df[
        (levels_df['importance'] >= min_importance) &
        (levels_df['level_price'] >= min_price) &
        (levels_df['level_price'] <= max_price)
    ]
    
    for _, row in filtered_levels.iterrows():
        level_type = row['type']
        
        if (level_type == 'support' and not show_support) or \
           (level_type == 'resistance' and not show_resistance):
            continue
        
        color = COLORS['support'] if level_type == 'support' else COLORS['resistance']
        dash_style = 'solid' if row['importance'] >= 4 else 'dash'
        width = 2 if row['importance'] >= 4 else 1
        
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[row['level_price'], row['level_price']],
            mode='lines+text',
            line=dict(color=color, width=width, dash=dash_style),
            text=["", f"${row['level_price']:.2f}"],
            textposition="middle right",
            textfont=dict(color=color, size=10),
            name=f"${row['level_price']:.2f}",
            showlegend=False,
            hovertemplate=(
                f"<b>${row['level_price']:.2f}</b><br>"
                f"{row['type'].title()}<br>"
                f"Touches: {row['touch_count']}<br>"
                f"Importance: {row['importance']}"
                f"<extra></extra>"
            )
        ))
    
    return fig


def get_tradingview_layout(
    ticker: str,
    timeframe: str,
    current_price: float,
    chart_df: pd.DataFrame,
    height: int = 700,
) -> dict:
    """
    Get TradingView-style layout configuration.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol
    timeframe : str
        Timeframe label (e.g., '1D', '4H')
    current_price : float
        Current/latest price
    chart_df : pd.DataFrame
        Chart data for axis range calculation
    height : int
        Chart height in pixels
        
    Returns:
    --------
    dict
        Layout configuration for Plotly
    """
    # Determine tick format based on timeframe
    if timeframe in ['15m', '5m', '1h']:
        tick_format = '%Y-%m-%d\n%H:%M'
    else:
        tick_format = '%Y-%m-%d'
    
    return dict(
        title=dict(
            text=f"{ticker} • {timeframe} • ${current_price:.2f}",
            font=dict(size=20, color=COLORS['text'], family='Arial'),
            x=0.01,
            xanchor='left'
        ),
        xaxis_title="",
        yaxis_title="",
        template="plotly_dark",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['paper'],
        font=dict(family='Arial', size=11, color=COLORS['text_secondary']),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False,
            rangeslider=dict(visible=False),
            showticklabels=True,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            showline=False,
            tickformat=tick_format,
        ),
        yaxis=dict(
            side='right',
            gridcolor=COLORS['grid'],
            showgrid=True,
            zeroline=False,
            tickprefix='$',
            tickformat='.2f',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            range=[
                chart_df['low'].min() * 0.98,
                chart_df['high'].max() * 1.02
            ]
        ),
        legend=dict(visible=False),
        height=height,
        margin=dict(l=10, r=60, t=50, b=20),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e222d',
            font_size=12,
            font_family='Arial',
            bordercolor='#2a2e39'
        )
    )


def create_trade_chart(
    ticker: str,
    chart_df: pd.DataFrame,
    timeframe: str = '1D',
    trades: List[Any] = None,
    levels_df: pd.DataFrame = None,
    show_trades: bool = True,
    show_levels: bool = True,
    show_support: bool = True,
    show_resistance: bool = True,
    min_importance: int = 1,
    height: int = 700,
) -> go.Figure:
    """
    Create a complete TradingView-style chart with trades and levels.
    
    This is the main function that combines all chart elements.
    
    Parameters:
    -----------
    ticker : str
        Stock symbol
    chart_df : pd.DataFrame
        OHLCV data
    timeframe : str
        Timeframe label for display
    trades : List[Trade]
        List of Trade objects to overlay
    levels_df : pd.DataFrame
        Key levels DataFrame
    show_trades : bool
        Whether to show trade markers
    show_levels : bool
        Whether to show key levels
    show_support : bool
        Whether to show support levels
    show_resistance : bool
        Whether to show resistance levels
    min_importance : int
        Minimum importance for levels
    height : int
        Chart height in pixels
        
    Returns:
    --------
    go.Figure
        Complete chart figure
    """
    # Create base candlestick chart
    fig = create_candlestick_chart(ticker, chart_df, timeframe)
    
    # Ensure date_plot column exists
    if 'date_plot' not in chart_df.columns:
        chart_df = chart_df.copy()
        if 'Date' in chart_df.columns:
            chart_df['date_plot'] = pd.to_datetime(chart_df['Date'])
        elif 'Datetime' in chart_df.columns:
            chart_df['date_plot'] = pd.to_datetime(chart_df['Datetime'])
        else:
            chart_df['date_plot'] = pd.to_datetime(chart_df.index)
    
    # Add key levels if provided
    if show_levels and levels_df is not None:
        fig = add_key_levels(
            fig, levels_df, chart_df,
            min_importance=min_importance,
            show_support=show_support,
            show_resistance=show_resistance
        )
    
    # Add trade markers if provided
    if show_trades and trades:
        fig = add_trade_markers(fig, trades, chart_df)
    
    # Apply layout
    current_price = chart_df['close'].iloc[-1]
    layout = get_tradingview_layout(ticker, timeframe, current_price, chart_df, height)
    fig.update_layout(**layout)
    
    return fig
