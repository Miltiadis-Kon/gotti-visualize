"""
Key Levels Chart Module

Interactive Plotly candlestick chart with key support/resistance levels
and Fibonacci retracement overlays.

Color Palette:
--------------
Fibonacci Levels (TradingView-style rainbow):
    0.000  Dark Gray   #787B86
    0.236  Red/Pink    #F23645
    0.382  Orange      #FF9800
    0.500  Green       #4CAF50
    0.618  Gold/Yellow #FFEB3B
    0.786  Blue        #2196F3
    1.000  Dark Gray   #787B86

Support & Resistance (Blue palette by importance):
    Major S/R   Navy Blue   #0D47A1   (Weekly/Monthly)
    Minor S/R   Royal Blue  #2962FF   (Daily/4H)
    Intraday    Sky Blue    #82B1FF   (1H/15m)
    S/R Zone    Light Cyan  #E3F2FD   (filled rectangles, opacity 0.3)

Usage:
------
    from strategies.key_levels import analyze
    from strategies.key_levels.chart import plot_analysis

    result = analyze("NVDA", resolutions=["1D"])
    plot_analysis(result)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from strategies.key_levels.fibonacci_levels import FIBONACCI_LEVELS, FIB_LEVEL_COLUMNS


# ── Fibonacci color palette ──────────────────────────────────────────────────
FIB_COLORS: Dict[float, str] = {
    0.0:   '#787B86',   # Dark Gray   – Baseline
    0.236: '#F23645',   # Red/Pink    – Shallow Retracement
    0.382: '#FF9800',   # Orange      – Common Retracement
    0.50:  '#4CAF50',   # Green       – Psychological Midpoint
    0.618: '#FFEB3B',   # Gold/Yellow – The Golden Ratio
    0.786: '#2196F3',   # Blue        – Deep Retracement
    1.0:   '#787B86',   # Dark Gray   – Full Retracement
}

FIB_LABELS: Dict[float, str] = {
    0.0:   '0.0%',
    0.236: '23.6%',
    0.382: '38.2%',
    0.50:  '50.0%',
    0.618: '61.8%',
    0.786: '78.6%',
    1.0:   '100.0%',
}

# ── Support & Resistance color palette ────────────────────────────────────────
SR_COLORS: Dict[str, str] = {
    'major':    '#0D47A1',   # Navy Blue   – Weekly/Monthly
    'minor':    '#2962FF',   # Royal Blue  – Daily/4H
    'intraday': '#82B1FF',   # Sky Blue    – 1H/15m
    'zone':     '#E3F2FD',   # Light Cyan  – Filled zones
}

# Map resolution → S/R tier
RESOLUTION_SR_TIER: Dict[str, str] = {
    '1D': 'major',
    '4H': 'minor',
    '1H': 'intraday',
    '15m': 'intraday',
    '5m': 'intraday',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: resolve date column from a candle DataFrame
# ═══════════════════════════════════════════════════════════════════════════════
def _get_date_series(df: pd.DataFrame) -> pd.Series:
    """Return a datetime Series from whichever column/index holds dates."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in ('date', 'datetime', 'timestamp'):
        if candidate in df_cols_lower:
            return pd.to_datetime(df[df_cols_lower[candidate]])
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series().reset_index(drop=True)
    raise ValueError("Cannot find a date column in candle data. "
                     "Expected one of: date, datetime, timestamp")


# ═══════════════════════════════════════════════════════════════════════════════
# Main plotting function
# ═══════════════════════════════════════════════════════════════════════════════
def plot_analysis(result,
                  resolution: str = None,
                  show_volume: bool = True,
                  show_key_levels: bool = True,
                  show_fibonacci: bool = True,
                  max_fib_patterns: int = 1,
                  height: int = 900,
                  width: int = 1400,
                  dark_theme: bool = True) -> go.Figure:
    """
    Plot candlestick chart with key levels and Fibonacci overlays.

    Parameters
    ----------
    result : AnalysisResult
        Output from ``analyze()`` or ``quick_analyze()``.
    resolution : str, optional
        Which resolution's candle data to plot (default: first available).
    show_volume : bool
        Show a volume sub-plot below the chart (default True).
    show_key_levels : bool
        Draw support/resistance lines (default True).
    show_fibonacci : bool
        Draw Fibonacci retracement levels (default True).
    max_fib_patterns : int
        Limit the number of Fibonacci patterns rendered (most recent first).
    height, width : int
        Figure dimensions in pixels.
    dark_theme : bool
        Use dark background (default True).

    Returns
    -------
    go.Figure  – the Plotly figure (also calls ``fig.show()``).
    """

    # ── resolve candle data ─────────────────────────────────────────────────
    if resolution is None:
        # Pick the first (usually highest) resolution available
        resolution = next(iter(result.candle_data))
    
    if resolution not in result.candle_data:
        available = list(result.candle_data.keys())
        raise ValueError(f"Resolution '{resolution}' not in candle_data. "
                         f"Available: {available}")

    df = result.candle_data[resolution].copy()
    df.columns = [c.lower() for c in df.columns]
    dates = _get_date_series(df)

    # ── create subplots (candlestick + optional volume) ─────────────────────
    row_heights = [0.75, 0.25] if show_volume else [1.0]
    rows = 2 if show_volume else 1

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ── candlestick ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26A69A',
            decreasing_fillcolor='#EF5350',
        ),
        row=1, col=1,
    )

    # ── volume bars ─────────────────────────────────────────────────────────
    if show_volume and 'volume' in df.columns:
        colors = np.where(
            df['close'] >= df['open'], '#26A69A', '#EF5350'
        )
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df['volume'],
                marker_color=colors,
                name='Volume',
                showlegend=False,
                opacity=0.5,
            ),
            row=2, col=1,
        )

    # ── key levels (S/R) ───────────────────────────────────────────────────
    if show_key_levels and not result.merged_levels.empty:
        _add_key_levels(fig, result.merged_levels, dates, resolution)

    # ── fibonacci levels ────────────────────────────────────────────────────
    if show_fibonacci and not result.trade_setups.empty:
        _add_fibonacci_levels(fig, result.trade_setups, dates, max_fib_patterns)

    # ── layout ──────────────────────────────────────────────────────────────
    template = 'plotly_dark' if dark_theme else 'plotly_white'
    bg_color = '#131722' if dark_theme else '#FFFFFF'
    grid_color = '#1E222D' if dark_theme else '#E0E0E0'
    font_color = '#D1D4DC' if dark_theme else '#131722'

    fig.update_layout(
        title=dict(
            text=f"{result.ticker}  •  {resolution}",
            font=dict(size=20, color=font_color),
            x=0.5,
        ),
        template=template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=height,
        width=width,
        font=dict(color=font_color),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
    )

    # Grid styling
    for ax in ['xaxis', 'yaxis'] + (['xaxis2', 'yaxis2'] if show_volume else []):
        fig.update_layout(**{
            ax: dict(
                gridcolor=grid_color,
                gridwidth=0.5,
                showgrid=True,
            )
        })

    # Date formatting on x-axis
    fig.update_xaxes(
        type='date',
        tickformat='%b %d\n%Y' if resolution == '1D' else '%b %d %H:%M',
        dtick=_date_tick(resolution),
    )

    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text='Volume', row=2, col=1)

    fig.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Support / Resistance overlay
# ═══════════════════════════════════════════════════════════════════════════════
def _add_key_levels(fig: go.Figure,
                    merged_levels: pd.DataFrame,
                    dates: pd.Series,
                    resolution: str) -> None:
    """Add horizontal S/R lines colored by importance tier."""
    x_start = dates.iloc[0]
    x_end = dates.iloc[-1]

    for _, row in merged_levels.iterrows():
        price = row['level_price']
        level_type = row.get('type', 'support')
        importance = row.get('importance', 1)
        touch_count = row.get('touch_count', 1)

        # Choose colour tier
        if importance >= 4:
            tier = 'major'
        elif importance >= 3:
            tier = 'minor'
        else:
            tier = 'intraday'
        color = SR_COLORS[tier]

        # Dashed for support, solid for resistance
        dash = 'dash' if level_type == 'support' else 'solid'

        # Level line
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[price, price],
                mode='lines',
                line=dict(color=color, width=1.5, dash=dash),
                name=f"{level_type.title()} ${price:.2f}",
                hovertemplate=(
                    f"<b>{level_type.title()}</b> ${price:.2f}<br>"
                    f"Touches: {touch_count}<br>"
                    f"Importance: {importance}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Zone fill (thin band around the level)
        zone_width = price * 0.002  # 0.2 % band
        fig.add_shape(
            type='rect',
            x0=x_start, x1=x_end,
            y0=price - zone_width, y1=price + zone_width,
            fillcolor=SR_COLORS['zone'],
            opacity=0.15,
            line_width=0,
            row=1, col=1,
        )

        # Price annotation on the right edge
        fig.add_annotation(
            x=x_end,
            y=price,
            text=f"  ${price:.2f}",
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor='left',
            row=1, col=1,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Fibonacci overlay
# ═══════════════════════════════════════════════════════════════════════════════
def _add_fibonacci_levels(fig: go.Figure,
                          trade_setups: pd.DataFrame,
                          dates: pd.Series,
                          max_patterns: int) -> None:
    """Draw Fibonacci retracement bands for each trade setup pattern (most recent first)."""
    # Sort by pattern_id descending so the latest-in-time pattern comes first.
    # pattern_id has the form "$low-$high"; use the original row order
    # (which reflects bar_index / recency) reversed.
    setups = trade_setups.iloc[::-1].head(max_patterns).reset_index(drop=True)
    x_start = dates.iloc[0]
    x_end = dates.iloc[-1]

    fib_col_map = FIB_LEVEL_COLUMNS          # {0.0: 'fib_0', ...}
    fib_ratios = sorted(fib_col_map.keys())   # [0.0, 0.236, ..., 1.0]

    for pat_idx, (_, setup) in enumerate(setups.iterrows()):
        trend = setup['trend']
        pattern_id = setup['pattern_id']
        opacity = max(0.25, 1.0 - pat_idx * 0.15)   # fade weaker patterns

        # Collect (ratio, price) pairs from the row
        level_prices = {}
        for ratio in fib_ratios:
            col = fib_col_map[ratio]
            if col in setup.index and pd.notna(setup[col]):
                level_prices[ratio] = setup[col]

        if not level_prices:
            continue

        # Draw each fib level as a horizontal line
        for ratio, price in level_prices.items():
            color = FIB_COLORS.get(ratio, '#787B86')
            label = FIB_LABELS.get(ratio, f'{ratio*100:.1f}%')

            # Only show in legend for the first pattern
            show_legend = (pat_idx == 0)

            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[price, price],
                    mode='lines',
                    line=dict(color=color, width=1.2, dash='dot'),
                    opacity=opacity,
                    name=f"Fib {label}" if show_legend else f"Fib {label}",
                    legendgroup=f"fib_{ratio}",
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>Fib {label}</b> ${price:.2f}<br>"
                        f"Trend: {trend}<br>"
                        f"Pattern: {pattern_id}"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Shaded zone between 0.618 (entry) and 0.786 (SL) – the "kill zone"
        entry_col = fib_col_map.get(0.618, 'fib_618')
        sl_col = fib_col_map.get(0.786, 'fib_786')
        if entry_col in setup.index and sl_col in setup.index:
            y_entry = setup[entry_col]
            y_sl = setup[sl_col]
            fig.add_shape(
                type='rect',
                x0=x_start, x1=x_end,
                y0=min(y_entry, y_sl),
                y1=max(y_entry, y_sl),
                fillcolor='#FFEB3B',
                opacity=0.07 * opacity,
                line_width=0,
                row=1, col=1,
            )

        # Annotation with trend arrow + pattern info
        mid_price = (setup.get('low_price', 0) + setup.get('high_price', 0)) / 2
        arrow = '▲' if trend == 'uptrend' else '▼'
        fig.add_annotation(
            x=x_start,
            y=mid_price,
            text=f"{arrow} {pattern_id}",
            showarrow=False,
            font=dict(size=8, color='#D1D4DC'),
            opacity=opacity,
            xanchor='right',
            xshift=-5,
            row=1, col=1,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _date_tick(resolution: str):
    """Return a sensible dtick value for the x-axis based on resolution."""
    mapping = {
        '1D': 'M1',          # monthly ticks
        '4H': 7 * 86400000,  # weekly (ms)
        '1H': 2 * 86400000,  # every 2 days
        '15m': 86400000,     # daily
        '5m':  43200000,     # every 12 h
    }
    return mapping.get(resolution, 'M1')


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone convenience
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ticker(ticker: str,
                resolution: str = '15m',
                days_back: int = 7,
                **kwargs) -> go.Figure:
    """
    One-liner: fetch data, analyze, and plot.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. "NVDA").
    resolution : str
        Timeframe to plot (default "15m").
    days_back : int
        Days of history to fetch (default 7).
    **kwargs
        Forwarded to ``plot_analysis()``.

    Returns
    -------
    go.Figure
    """
    from strategies.key_levels.analyzer import quick_analyze
    result = quick_analyze(ticker, resolution=resolution, days_back=days_back)
    return plot_analysis(result, resolution=resolution, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    res = sys.argv[2] if len(sys.argv) > 2 else '15m'
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    plot_ticker(ticker, resolution=res, days_back=days)
