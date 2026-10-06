"""
Plots module — Visualization pipeline for trading strategies.

New API (recommended)
---------------------
    from plots import ChartBuilder, get_renderer
    from plots import PlotSpec, CandlestickLayer, KeyLevelsLayer, TradeMarkersLayer, IndicatorLayer

    spec = (
        ChartBuilder("NVDA", chart_df, timeframe="1D")
        .add_candlesticks()
        .add_key_levels(levels_df, min_importance=3)
        .add_trades(trades)
        .build()
    )
    fig = get_renderer("plotly").render(spec)
    fig.show()
"""

# ── New plotting pipeline ────────────────────────────────────────────────────
from plots.plot_spec import (
    PlotSpec,
    PlotLayer,
    CandlestickLayer,
    KeyLevelsLayer,
    TradeMarkersLayer,
    IndicatorLayer,
)
from plots.chart_builder import ChartBuilder
from plots.renderers import get_renderer, BaseChartRenderer

__all__ = [
    # New API
    'PlotSpec',
    'PlotLayer',
    'CandlestickLayer',
    'KeyLevelsLayer',
    'TradeMarkersLayer',
    'IndicatorLayer',
    'ChartBuilder',
    'get_renderer',
    'BaseChartRenderer',
]
