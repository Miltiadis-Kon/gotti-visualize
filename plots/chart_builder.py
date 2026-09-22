"""
Chart Builder (Builder Pattern)

Provides a fluent API for constructing a PlotSpec step-by-step.
Strategies, dashboards, and scripts all use this single entry-point
instead of manually assembling PlotLayer objects.

Example usage
-------------
    from plots.chart_builder import ChartBuilder

    spec = (
        ChartBuilder("NVDA", chart_df, timeframe="1D")
        .add_candlesticks()
        .add_key_levels(levels_df, min_importance=3)
        .add_trades(trades, show_tp_sl=True, show_zones=True)
        .add_indicator("SMA 50", sma_series, color="#FFD700")
        .set_theme("tradingview_dark")
        .set_height(700)
        .build()
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from plots.plot_spec import (
    PlotSpec,
    PlotLayer,
    CandlestickLayer,
    KeyLevelsLayer,
    TradeMarkersLayer,
    IndicatorLayer,
)


class ChartBuilder:
    """Fluent builder that assembles a :class:`PlotSpec`.

    Parameters
    ----------
    ticker : str
        Asset symbol, e.g. ``"NVDA"``.
    chart_df : pd.DataFrame
        OHLCV DataFrame.  The builder stores a reference; it is passed
        verbatim to :class:`CandlestickLayer`.
    timeframe : str
        Primary timeframe label, e.g. ``"1D"``, ``"4H"`` (display only).
    """

    def __init__(
        self,
        ticker: str,
        chart_df: pd.DataFrame,
        timeframe: str = "1D",
    ) -> None:
        self._ticker = ticker
        self._chart_df = chart_df
        self._timeframe = timeframe
        self._layers: List[PlotLayer] = []
        self._theme: str = "tradingview_dark"
        self._height: int = 700
        self._title: str = ""
        self._extra: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Layer-adding methods
    # ------------------------------------------------------------------

    def add_candlesticks(self) -> "ChartBuilder":
        """Add a candlestick layer using the DataFrame supplied at construction."""
        self._layers.append(
            CandlestickLayer(
                data=self._chart_df,
                ticker=self._ticker,
                timeframe=self._timeframe,
            )
        )
        return self

    def add_key_levels(
        self,
        levels_df: pd.DataFrame,
        *,
        min_importance: int = 1,
        show_support: bool = True,
        show_resistance: bool = True,
        proximity_filter: float = 0.25,
        visible: bool = True,
    ) -> "ChartBuilder":
        """Add a support/resistance key-levels layer.

        Parameters
        ----------
        levels_df : pd.DataFrame
            Merged levels with columns: level_price, type, importance, touch_count.
        min_importance : int
            Minimum importance score to include.
        show_support : bool
            Show support levels.
        show_resistance : bool
            Show resistance levels.
        proximity_filter : float
            Fractional distance from current price (0.25 = ±25%).
        visible : bool
            Layer visibility toggle.
        """
        if levels_df is None or levels_df.empty:
            return self
        self._layers.append(
            KeyLevelsLayer(
                levels_df=levels_df,
                min_importance=min_importance,
                show_support=show_support,
                show_resistance=show_resistance,
                proximity_filter=proximity_filter,
                visible=visible,
            )
        )
        return self

    def add_trades(
        self,
        trades: List[Any],
        *,
        show_tp_sl: bool = True,
        show_zones: bool = True,
        show_connection_line: bool = True,
        visible: bool = True,
    ) -> "ChartBuilder":
        """Add a trade-markers layer.

        Parameters
        ----------
        trades : list of Trade
            Trade objects from ``strategies.trade_tracker``.
        show_tp_sl : bool
            Draw TP/SL horizontal dashed lines.
        show_zones : bool
            Shade the trade duration zone.
        show_connection_line : bool
            Draw a dotted entry-to-exit line.
        visible : bool
            Layer visibility toggle.
        """
        if not trades:
            return self
        self._layers.append(
            TradeMarkersLayer(
                trades=trades,
                show_tp_sl=show_tp_sl,
                show_zones=show_zones,
                show_connection_line=show_connection_line,
                visible=visible,
            )
        )
        return self

    def add_indicator(
        self,
        name: str,
        data: Any,
        *,
        color: str = "#FFD700",
        width: int = 1,
        dash: str = "solid",
        row: int = 1,
        visible: bool = True,
    ) -> "ChartBuilder":
        """Add a generic indicator overlay (SMA, EMA, RSI, ATR, …).

        Parameters
        ----------
        name : str
            Display name, e.g. ``"SMA 50"``.
        data : pd.Series or pd.DataFrame
            Indicator values aligned to the candlestick time axis.
        color : str
            CSS / Plotly colour string.
        width : int
            Line width in pixels.
        dash : str
            Plotly dash style: ``"solid"``, ``"dash"``, ``"dot"``.
        row : int
            Subplot row (1 = main panel, 2+ = sub-panels).
        visible : bool
            Layer visibility toggle.
        """
        self._layers.append(
            IndicatorLayer(
                name=name,
                data=data,
                color=color,
                width=width,
                dash=dash,
                row=row,
                visible=visible,
            )
        )
        return self

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------

    def set_theme(self, theme: str) -> "ChartBuilder":
        """Set the chart theme.  Currently ``"tradingview_dark"`` only."""
        self._theme = theme
        return self

    def set_height(self, height: int) -> "ChartBuilder":
        """Override the default chart height (700 px)."""
        self._height = height
        return self

    def set_title(self, title: str) -> "ChartBuilder":
        """Override the auto-generated chart title."""
        self._title = title
        return self

    def set_extra(self, **kwargs: Any) -> "ChartBuilder":
        """Pass renderer-specific options through :attr:`PlotSpec.extra`.

        Example::

            builder.set_extra(rangeslider=False, hovermode="x unified")
        """
        self._extra.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> PlotSpec:
        """Finalise and return the :class:`PlotSpec`.

        Raises
        ------
        ValueError
            If no candlestick layer has been added (the chart would be
            empty without price data).
        """
        has_candles = any(isinstance(l, CandlestickLayer) for l in self._layers)
        if not has_candles:
            raise ValueError(
                "ChartBuilder.build() requires at least one candlestick layer. "
                "Call .add_candlesticks() before .build()."
            )
        return PlotSpec(
            ticker=self._ticker,
            timeframe=self._timeframe,
            layers=list(self._layers),  # defensive copy
            theme=self._theme,
            height=self._height,
            title=self._title,
            extra=dict(self._extra),
        )

    # ------------------------------------------------------------------
    # Alternate constructor — strategy-driven flow
    # ------------------------------------------------------------------

    @classmethod
    def from_plot_spec(cls, spec: PlotSpec) -> "ChartBuilder":
        """Construct a builder pre-populated from an existing :class:`PlotSpec`.

        Useful when you want to start from a strategy's ``get_plot_spec()``
        output and add extra layers before rendering::

            builder = ChartBuilder.from_plot_spec(strategy.get_plot_spec())
            builder.add_indicator("SMA 20", sma_series)
            fig = get_renderer("plotly").render(builder.build())
        """
        cs = spec.candlestick_layer
        if cs is None or cs.data is None:
            raise ValueError("PlotSpec must contain a CandlestickLayer with data.")

        builder = cls(
            ticker=spec.ticker,
            chart_df=cs.data,
            timeframe=spec.timeframe,
        )
        builder._layers = list(spec.layers)
        builder._theme = spec.theme
        builder._height = spec.height
        builder._title = spec.title
        builder._extra = dict(spec.extra)
        return builder
