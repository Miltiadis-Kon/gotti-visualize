"""
Plot Specification — Data Model (Strategy Pattern)

Defines the pure-data layer that sits between strategies and renderers.
Strategies produce a PlotSpec describing WHAT to visualize; renderers
consume it to decide HOW to draw it.

No Plotly / chart-library imports live here intentionally — this module
must remain importable even when optional rendering dependencies are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Layer definitions (Composite Pattern — each layer is an independent element)
# ---------------------------------------------------------------------------

@dataclass
class PlotLayer:
    """Base class for all composable chart layers.

    Attributes
    ----------
    visible : bool
        Whether this layer is rendered by default.
    label : str
        Optional human-readable label shown in the legend / tooltip.
    """
    visible: bool = True
    label: str = ""


@dataclass
class CandlestickLayer(PlotLayer):
    """OHLCV candlestick data layer.

    Attributes
    ----------
    data : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close.
        The time column can be any of: Date, Datetime, datetime,
        timestamp — or the DataFrame index.
    ticker : str
        Asset symbol, used in the trace name and tooltip.
    timeframe : str
        Human-readable timeframe label, e.g. ``"1D"``, ``"4H"``.
    """
    data: Optional[pd.DataFrame] = None
    ticker: str = ""
    timeframe: str = "1D"


@dataclass
class KeyLevelsLayer(PlotLayer):
    """Support / resistance key-levels layer.

    Attributes
    ----------
    levels_df : pd.DataFrame
        Merged levels DataFrame with columns:
        level_price, type (support/resistance), importance, touch_count.
    min_importance : int
        Levels below this importance score are filtered out.
    show_support : bool
        Whether to render support levels.
    show_resistance : bool
        Whether to render resistance levels.
    proximity_filter : float
        Only render levels within this fractional distance of the
        current price (e.g. 0.25 = ±25%).  Set to 1.0 to disable.
    """
    levels_df: Optional[pd.DataFrame] = None
    min_importance: int = 1
    show_support: bool = True
    show_resistance: bool = True
    proximity_filter: float = 0.25


@dataclass
class TradeMarkersLayer(PlotLayer):
    """Entry/exit trade-marker layer.

    Attributes
    ----------
    trades : list
        List of ``Trade`` objects (from ``strategies.trade_tracker``).
    show_tp_sl : bool
        Whether to draw TP/SL horizontal dashed lines per trade.
    show_zones : bool
        Whether to shade the trade duration zone (green=win, red=loss).
    show_connection_line : bool
        Whether to draw a dotted line from entry marker to exit marker.
    """
    trades: List[Any] = field(default_factory=list)
    show_tp_sl: bool = True
    show_zones: bool = True
    show_connection_line: bool = True


@dataclass
class IndicatorLayer(PlotLayer):
    """Generic technical-indicator overlay.

    Attributes
    ----------
    name : str
        Display name, e.g. ``"SMA 50"``, ``"RSI 14"``.
    data : pd.Series or pd.DataFrame
        Indicator values aligned to the same time axis as the candlesticks.
    color : str
        CSS / Plotly colour string for the line.
    width : int
        Line width in pixels.
    dash : str
        Plotly dash style: ``"solid"``, ``"dash"``, ``"dot"``, ``"dashdot"``.
    row : int
        Subplot row (1 = main price chart, 2+ = sub-panels).  Renderers
        that do not support sub-panels may ignore this.
    """
    name: str = ""
    data: Any = None          # pd.Series or pd.DataFrame
    color: str = "#FFD700"
    width: int = 1
    dash: str = "solid"
    row: int = 1


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class PlotSpec:
    """Top-level plot specification produced by a strategy or ChartBuilder.

    This is the single artefact passed between the strategy/builder layer
    and the renderer layer.  Renderers must not mutate it.

    Attributes
    ----------
    ticker : str
        Asset symbol.
    timeframe : str
        Primary timeframe label, e.g. ``"1D"``, ``"4H"``.
    layers : list[PlotLayer]
        Ordered list of layers to render (rendered bottom-to-top).
    theme : str
        Named theme understood by renderers.  Currently:
        ``"tradingview_dark"`` (default).
    height : int
        Chart height in pixels.
    title : str
        Override for the chart title.  If empty, renderers construct a
        default title from ticker / timeframe / current price.
    extra : dict
        Arbitrary renderer-specific options (e.g. ``{"rangeslider": False}``).
    """
    ticker: str = ""
    timeframe: str = "1D"
    layers: List[PlotLayer] = field(default_factory=list)
    theme: str = "tradingview_dark"
    height: int = 700
    title: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_layers_of_type(self, layer_type: type) -> List[PlotLayer]:
        """Return all layers matching a given type."""
        return [l for l in self.layers if isinstance(l, layer_type)]

    @property
    def candlestick_layer(self) -> Optional[CandlestickLayer]:
        """Return the first CandlestickLayer, or None."""
        layers = self.get_layers_of_type(CandlestickLayer)
        return layers[0] if layers else None

    @property
    def has_trades(self) -> bool:
        """True if there is at least one TradeMarkersLayer with trades."""
        return any(
            isinstance(l, TradeMarkersLayer) and l.trades
            for l in self.layers
        )
