"""
Abstract Base Chart Renderer (Template Method Pattern)

Defines the rendering skeleton. Concrete subclasses implement the
abstract methods for each layer type and backend.

The rendering lifecycle:
    1. _create_figure(spec)          ← backend-specific figure/chart object
    2. For each visible layer:
       _render_layer(fig, layer, spec) → dispatches to typed method
    3. _apply_layout(fig, spec)      ← apply title, axes, theme, height
    4. Return fig to caller

Concrete renderers only need to implement the abstract methods;
the orchestration in ``render()`` is inherited unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from plots.plot_spec import (
    PlotSpec,
    PlotLayer,
    CandlestickLayer,
    KeyLevelsLayer,
    TradeMarkersLayer,
    IndicatorLayer,
)


class BaseChartRenderer(ABC):
    """Abstract base for all chart renderers.

    Subclasses must implement:
      - :meth:`_create_figure`
      - :meth:`_render_candlestick`
      - :meth:`_render_key_levels`
      - :meth:`_render_trades`
      - :meth:`_render_indicator`
      - :meth:`_apply_layout`
    """

    # ------------------------------------------------------------------
    # Public API (the Template Method)
    # ------------------------------------------------------------------

    def render(self, spec: PlotSpec) -> Any:
        """Render the full chart described by *spec*.

        This is the **only** method that callers should call.  It
        orchestrates the rendering steps in the correct order and
        returns the backend-specific figure object.

        Parameters
        ----------
        spec : PlotSpec
            The plot specification produced by a strategy or builder.

        Returns
        -------
        Any
            Backend-specific figure (e.g. ``plotly.graph_objects.Figure``).
        """
        fig = self._create_figure(spec)

        for layer in spec.layers:
            if not layer.visible:
                continue
            self._render_layer(fig, layer, spec)

        self._apply_layout(fig, spec)
        return fig

    def show(self, fig: Any) -> None:
        """Display the rendered figure.

        Default implementation is a no-op; subclasses override for
        their specific show mechanism (e.g. ``fig.show()``).
        """
        pass

    # ------------------------------------------------------------------
    # Layer dispatcher (do not override in concrete classes)
    # ------------------------------------------------------------------

    def _render_layer(self, fig: Any, layer: PlotLayer, spec: PlotSpec) -> None:
        """Dispatch a layer to the appropriate typed render method."""
        if isinstance(layer, CandlestickLayer):
            self._render_candlestick(fig, layer, spec)
        elif isinstance(layer, KeyLevelsLayer):
            self._render_key_levels(fig, layer, spec)
        elif isinstance(layer, TradeMarkersLayer):
            self._render_trades(fig, layer, spec)
        elif isinstance(layer, IndicatorLayer):
            self._render_indicator(fig, layer, spec)
        else:
            # Unknown layer types are silently skipped so that new layer
            # types added to plot_spec don't crash older renderers.
            pass

    # ------------------------------------------------------------------
    # Abstract hooks — implement in concrete subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_figure(self, spec: PlotSpec) -> Any:
        """Create and return an empty backend figure object."""

    @abstractmethod
    def _render_candlestick(
        self, fig: Any, layer: CandlestickLayer, spec: PlotSpec
    ) -> None:
        """Add OHLCV candlestick traces to *fig*."""

    @abstractmethod
    def _render_key_levels(
        self, fig: Any, layer: KeyLevelsLayer, spec: PlotSpec
    ) -> None:
        """Add support/resistance level traces to *fig*."""

    @abstractmethod
    def _render_trades(
        self, fig: Any, layer: TradeMarkersLayer, spec: PlotSpec
    ) -> None:
        """Add trade entry/exit markers and TP/SL lines to *fig*."""

    @abstractmethod
    def _render_indicator(
        self, fig: Any, layer: IndicatorLayer, spec: PlotSpec
    ) -> None:
        """Add a technical-indicator line to *fig*."""

    @abstractmethod
    def _apply_layout(self, fig: Any, spec: PlotSpec) -> None:
        """Apply theme, title, axes configuration, and height to *fig*."""
