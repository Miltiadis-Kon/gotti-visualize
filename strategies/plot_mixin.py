"""
PlottableStrategyMixin — Strategy Pattern (Strategy side)

A lightweight mixin that any Lumibot Strategy class can inherit to gain
plotting capabilities backed by the new ChartBuilder + Renderer pipeline.

Usage example (opt-in, added to any strategy class)
----------------------------------------------------

    class LongTrendHighMomentum(Strategy, PlottableStrategyMixin):

        def get_plot_spec(self) -> PlotSpec:
            builder = ChartBuilder(self._plot_ticker, self._chart_df, timeframe="1D")
            builder.add_candlesticks()
            for scatter in self._scheduled_trades:
                builder.add_trades([scatter])
            return builder.build()

        def on_strategy_end(self):
            self.render_plot(save_html=f"logs/charts/{self._plot_ticker}_chart.html")

Design notes
------------
- The mixin carries *no state* — all state lives in the host strategy.
- ``get_plot_spec()`` is the only method that subclasses must implement.
- ``render_plot()`` is the convenience method that ties builder →
  renderer → output together.
- ``save_plot_html()`` is a thin shortcut for ``render_plot(show=False, save_html=...)``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from plots.plot_spec import PlotSpec


class PlottableStrategyMixin:
    """Mixin that gives any Lumibot Strategy class first-class plotting support.

    Inherit this alongside ``lumibot.strategies.Strategy``::

        class MyStrategy(Strategy, PlottableStrategyMixin):
            ...

    Then implement :meth:`get_plot_spec` and call :meth:`render_plot`
    (typically in ``on_strategy_end`` or ``on_abrupt_closing``).
    """

    # ------------------------------------------------------------------
    # Contract — subclasses must implement this
    # ------------------------------------------------------------------

    def get_plot_spec(self) -> "PlotSpec":
        """Return a :class:`~plots.plot_spec.PlotSpec` for this strategy.

        Override in subclasses.  The default raises ``NotImplementedError``
        so that a clear message is surfaced if the method is forgotten.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_plot_spec() "
            "to use PlottableStrategyMixin."
        )

    # ------------------------------------------------------------------
    # Convenience rendering helpers
    # ------------------------------------------------------------------

    def render_plot(
        self,
        *,
        backend: str = "plotly",
        show: bool = True,
        save_html: Optional[str] = None,
    ) -> Any:
        """Render the strategy's plot.

        Parameters
        ----------
        backend : str
            Renderer backend — currently only ``"plotly"`` is supported.
        show : bool
            If ``True``, open the chart in the browser
            (``fig.show()``).  Set to ``False`` when running in
            headless / CI environments.
        save_html : str, optional
            Absolute or relative path to write an HTML chart file.
            Parent directories are created automatically.

        Returns
        -------
        Any
            The backend-specific figure object (e.g. ``go.Figure``).
        """
        from plots.renderers import get_renderer

        spec = self.get_plot_spec()
        renderer = get_renderer(backend)
        fig = renderer.render(spec)

        if save_html:
            _ensure_dir(save_html)
            fig.write_html(save_html)
            print(f"[PlottableStrategyMixin] Chart saved -> {save_html}")

        if show:
            renderer.show(fig)

        return fig

    def save_plot_html(self, path: str, *, backend: str = "plotly") -> Any:
        """Shortcut for ``render_plot(show=False, save_html=path)``.

        Parameters
        ----------
        path : str
            File path for the HTML output.
        backend : str
            Renderer backend.

        Returns
        -------
        Any
            The rendered figure.
        """
        return self.render_plot(backend=backend, show=False, save_html=path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(filepath: str) -> None:
    """Create parent directories for *filepath* if they do not exist."""
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)
