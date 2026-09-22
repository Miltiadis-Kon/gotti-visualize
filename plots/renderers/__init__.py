"""
Renderers package — Template Method pattern.

Provides a single public factory function:

    get_renderer(backend="plotly") -> BaseChartRenderer

Currently supports only "plotly".  Adding a new backend is a matter of
creating a new module and adding an entry to ``_REGISTRY``.
"""

from __future__ import annotations

from plots.renderers.base import BaseChartRenderer

# Registry maps backend name → lazy import path
_REGISTRY: dict[str, str] = {
    "plotly": "plots.renderers.plotly_renderer.PlotlyRenderer",
}


def get_renderer(backend: str = "plotly") -> BaseChartRenderer:
    """Return a concrete chart renderer for the requested backend.

    Parameters
    ----------
    backend : str
        Name of the rendering backend.  Only ``"plotly"`` is currently
        supported.

    Returns
    -------
    BaseChartRenderer
        A concrete renderer instance ready to call ``.render(spec)``.

    Raises
    ------
    ValueError
        If the backend name is not recognised.

    Example
    -------
        from plots.renderers import get_renderer
        fig = get_renderer("plotly").render(spec)
        fig.show()
    """
    if backend not in _REGISTRY:
        supported = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise ValueError(
            f"Unknown renderer backend: {backend!r}. Supported: {supported}"
        )

    import importlib
    module_path, cls_name = _REGISTRY[backend].rsplit(".", 1)
    module = importlib.import_module(module_path)
    renderer_cls = getattr(module, cls_name)
    return renderer_cls()


__all__ = ["get_renderer", "BaseChartRenderer"]
