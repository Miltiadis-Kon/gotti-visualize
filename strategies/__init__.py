"""
Strategies package initializer.

Creates a package namespace so modules under `strategies` can be
imported using package-style imports (e.g. `from strategies.key_levels ...`).
"""

from strategies.strat_baseplate import StrategyBaseplate, OpeningGap
from strategies.risk_management import (
    optimize_sl_tp,
    calculate_atr,
    calculate_trailing_stop,
    ATRRiskLevels,
    TRADING_STYLES,
)

__all__ = [
    'StrategyBaseplate',
    'OpeningGap',
    'optimize_sl_tp',
    'calculate_atr',
    'calculate_trailing_stop',
    'ATRRiskLevels',
    'TRADING_STYLES',
    'key_levels',
    'book_based',
]
