"""
Strategies package initializer.

Creates a package namespace so modules under `strategies` can be
imported using package-style imports (e.g. `from strategies.key_levels ...`).
"""

from strategies.strat_baseplate import StrategyBaseplate
from strategies.liquidity_sweep import LiquiditySweepStrategy
from strategies.base_key_levels_strategy import BaseKeyLevelsStrategy
from strategies.multi_tf_strategy import MultiTimeframeKeyLevelsStrategy
from strategies.macd_strategy import MACDTradingStrategy
from strategies.alpaca_smc_bridge import AlpacaSMCBridge
from strategies.smc import SmartMoneyConceptsStrategy, SMCStrategy
from strategies.smc_bias_model import SMCIntradayBiasStrategy, SMCBiasModelStrategy
from strategies.risk_management import (
    optimize_sl_tp,
    calculate_atr,
    calculate_trailing_stop,
    ATRRiskLevels,
    TRADING_STYLES,
)

__all__ = [
    'StrategyBaseplate',
    'LiquiditySweepStrategy',
    'BaseKeyLevelsStrategy',
    'MultiTimeframeKeyLevelsStrategy',
    'MACDTradingStrategy',
    'AlpacaSMCBridge',
    'SmartMoneyConceptsStrategy',
    'SMCStrategy',
    'SMCIntradayBiasStrategy',
    'SMCBiasModelStrategy',
    'optimize_sl_tp',
    'calculate_atr',
    'calculate_trailing_stop',
    'ATRRiskLevels',
    'TRADING_STYLES',
    'key_levels',
]
