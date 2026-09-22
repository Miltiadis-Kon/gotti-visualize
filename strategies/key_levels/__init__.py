"""
Key Levels Package

A modular toolkit for detecting support/resistance levels and Fibonacci retracements.

Modules:
--------
- key_levels: Pivot-based support/resistance detection
- fibonacci_levels: Swing-based Fibonacci retracement detection
- analyzer: Data fetching and orchestration

Quick Start:
------------
    from strategies.key_levels import analyze
    
    # Full analysis
    result = analyze("NVDA")
    print(result.merged_levels)
    print(result.trade_setups)
    
    # Single resolution analysis
    from strategies.key_levels import quick_analyze
    result = quick_analyze("AAPL", resolution="1D", days_back=100)

Direct Functions:
-----------------
    from strategies.key_levels import find_key_levels, find_fibonacci_levels
    
    # Use with your own DataFrame
    levels = find_key_levels(df, resolution="1D", days_back=150)
    fibs = find_fibonacci_levels(df, resolution="1D", days_back=150)
"""

# Key Levels Detection
from .key_levels import (
    KeyLevelDetector,
    find_key_levels,
    merge_key_levels,
    RESOLUTION_IMPORTANCE,
    DEFAULT_PRICE_THRESHOLD
)

# Fibonacci Levels Detection
from .fibonacci_levels import (
    FibonacciDetector,
    find_fibonacci_levels,
    get_fibonacci_trade_setups,
    FIBONACCI_LEVELS,
    FIBONACCI_ENTRY_LEVEL,
    FIBONACCI_SL_LEVEL,
    FIB_LEVEL_COLUMNS,
    DEFAULT_BACKCANDLES,
    DEFAULT_GAP_CANDLES
)

# Analyzer (Data Fetcher + Orchestration)
from .analyzer import (
    DataFetcher,
    KeyLevelAnalyzer,
    AnalysisResult,
    analyze,
    quick_analyze,
    RESOLUTION_CONFIG
)

# Chart (Plotly visualization)
from .chart import (
    plot_analysis,
    plot_ticker,
    FIB_COLORS,
    SR_COLORS,
)

__all__ = [
    # Key Levels
    'KeyLevelDetector',
    'find_key_levels',
    'merge_key_levels',
    'RESOLUTION_IMPORTANCE',
    'DEFAULT_PRICE_THRESHOLD',
    
    # Fibonacci
    'FibonacciDetector',
    'find_fibonacci_levels',
    'get_fibonacci_trade_setups',
    'FIBONACCI_LEVELS',
    'FIBONACCI_ENTRY_LEVEL',
    'FIBONACCI_SL_LEVEL',
    'FIB_LEVEL_COLUMNS',
    'DEFAULT_BACKCANDLES',
    'DEFAULT_GAP_CANDLES',
    
    # Analyzer
    'DataFetcher',
    'KeyLevelAnalyzer',
    'AnalysisResult',
    'analyze',
    'quick_analyze',
    'RESOLUTION_CONFIG',
    
    # Chart
    'plot_analysis',
    'plot_ticker',
    'FIB_COLORS',
    'SR_COLORS',

    # v1 backward-compatibility
    'KeyLevels',
    'TIMEFRAME_LOOKBACK',
    'TIMEFRAME_IMPORTANCE',
    'TIMEFRAME_COLORS',
    'PRICE_THRESHOLD',
    'FIBONACCI_THRESHOLD',
    'FIBONACCI_IMPORTANCE',
]


__version__ = '2.0.0'

# ── v1 backward-compatibility shims ────────────────────────────────────────
# The v1 API used a class-based KeyLevels object and different constant names.
# These aliases keep old plot/strategy code working without a mass rename.

# Constant aliases
TIMEFRAME_LOOKBACK = {res: cfg['days'] for res, cfg in RESOLUTION_CONFIG.items()}
TIMEFRAME_IMPORTANCE = RESOLUTION_IMPORTANCE
TIMEFRAME_COLORS = {
    '1D': '#FF6B6B',
    '4H': '#FFD93D',
    '1H': '#6BCB77',
    '15m': '#4D96FF',
    '5m': '#C77DFF',
}
PRICE_THRESHOLD = DEFAULT_PRICE_THRESHOLD
FIBONACCI_THRESHOLD = 0.20   # 20% — default range filter for fib patterns
FIBONACCI_IMPORTANCE = 2     # minimum resolution importance for fib levels


class KeyLevels:
    """
    v1 compatibility wrapper around the v2 KeyLevelAnalyzer API.

    Usage (same as v1):
        kl = KeyLevels(ticker='NVDA', use_alpaca=True)
        kl.find_all_key_levels(timeframes=['1d', '4h'])
        merged = kl.get_merged_levels(price_threshold=0.5)
        fibs = kl.calculate_fibonacci_levels(merged, fib_threshold=0.2)
    """

    def __init__(self, ticker: str, use_alpaca: bool = True):
        self.ticker = ticker.upper()
        self._analyzer = KeyLevelAnalyzer(use_alpaca=use_alpaca)
        self._result = None
        self.timeframe_data: dict = {}
        self.all_levels_df = None

    def find_all_key_levels(self, timeframes=None):
        """Run analysis across all given timeframes (v2 resolution format or v1 interval format)."""
        # Map old yfinance interval names to v2 resolution keys
        _interval_to_res = {'1d': '1D', '4h': '4H', '1h': '1H', '15m': '15m', '5m': '5m'}
        if timeframes:
            resolutions = [_interval_to_res.get(tf, tf.upper()) for tf in timeframes]
        else:
            resolutions = ['1D', '4H', '1H', '15m', '5m']

        self._result = self._analyzer.analyze(self.ticker, resolutions=resolutions)
        self.timeframe_data = self._result.candle_data
        self.all_levels_df = self._result.key_levels

    def get_merged_levels(self, price_threshold: float = DEFAULT_PRICE_THRESHOLD):
        """Return merged S/R levels DataFrame (v2: merged_levels)."""
        if self._result is None:
            self.find_all_key_levels()
        return self._result.merged_levels

    def calculate_fibonacci_levels(self, merged_df=None, fib_threshold: float = 0.20,
                                   min_importance: int = 1):
        """Return Fibonacci trade setups DataFrame (v2: trade_setups)."""
        if self._result is None:
            self.find_all_key_levels()
        return self._result.trade_setups

    def get_all_data(self):
        """Return the raw AnalysisResult (v2 object)."""
        return self._result
