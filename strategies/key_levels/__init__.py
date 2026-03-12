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
]

__version__ = '2.0.0'
