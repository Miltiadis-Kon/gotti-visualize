"""
Test file for Multi-Timeframe Key Levels Strategy

Tests the key levels detection for NVDA across multiple timeframes:
- 1D (Daily): 5 months lookback - Lime color
- 4H: 3 months lookback - Cyan color
- 1H: 1 month lookback - Yellow color
- 15M: 1 week lookback - Magenta color
- 5M: 2 days lookback - Orange color (NEW)
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.key_levels import (
    KeyLevels, 
    run_key_levels_analysis, 
    run_merged_key_levels,
    run_fibonacci_analysis,
    FIBONACCI_LEVELS,
    FIBONACCI_THRESHOLD,
    FIBONACCI_IMPORTANCE
)
from datetime import datetime


def test_nvda_multi_timeframe():
    """
    Test key levels detection for NVDA across all timeframes.
    """
    print("=" * 70)
    print("MULTI-TIMEFRAME KEY LEVELS ANALYSIS - NVDA")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTimeframe Configuration:")
    print("  - 1D  (Daily):  5 months   → Lime")
    print("  - 4H  (4-Hour): 3 months   → Cyan")
    print("  - 1H  (1-Hour): 1 month    → Yellow")
    print("  - 15m (15-Min): 1 week     → Magenta")
    print("  - 5m  (5-Min):  2 days     → Orange")
    print("=" * 70)
    
    # Run the multi-timeframe analysis with all timeframes
    levels_df = run_key_levels_analysis(
        ticker="NVDA",
        timeframes=['1d', '4h', '1h', '15m', '5m'],
        tolerance_pct=0.5,
        pivot_lookback=10,
        show_plot=True,
        save_html=False
    )
    
    # Print summary by resolution
    print("\n" + "=" * 70)
    print("SUMMARY BY RESOLUTION")
    print("=" * 70)
    
    for resolution in ['1D', '4H', '1H', '15m', '5m']:
        res_levels = levels_df[levels_df['resolution'] == resolution]
        support_count = len(res_levels[res_levels['type'] == 'support'])
        resistance_count = len(res_levels[res_levels['type'] == 'resistance'])
        print(f"\n{resolution} Timeframe:")
        print(f"  Total levels: {len(res_levels)}")
        print(f"  - Support: {support_count}")
        print(f"  - Resistance: {resistance_count}")
    
    return levels_df


def test_merged_levels():
    """
    Test merged key levels with importance scores.
    """
    print("\n" + "=" * 70)
    print("MERGED KEY LEVELS WITH IMPORTANCE")
    print("=" * 70)
    
    merged_df = run_merged_key_levels(
        ticker="NVDA",
        timeframes=['1d', '4h', '1h', '15m', '5m'],
        price_threshold=0.5,
        show_plot=False
    )
    
    return merged_df


def test_fibonacci_analysis():
    """
    Test Fibonacci retracement levels between key levels.
    """
    print("\n" + "=" * 70)
    print("FIBONACCI RETRACEMENT ANALYSIS")
    print("=" * 70)
    print(f"Fibonacci Levels: {[f'{l*100:.1f}%' for l in FIBONACCI_LEVELS]}")
    print(f"Threshold: {FIBONACCI_THRESHOLD*100:.0f}%")
    print(f"Min Importance: {FIBONACCI_IMPORTANCE}")
    print("=" * 70)
    
    merged_df, fib_df = run_fibonacci_analysis(
        ticker="NVDA",
        timeframes=['1d', '4h', '1h', '15m', '5m'],
        show_plot=False
    )
    
    return merged_df, fib_df


if __name__ == "__main__":
    # Run multi-timeframe test
    levels = test_nvda_multi_timeframe()
    
    # Run merged levels test
    merged = test_merged_levels()
    
    # Run Fibonacci analysis
    merged, fib = test_fibonacci_analysis()
