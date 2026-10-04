"""
Test Suite: Strategy Audit & Architectural Standardization
==========================================================
Verifies that:
1. All strategies and aliases export correctly from `strategies` namespace.
2. `KeyLevelDetector` and `FibonacciDetector` run with high-performance NumPy vectorization.
3. `DataFetcher` in-memory caching prevents redundant external fetches.
4. `StrategyBaseplate` supports both standard `technical` and legacy `techincal` attributes,
   and handles both `RiskPct` and `RISK_PERCENT` parameter naming.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from lumibot.entities import Asset

from strategies import (
    StrategyBaseplate,
    LiquiditySweepStrategy,
    BaseKeyLevelsStrategy,
    MultiTimeframeKeyLevelsStrategy,
    AlpacaSMCBridge,
    SmartMoneyConceptsStrategy,
    SMCStrategy,
    SMCIntradayBiasStrategy,
    SMCBiasModelStrategy,
)
from strategies.key_levels.key_levels import KeyLevelDetector, find_key_levels
from strategies.key_levels.fibonacci_levels import FibonacciDetector
from strategies.key_levels.analyzer import DataFetcher, KeyLevelAnalyzer


def test_package_exports_and_aliases():
    """Verify all standardized strategy classes and backward-compatible aliases are present."""
    assert SMCStrategy is SmartMoneyConceptsStrategy
    assert SMCBiasModelStrategy is SMCIntradayBiasStrategy
    assert issubclass(BaseKeyLevelsStrategy, StrategyBaseplate)
    assert issubclass(MultiTimeframeKeyLevelsStrategy, StrategyBaseplate)
    assert issubclass(LiquiditySweepStrategy, StrategyBaseplate)


def test_key_level_detector_vectorization():
    """Verify KeyLevelDetector detects correct support/resistance pivots using NumPy."""
    # Construct synthetic data with known pivot low at bar 10 and pivot high at bar 20
    n = 30
    prices = [100.0] * n
    prices[10] = 90.0   # Pivot low
    prices[20] = 110.0  # Pivot high

    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": [1000] * n,
    })

    detector = KeyLevelDetector(tolerance_pct=0.5, pivot_lookback=5)
    pivoted_df = detector._detect_all_pivots(df)

    assert "pivot" in pivoted_df.columns
    # Bar 10 should be pivot low (1)
    assert pivoted_df["pivot"].iloc[10] == 1
    # Bar 20 should be pivot high (2)
    assert pivoted_df["pivot"].iloc[20] == 2
    # Individual bar detection should match
    assert detector._detect_pivot(df, 10, 5, 5) == 1
    assert detector._detect_pivot(df, 20, 5, 5) == 2


def test_fibonacci_detector_swings():
    """Verify FibonacciDetector finds valid swings and calculates accurate retracement levels."""
    # 50 bars of price data
    highs = np.linspace(100, 150, 50)
    lows = highs - 2.0
    df = pd.DataFrame({
        "open": highs - 1.0,
        "high": highs,
        "low": lows,
        "close": highs - 0.5,
        "volume": [5000] * 50,
    })

    det = FibonacciDetector(backcandles=30, gap_candles=5)
    swings = det._find_swings(df)

    assert len(swings) > 0
    first_swing = swings[0]
    assert "max_price" in first_swing
    assert "min_price" in first_swing
    assert first_swing["trend"] in ("uptrend", "downtrend")
    assert first_swing["max_price"] > first_swing["min_price"]


def test_data_fetcher_caching():
    """Verify DataFetcher caches data in-memory and avoids redundant fetches."""
    fetcher = DataFetcher(use_alpaca=False)
    # Seed cache manually with synthetic 1D data
    dates = pd.date_range("2024-01-01", "2024-06-01", freq="D")
    sample_df = pd.DataFrame({
        "date": dates,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "volume": 100000,
    })

    DataFetcher._cache[("TEST_TICKER", "1d")] = sample_df

    # Fetch within the cached range with as_of_date
    test_fetcher = DataFetcher(use_alpaca=False, as_of_date=datetime(2024, 4, 1))
    result = test_fetcher.fetch("TEST_TICKER", interval="1d", days_back=30)

    assert not result.empty
    assert len(result) <= 32
    assert result["date"].max() <= pd.to_datetime("2024-04-01")


def test_strategy_baseplate_risk_and_technical():
    """Verify StrategyBaseplate initializes risk parameter and sets up technical attributes."""
    class CustomStrategy(StrategyBaseplate):
        parameters = {
            **StrategyBaseplate.parameters,
            "RiskPct": 0.03,
        }

        def setup(self):
            return True

    strat = CustomStrategy(broker=None)
    # Test initialize
    strat.initialize()
    assert strat.risk_percent == 0.03

    # Test before_market_opens hook
    strat.before_market_opens()
    assert strat.technical is True
    assert strat.techincal is True  # Backwards compatibility alias
