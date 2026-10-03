"""
Tests to verify audit fixes and StrategyBaseplate conformity across all 27 strategies.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from strategies.strat_baseplate import StrategyBaseplate

# Book Based
from strategies.book_based.opening_gap import OpeningGap
from strategies.book_based.long_mean_reversion_selloff import LongMeanReversionSelloff
from strategies.book_based.long_mean_reversion_high_ADX_reversal import LongMeanReversionHighADXReversal
from strategies.book_based.long_trend_high_momentum import LongTrendHighMomentum
from strategies.book_based.long_trend_low_volatility import LongTrendLowVolatility
from strategies.book_based.the_catastrophe_hedge import TheCatastropheHedge
from strategies.book_based.short_mean_reversion_high_6D_surge import ShortMeanReversionHigh6DSurge
from strategies.book_based.short_rsi_thrust import ShortRSIThrust
from strategies.book_based.signal_key_levels_strategy import SignalKeyLevelsStrategy

# Book Based 2
from strategies.book_based_2.gap_setup_long import OpeningGapLong as BB2GapLong
from strategies.book_based_2.gap_setup_short import OpeningGapShort as BB2GapShort
from strategies.book_based_2.hcr_breakout_long import HCRBreakoutLong as BB2HCRLong
from strategies.book_based_2.hcr_breakout_short import HCRBreakoutShort as BB2HCRShort
from strategies.book_based_2.retrace_long import RetraceLong as BB2RetraceLong
from strategies.book_based_2.retrace_short import RetraceShort as BB2RetraceShort
from strategies.book_based_2.rsi_setup import RSISetup as BB2RSISetup
from strategies.book_based_2.fib_retrace_swing import FibRetraceSwing as BB2FibRetrace

# Swing
from strategies.swing.gap_setup_long import GapSetupLongSwing
from strategies.swing.gap_setup_short import GapSetupShortSwing
from strategies.swing.hcr_breakout_long import HCRBreakoutLongSwing
from strategies.swing.hcr_breakout_short import HCRBreakoutShortSwing
from strategies.swing.retrace_long import RetraceLongSwing
from strategies.swing.retrace_short import RetraceShortSwing
from strategies.swing.rsi_setup import RSISetupSwing
from strategies.swing.fib_retrace_swing import FibRetraceSwing as SwingFibRetrace

# Root Key Level strategies
from strategies.base_key_levels_strategy import BaseKeyLevelsStrategy
from strategies.multi_tf_strategy import MultiTimeframeKeyLevelsStrategy
from strategies.trade_tracker import TradeTracker


ALL_STRATEGIES = [
    OpeningGap,
    LongMeanReversionSelloff,
    LongMeanReversionHighADXReversal,
    LongTrendHighMomentum,
    LongTrendLowVolatility,
    TheCatastropheHedge,
    ShortMeanReversionHigh6DSurge,
    ShortRSIThrust,
    SignalKeyLevelsStrategy,
    BB2GapLong,
    BB2GapShort,
    BB2HCRLong,
    BB2HCRShort,
    BB2RetraceLong,
    BB2RetraceShort,
    BB2RSISetup,
    BB2FibRetrace,
    GapSetupLongSwing,
    GapSetupShortSwing,
    HCRBreakoutLongSwing,
    HCRBreakoutShortSwing,
    RetraceLongSwing,
    RetraceShortSwing,
    RSISetupSwing,
    SwingFibRetrace,
    BaseKeyLevelsStrategy,
    MultiTimeframeKeyLevelsStrategy,
]


@pytest.mark.parametrize("strat_cls", ALL_STRATEGIES)
def test_all_strategies_inherit_baseplate(strat_cls):
    """Verify every single strategy inherits StrategyBaseplate."""
    assert issubclass(strat_cls, StrategyBaseplate), f"{strat_cls.__name__} does not inherit StrategyBaseplate"


@pytest.mark.parametrize("strat_cls", ALL_STRATEGIES)
def test_baseplate_parameters_presence(strat_cls):
    """Verify parameters dictionary contains baseplate keys."""
    assert hasattr(strat_cls, "parameters")
    assert "TradingStyle" in strat_cls.parameters or "TradingStyle" in StrategyBaseplate.parameters


def test_strat_baseplate_get_atr():
    """Verify get_atr handles both signature styles without crashing."""
    strat = StrategyBaseplate()
    # Mock ticker_bars
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, 30),
        "high": np.linspace(102, 112, 30),
        "low": np.linspace(98, 108, 30),
        "close": np.linspace(101, 111, 30),
        "volume": np.ones(30) * 1000000,
    }, index=dates)
    strat.ticker_bars = df

    # Standard call
    atr1 = strat.get_atr(length=14)
    assert atr1 > 0

    # Call with df
    atr2 = strat.get_atr(df=df, length=14)
    assert atr2 > 0

    # Positional ticker arg tolerance (BP-1 fix)
    atr3 = strat.get_atr("NVDA", length=14)
    assert atr3 > 0


def test_trade_tracker_double_close_guard():
    """Verify TradeTracker protects against duplicate trade close calls."""
    tracker = TradeTracker(ticker="TEST")
    trade_id = tracker.open_trade(
        date=datetime(2025, 1, 1),
        entry_price=100.0,
        quantity=10,
        take_profit=110.0,
        stop_loss=95.0,
        trade_type="BUY"
    )
    assert len(tracker.trades) == 1

    # First close
    t1 = tracker.close_trade(trade_id, date=datetime(2025, 1, 2), exit_price=105.0, exit_reason="TP")
    assert t1.pnl == 50.0
    assert tracker._total_pnl == 50.0

    # Second close should be guarded and not double-add PnL
    t2 = tracker.close_trade(trade_id, date=datetime(2025, 1, 2), exit_price=105.0, exit_reason="TP")
    assert tracker._total_pnl == 50.0


def test_trade_tracker_breakeven_pnl_dataframe():
    """Verify breakeven trades (pnl=0.0) are preserved in to_dataframe."""
    tracker = TradeTracker(ticker="TEST")
    trade_id = tracker.open_trade(
        date=datetime(2025, 1, 1),
        entry_price=100.0,
        quantity=10,
        take_profit=110.0,
        stop_loss=95.0,
        trade_type="BUY"
    )
    tracker.close_trade(trade_id, date=datetime(2025, 1, 2), exit_price=100.0, exit_reason="MANUAL")
    df = tracker.to_dataframe()
    assert len(df) == 1
    assert df["pnl"].iloc[0] == 0.0
    assert df["total_pnl"].iloc[0] == 0.0


def test_short_mean_reversion_entry_price():
    """Verify short mean reversion calculates sell limit 5% above close."""
    strat = ShortMeanReversionHigh6DSurge()
    strat.ticker_bars = pd.DataFrame({"close": [100.0]})
    entry_price = strat.get_entry_price()
    assert entry_price == 105.0
