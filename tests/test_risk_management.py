"""
Unit tests for ATR-based volatility optimization of Take Profit and Stop Loss,
and StrategyBaseplate integration.
"""

import pytest
import pandas as pd
import numpy as np
from strategies.risk_management import (
    calculate_atr,
    optimize_sl_tp,
    calculate_trailing_stop,
    ATRRiskLevels,
    TRADING_STYLES,
)
from strategies.strat_baseplate import StrategyBaseplate, OpeningGap


# ---------------------------------------------------------------------------
# Synthetic Data Helpers
# ---------------------------------------------------------------------------

def make_ohlcv_df(n: int = 50, base_price: float = 100.0, volatility: float = 2.0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing ATR calculation."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    
    close = [base_price]
    for _ in range(1, n):
        close.append(close[-1] + np.random.uniform(-volatility, volatility))
        
    close = np.array(close)
    high = close + np.random.uniform(0.5, volatility, n)
    low = close - np.random.uniform(0.5, volatility, n)
    open_p = low + np.random.uniform(0.1, 0.9, n) * (high - low)
    volume = np.random.randint(100_000, 1_000_000, n)
    
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


# ---------------------------------------------------------------------------
# Test Trading Styles Configuration
# ---------------------------------------------------------------------------

def test_trading_styles_spec():
    """Verify all 4 trading styles are defined with correct multipliers and RR ratios."""
    assert "scalping" in TRADING_STYLES
    assert "day_trading" in TRADING_STYLES
    assert "swing_trading" in TRADING_STYLES
    assert "trend_following" in TRADING_STYLES

    # Scalping: 0.5 - 1.0 ATR SL, 1:1 - 1:1.5 RR
    scalp = TRADING_STYLES["scalping"]
    assert scalp["sl_range"] == (0.5, 1.0)
    assert scalp["sl_multiplier"] == 1.0
    assert scalp["rr_range"] == (1.0, 1.5)
    assert scalp["risk_reward"] == 1.5

    # Day Trading: 1.0 - 1.5 ATR SL, 1:1.5 - 1:2 RR
    day = TRADING_STYLES["day_trading"]
    assert day["sl_range"] == (1.0, 1.5)
    assert day["sl_multiplier"] == 1.5
    assert day["rr_range"] == (1.5, 2.0)
    assert day["risk_reward"] == 2.0

    # Swing Trading: 2.0 - 3.0 ATR SL, 1:2 - 1:3 RR
    swing = TRADING_STYLES["swing_trading"]
    assert swing["sl_range"] == (2.0, 3.0)
    assert swing["sl_multiplier"] == 2.0
    assert swing["rr_range"] == (2.0, 3.0)
    assert swing["risk_reward"] == 2.0

    # Trend Following: 3.0 - 5.0 ATR SL, Trailing stop
    trend = TRADING_STYLES["trend_following"]
    assert trend["sl_range"] == (3.0, 5.0)
    assert trend["sl_multiplier"] == 3.0
    assert trend["trailing_stop"] is True


# ---------------------------------------------------------------------------
# Test User Prompt Specific Example Scenario
# ---------------------------------------------------------------------------

def test_user_example_scenario():
    """
    Test exact scenario from prompt:
    - Asset: ABC Stock
    - Current Price (Entry): $150.00
    - ATR (14-day): $2.50
    - Trading Style: Swing Trading (2x ATR SL, 1:2 Risk/Reward)
    - Expected SL: $145.00
    - Expected TP: $160.00
    - Trailing Stop: Stock rallies to $160.00 -> Stop moves to $155.00
    """
    entry_price = 150.00
    atr = 2.50
    
    levels = optimize_sl_tp(
        entry_price=entry_price,
        side="buy",
        atr=atr,
        trading_style="swing_trading",
        sl_multiplier=2.0,
        risk_reward=2.0,
    )
    
    assert levels.entry_price == 150.00
    assert levels.atr == 2.50
    assert pytest.approx(levels.risk_distance, abs=0.01) == 5.00
    assert pytest.approx(levels.stop_loss, abs=0.01) == 145.00
    assert pytest.approx(levels.reward_distance, abs=0.01) == 10.00
    assert pytest.approx(levels.take_profit, abs=0.01) == 160.00

    # Trailing Stop ratcheting
    # When price hits $160.00, trailing stop should be 160 - (2.0 * 2.50) = $155.00
    trailing_stop = calculate_trailing_stop(
        highest_price=160.00,
        atr=2.50,
        sl_multiplier=2.0,
        side="buy",
    )
    assert pytest.approx(trailing_stop, abs=0.01) == 155.00


# ---------------------------------------------------------------------------
# Test Long and Short Calculations
# ---------------------------------------------------------------------------

def test_long_and_short_calculations():
    """Verify formulas for long vs short trades."""
    # Long trade
    long_res = optimize_sl_tp(
        entry_price=100.0,
        side="buy",
        atr=2.0,
        trading_style="day_trading",
        sl_multiplier=1.5,
        risk_reward=2.0,
    )
    # SL = 100 - (2.0 * 1.5) = 97.0
    # TP = 100 + (3.0 * 2.0) = 106.0
    assert pytest.approx(long_res.stop_loss, abs=0.01) == 97.0
    assert pytest.approx(long_res.take_profit, abs=0.01) == 106.0

    # Short trade
    short_res = optimize_sl_tp(
        entry_price=100.0,
        side="sell",
        atr=2.0,
        trading_style="day_trading",
        sl_multiplier=1.5,
        risk_reward=2.0,
    )
    # SL = 100 + (2.0 * 1.5) = 103.0
    # TP = 100 - (3.0 * 2.0) = 94.0
    assert pytest.approx(short_res.stop_loss, abs=0.01) == 103.0
    assert pytest.approx(short_res.take_profit, abs=0.01) == 94.0

    # Tuple unpacking support
    sl, tp = long_res
    assert sl == long_res.stop_loss
    assert tp == long_res.take_profit


# ---------------------------------------------------------------------------
# Test Custom SL and TP Volatility Optimization
# ---------------------------------------------------------------------------

def test_custom_sl_noise_zone_expansion():
    """
    When custom SL is placed inside the market noise zone (< min_sl * ATR),
    it must be expanded to give the trade breathing room and prevent premature stop-out.
    """
    entry = 100.0
    atr = 2.0
    # Day trading range: 1.0 - 1.5 ATR. Min SL distance = 1.0 * 2.0 = 2.0
    # Custom SL at 99.0 -> distance is only 1.0 (inside noise zone)
    custom_sl = 99.0
    
    res = optimize_sl_tp(
        entry_price=entry,
        stop_loss=custom_sl,
        side="buy",
        atr=atr,
        trading_style="day_trading",
    )
    
    # Distance should be expanded to at least min_mult * atr = 1.0 * 2.0 = 2.0 -> SL = 98.0
    assert res.stop_loss <= 98.0
    assert res.risk_distance >= 2.0


def test_custom_sl_over_risk_contraction():
    """
    When custom SL is excessively loose (> max_sl * ATR),
    it must be contracted to the upper volatility limit to protect capital.
    """
    entry = 100.0
    atr = 2.0
    # Day trading range: 1.0 - 1.5 ATR. Max SL distance = 1.5 * 2.0 = 3.0
    # Custom SL at 90.0 -> distance is 10.0 (excessive risk)
    custom_sl = 90.0
    
    res = optimize_sl_tp(
        entry_price=entry,
        stop_loss=custom_sl,
        side="buy",
        atr=atr,
        trading_style="day_trading",
    )
    
    # Distance should be capped at max_mult * atr = 1.5 * 2.0 = 3.0 -> SL = 97.0
    assert pytest.approx(res.stop_loss, abs=0.01) == 97.0
    assert pytest.approx(res.risk_distance, abs=0.01) == 3.0


def test_custom_tp_under_reward_expansion():
    """
    When custom TP does not provide enough reward relative to the actual risk (RR),
    it must be adjusted to meet the target Risk-to-Reward ratio.
    """
    entry = 100.0
    atr = 2.0
    # Swing trading: default SL = 2.0 * 2.0 = 4.0 -> SL = 96.0
    # Target RR = 2.0 -> Reward should be at least 8.0 -> TP >= 108.0
    # Custom TP at 102.0 only gives 2.0 reward (RR 0.5)
    custom_tp = 102.0
    
    res = optimize_sl_tp(
        entry_price=entry,
        take_profit=custom_tp,
        side="buy",
        atr=atr,
        trading_style="swing_trading",
        risk_reward=2.0,
    )
    
    assert res.take_profit >= 108.0
    assert res.reward_distance >= 8.0


# ---------------------------------------------------------------------------
# Test ATR Calculation
# ---------------------------------------------------------------------------

def test_calculate_atr():
    """Verify Wilder's ATR calculation on OHLCV data."""
    df = make_ohlcv_df(n=50, base_price=100.0, volatility=3.0)
    atr = calculate_atr(df, length=14)
    
    assert atr is not None
    assert atr > 0
    assert isinstance(atr, float)

    # Empty or insufficient dataframe returns 0.0 fallback
    empty_df = pd.DataFrame()
    assert calculate_atr(empty_df) == 0.0


# ---------------------------------------------------------------------------
# Test Trailing Stop Ratchet
# ---------------------------------------------------------------------------

def test_trailing_stop_ratchet():
    """Test that trailing stops ratchet forward and never backwards."""
    atr = 2.0
    mult = 2.0
    # Long: highest price moves 100 -> 105 -> 103 -> 110
    stop1 = calculate_trailing_stop(100.0, atr, mult, side="buy")  # 96.0
    stop2 = calculate_trailing_stop(105.0, atr, mult, side="buy")  # 101.0
    stop3 = calculate_trailing_stop(103.0, atr, mult, side="buy", current_stop=stop2)  # Should remain 101.0
    stop4 = calculate_trailing_stop(110.0, atr, mult, side="buy", current_stop=stop3)  # 106.0

    assert pytest.approx(stop1, abs=0.01) == 96.0
    assert pytest.approx(stop2, abs=0.01) == 101.0
    assert pytest.approx(stop3, abs=0.01) == 101.0
    assert pytest.approx(stop4, abs=0.01) == 106.0

    # Short: lowest price moves 100 -> 95 -> 97 -> 90
    s_stop1 = calculate_trailing_stop(100.0, atr, mult, side="sell")  # 104.0
    s_stop2 = calculate_trailing_stop(95.0, atr, mult, side="sell")   # 99.0
    s_stop3 = calculate_trailing_stop(97.0, atr, mult, side="sell", current_stop=s_stop2)  # Should remain 99.0
    s_stop4 = calculate_trailing_stop(90.0, atr, mult, side="sell", current_stop=s_stop3)  # 94.0

    assert pytest.approx(s_stop1, abs=0.01) == 104.0
    assert pytest.approx(s_stop2, abs=0.01) == 99.0
    assert pytest.approx(s_stop3, abs=0.01) == 99.0
    assert pytest.approx(s_stop4, abs=0.01) == 94.0


# ---------------------------------------------------------------------------
# Test StrategyBaseplate and Backward Compatibility
# ---------------------------------------------------------------------------

def test_strategy_baseplate_interface():
    """Verify StrategyBaseplate exposes ATR methods and OpeningGap alias."""
    assert OpeningGap is StrategyBaseplate
    assert hasattr(StrategyBaseplate, "get_atr")
    assert hasattr(StrategyBaseplate, "optimize_sl_tp")
    assert hasattr(StrategyBaseplate, "get_trailing_stop")


# ---------------------------------------------------------------------------
# Test ALL Strategies Inherit from StrategyBaseplate
# ---------------------------------------------------------------------------

def test_all_strategies_inherit_from_baseplate():
    """Verify 100% of active strategies across the repository inherit from StrategyBaseplate."""
    # 1. Book Based
    from strategies.book_based.opening_gap import OpeningGap as BG_OpeningGap
    from strategies.book_based.long_mean_reversion_selloff import LongMeanReversionSelloff
    from strategies.book_based.long_mean_reversion_high_ADX_reversal import LongMeanReversionHighADXReversal
    from strategies.book_based.long_trend_low_volatility import LongTrendLowVolatility
    from strategies.book_based.the_catastrophe_hedge import TheCatastropheHedge
    from strategies.book_based.short_mean_reversion_high_6D_surge import ShortMeanReversionHigh6DSurge
    from strategies.book_based.long_trend_high_momentum import LongTrendHighMomentum
    from strategies.book_based.short_rsi_thrust import ShortRSIThrust
    from strategies.book_based.signal_key_levels_strategy import SignalKeyLevelsStrategy

    book_based_classes = [
        BG_OpeningGap,
        LongMeanReversionSelloff,
        LongMeanReversionHighADXReversal,
        LongTrendLowVolatility,
        TheCatastropheHedge,
        ShortMeanReversionHigh6DSurge,
        LongTrendHighMomentum,
        ShortRSIThrust,
        SignalKeyLevelsStrategy,
    ]
    for cls in book_based_classes:
        assert issubclass(cls, StrategyBaseplate), f"{cls.__name__} in book_based must subclass StrategyBaseplate"

    # 2. Key Levels Strategies
    from strategies.base_key_levels_strategy import BaseKeyLevelsStrategy
    from strategies.multi_tf_strategy import MultiTimeframeKeyLevelsStrategy
    assert issubclass(BaseKeyLevelsStrategy, StrategyBaseplate)
    assert issubclass(MultiTimeframeKeyLevelsStrategy, StrategyBaseplate)

    # 3. Swing Strategies
    from strategies.swing.gap_setup_long import GapSetupLongSwing
    from strategies.swing.gap_setup_short import GapSetupShortSwing
    from strategies.swing.hcr_breakout_long import HCRBreakoutLongSwing
    from strategies.swing.hcr_breakout_short import HCRBreakoutShortSwing
    from strategies.swing.retrace_long import RetraceLongSwing
    from strategies.swing.retrace_short import RetraceShortSwing
    from strategies.swing.rsi_setup import RSISetupSwing
    from strategies.swing.fib_retrace_swing import FibRetraceSwing as FibSwing

    swing_classes = [
        GapSetupLongSwing,
        GapSetupShortSwing,
        HCRBreakoutLongSwing,
        HCRBreakoutShortSwing,
        RetraceLongSwing,
        RetraceShortSwing,
        RSISetupSwing,
        FibSwing,
    ]
    for cls in swing_classes:
        assert issubclass(cls, StrategyBaseplate), f"{cls.__name__} in swing must subclass StrategyBaseplate"

    # 4. Book Based 2 Strategies
    from strategies.book_based_2.gap_setup_long import OpeningGapLong as BB2_GapLong
    from strategies.book_based_2.gap_setup_short import OpeningGapShort as BB2_GapShort
    from strategies.book_based_2.hcr_breakout_long import HCRBreakoutLong as BB2_HCRLong
    from strategies.book_based_2.hcr_breakout_short import HCRBreakoutShort as BB2_HCRShort
    from strategies.book_based_2.retrace_long import RetraceLong as BB2_RetraceLong
    from strategies.book_based_2.retrace_short import RetraceShort as BB2_RetraceShort
    from strategies.book_based_2.rsi_setup import RSISetup as BB2_RSI
    from strategies.book_based_2.fib_retrace_swing import FibRetraceSwing as BB2_Fib

    bb2_classes = [
        BB2_GapLong,
        BB2_GapShort,
        BB2_HCRLong,
        BB2_HCRShort,
        BB2_RetraceLong,
        BB2_RetraceShort,
        BB2_RSI,
        BB2_Fib,
    ]
    for cls in bb2_classes:
        assert issubclass(cls, StrategyBaseplate), f"{cls.__name__} in book_based_2 must subclass StrategyBaseplate"


# ---------------------------------------------------------------------------
# Test API Endpoints
# ---------------------------------------------------------------------------

def test_api_risk_calculator_endpoints():
    """Verify FastAPI endpoints /api/risk-calculator/styles and /api/risk-calculator."""
    from starlette.testclient import TestClient
    from api import app

    client = TestClient(app)

    # 1. Styles endpoint
    styles_resp = client.get("/api/risk-calculator/styles")
    assert styles_resp.status_code == 200
    styles_data = styles_resp.json()
    assert "styles" in styles_data
    assert "scalping" in styles_data["styles"]
    assert "day_trading" in styles_data["styles"]
    assert "swing_trading" in styles_data["styles"]
    assert "trend_following" in styles_data["styles"]

    # 2. Risk calculator endpoint with user reference scenario
    calc_resp = client.post("/api/risk-calculator", json={
        "entry_price": 150.0,
        "atr": 2.5,
        "trading_style": "swing_trading",
        "current_price": 160.0,
    })
    assert calc_resp.status_code == 200
    calc_data = calc_resp.json()
    assert calc_data["entry_price"] == 150.0
    assert calc_data["stop_loss"] == 145.0
    assert calc_data["take_profit"] == 160.0
    assert calc_data["risk_distance"] == 5.0
    assert calc_data["reward_distance"] == 10.0
    assert calc_data["risk_reward_ratio"] == 2.0
    assert calc_data["current_trailing_stop"] == 155.0

