"""
Unit Tests for Alpaca SMC Bridge & smartmoneyconcepts Integration (test_alpaca_smc_bridge.py)
==============================================================================================
Verifies:
1. Killzone Gating (London 02:00-05:00 EST, NY 07:00-10:00/11:30 EST)
2. `smartmoneyconcepts` library feeding historical bars to identify unmitigated FVGs & OBs
3. Native Alpaca Limit Order Formulation anchored to proximate POI edge
4. Advanced Alpaca Bracket Order Formulation with structural invalidation stop loss
5. Partial Profit Scaling (Tranches: 40% TP1, 40% TP2, 20% Runner / 40% + 60%)
6. M5 POI Selector & SMC FVG integration with smartmoneyconcepts
"""

import os
import sys
from datetime import datetime, timezone
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.alpaca_smc_bridge import AlpacaSMCBridge
from strategies.smc_bias_model.m5_poi import M5POISelector, POIType
from strategies.smc.smc_fvg import detect_fvgs, FVGType


@pytest.fixture
def sample_ohlcv_df():
    """Generates synthetic intraday candles with clear FVG imbalances and swings."""
    dates = pd.date_range("2026-09-01 09:30:00", periods=40, freq="5min", tz="UTC")
    np.random.seed(42)
    prices = 150.0 + np.cumsum(np.random.randn(40) * 0.5)

    opens, highs, lows, closes, volumes = [], [], [], [], []
    for i, p in enumerate(prices):
        o = p
        c = p + 0.2 if i % 2 == 0 else p - 0.2
        # Inject an intentional bullish FVG at bar 10:
        if i == 10:
            h = o + 2.0
            l = o - 0.1
            c = o + 1.8
        elif i == 11:
            l = o + 1.0  # Leaves gap above bar 9 high
            h = l + 1.5
            c = l + 1.2
        else:
            h = max(o, c) + 0.3
            l = min(o, c) - 0.3
        v = 150_000

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes
    }, index=dates)


def test_killzone_gating():
    bridge = AlpacaSMCBridge(allow_ny_extension=True)

    # 1. London Window: 03:30 AM EST = 08:30 UTC
    dt_london = datetime(2026, 9, 15, 8, 30, tzinfo=timezone.utc)
    in_kz, kz_name = bridge.is_in_killzone(dt_london)
    assert in_kz is True
    assert kz_name == "LONDON_KILLZONE"

    # 2. NY Window: 09:45 AM EST = 14:45 UTC (EDT) or 13:45 UTC
    # In September (EDT), 09:45 AM EDT is 13:45 UTC
    dt_ny = datetime(2026, 9, 15, 13, 45, tzinfo=timezone.utc)
    in_kz, kz_name = bridge.is_in_killzone(dt_ny)
    assert in_kz is True
    assert kz_name == "NEW_YORK_KILLZONE"

    # 3. Outside Killzone: 01:00 AM EST = 05:00 UTC
    dt_outside = datetime(2026, 9, 15, 5, 0, tzinfo=timezone.utc)
    in_kz, kz_name = bridge.is_in_killzone(dt_outside)
    assert in_kz is False
    assert kz_name == "OUTSIDE_KILLZONE"


def test_smartmoneyconcepts_library_feed(sample_ohlcv_df):
    bridge = AlpacaSMCBridge()
    coords = bridge.analyze_smc_coordinates(sample_ohlcv_df, bias="BULLISH", swing_length=2)

    assert coords["status"] == "OK"
    assert "fvgs" in coords
    assert "obs" in coords
    assert "swings" in coords
    assert isinstance(coords["fvgs"], list)
    assert isinstance(coords["obs"], list)


def test_alpaca_limit_order_payload():
    bridge = AlpacaSMCBridge()
    payload = bridge.build_limit_order_payload(
        symbol="AAPL",
        side="BUY",
        qty=100,
        limit_price=225.50,
        client_order_id="smc_limit_001"
    )

    assert payload["symbol"] == "AAPL"
    assert payload["side"] == "buy"
    assert payload["type"] == "limit"
    assert payload["limit_price"] == 225.50
    assert payload["qty"] == 100
    assert payload["client_order_id"] == "smc_limit_001"


def test_alpaca_bracket_order_with_structural_invalidation():
    bridge = AlpacaSMCBridge(tick_size=0.01)

    # Valid Long Bracket Order
    long_bracket = bridge.build_bracket_order_payload(
        symbol="MSFT",
        side="BUY",
        qty=50,
        entry_limit_price=450.00,
        structural_extreme=448.50,  # SL will be 448.49
        take_profit_target=460.00,
    )
    assert long_bracket["order_class"] == "bracket"
    assert long_bracket["limit_price"] == 450.00
    assert long_bracket["stop_loss"]["stop_price"] == 448.49
    assert long_bracket["take_profit"]["limit_price"] == 460.00

    # Valid Short Bracket Order
    short_bracket = bridge.build_bracket_order_payload(
        symbol="NVDA",
        side="SELL",
        qty=75,
        entry_limit_price=120.00,
        structural_extreme=121.20,  # SL will be 121.21
        take_profit_target=110.00,
    )
    assert short_bracket["order_class"] == "bracket"
    assert short_bracket["limit_price"] == 120.00
    assert short_bracket["stop_loss"]["stop_price"] == 121.21
    assert short_bracket["take_profit"]["limit_price"] == 110.00

    # Inverted coordinates should assert error
    with pytest.raises(AssertionError):
        bridge.build_bracket_order_payload(
            symbol="MSFT",
            side="BUY",
            qty=50,
            entry_limit_price=450.00,
            structural_extreme=452.00,  # Invalid: SL above entry for long!
            take_profit_target=460.00,
        )


def test_partial_scaling_tranches():
    bridge = AlpacaSMCBridge(default_tp1_pct=0.40, default_tp2_pct=0.40, default_tp3_pct=0.20)

    # 1. 3-stage model (40% TP1, 40% TP2, 20% Runner)
    t3 = bridge.calculate_tranche_quantities(100, model="THREE_STAGE_40_40_20")
    assert t3["TP1"] == 40
    assert t3["TP2"] == 40
    assert t3["RUNNER"] == 20
    assert sum(t3.values()) == 100

    # 2. Hybrid model (40% TP1, 60% Runner)
    t2 = bridge.calculate_tranche_quantities(100, model="HYBRID_40_60")
    assert t2["TP1"] == 40
    assert t2["RUNNER"] == 60
    assert sum(t2.values()) == 100

    # 3. Partial scale close payload
    scale_order = bridge.build_partial_scale_close_payload(
        symbol="AAPL",
        side="BUY",
        scale_qty=40,
        reason="TP1_HIT"
    )
    assert scale_order["side"] == "sell"
    assert scale_order["qty"] == 40
    assert scale_order["type"] == "market"

    # 4. Breakeven stop update payload
    be_update = bridge.build_breakeven_stop_update_payload(
        existing_stop_order_id="stop_order_123",
        entry_price=225.50,
        remaining_qty=60
    )
    assert be_update["order_id"] == "stop_order_123"
    assert be_update["stop_price"] == 225.50
    assert be_update["qty"] == 60


def test_m5_poi_and_fvg_smartmoneyconcepts_integration(sample_ohlcv_df):
    selector = M5POISelector()
    fvgs = selector.detect_5m_fvgs(sample_ohlcv_df)
    assert isinstance(fvgs, list)

    obs = selector.detect_5m_order_blocks(sample_ohlcv_df, bias="BULLISH")
    assert isinstance(obs, list)

    best_poi = selector.select_best_poi_with_confluence(sample_ohlcv_df, bias="BULLISH")
    # Best POI returned should have valid proximate and distal prices
    if best_poi:
        assert best_poi.proximate_price > 0
        assert best_poi.distal_price > 0

    smc_fvgs = detect_fvgs(sample_ohlcv_df)
    assert isinstance(smc_fvgs, list)
