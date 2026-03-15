"""
Tests for the signal_processor module.

Run with:
    python -m pytest tests/test_signal_processor.py -v
"""

import os
import sys
from datetime import date

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_processor import Signal, SignalLoader, SignalStore

# Path to real CSV
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cloud",
    "signals.csv",
)


# ------------------------------------------------------------------
# Helper to build a small in-memory store for deterministic tests
# ------------------------------------------------------------------

def _make_test_store() -> SignalStore:
    signals = [
        Signal(
            signal_id="aaa",
            ticker="AAPL",
            signal_date=date(2026, 1, 10),
            signal_position="BUY",
            news_keys=["111"],
            sentiment={"gemini_sentiment": 4, "news_sentiment": "POSITIVE", "fundamental_sentiment": "BUY"},
        ),
        Signal(
            signal_id="bbb",
            ticker="AAPL",
            signal_date=date(2026, 1, 15),
            signal_position="SELL",
            sentiment={"gemini_sentiment": 2, "news_sentiment": "NEGATIVE", "fundamental_sentiment": "SELL"},
        ),
        Signal(
            signal_id="ccc",
            ticker="NVDA",
            signal_date=date(2026, 1, 12),
            signal_position="BUY",
            calendar_event_keys=["cal1"],
            news_keys=["222", "333"],
            sentiment={"gemini_sentiment": 5, "fundamental_sentiment": "BUY"},
        ),
        Signal(
            signal_id="ddd",
            ticker="NVDA",
            signal_date=date(2026, 2, 1),
            signal_position="BUY",
            fundamental_analysis_key="fund_xyz",
        ),
        Signal(
            signal_id="eee",
            ticker="TSLA",
            signal_date=date(2026, 1, 20),
            signal_position="SELL",
            sentiment={"gemini_sentiment": 1, "fundamental_sentiment": "SELL", "news_sentiment": "NEGATIVE"},
        ),
    ]
    return SignalStore(signals)


# ------------------------------------------------------------------
# Unit tests (deterministic, no CSV needed)
# ------------------------------------------------------------------

class TestSignalModel:
    def test_properties(self):
        s = Signal(
            signal_id="test",
            ticker="X",
            signal_date=date(2026, 1, 1),
            signal_position="BUY",
            sentiment={"gemini_sentiment": 4, "news_sentiment": "POSITIVE", "fundamental_sentiment": "HOLD", "gemini_impact_type": "High"},
        )
        assert s.gemini_sentiment == 4
        assert s.news_sentiment == "POSITIVE"
        assert s.fundamental_sentiment == "HOLD"
        assert s.gemini_impact_type == "High"

    def test_no_sentiment(self):
        s = Signal(signal_id="t", ticker="X", signal_date=date(2026, 1, 1), signal_position="BUY")
        assert s.gemini_sentiment is None
        assert s.news_sentiment is None
        assert s.fundamental_sentiment is None

    def test_to_dict(self):
        s = Signal(signal_id="abc", ticker="AAPL", signal_date=date(2026, 1, 5), signal_position="BUY")
        d = s.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["signal_date"] == "2026-01-05"


class TestSignalStore:
    def test_by_ticker(self):
        store = _make_test_store()
        aapl = store.by_ticker("AAPL")
        assert aapl.count == 2
        assert all(s.ticker == "AAPL" for s in aapl)

    def test_by_ticker_case_insensitive(self):
        store = _make_test_store()
        assert store.by_ticker("aapl").count == 2

    def test_by_date(self):
        store = _make_test_store()
        result = store.by_date(date(2026, 1, 12))
        assert result.count == 1
        assert result[0].ticker == "NVDA"

    def test_by_date_range(self):
        store = _make_test_store()
        result = store.by_date_range(date(2026, 1, 10), date(2026, 1, 15))
        assert result.count == 3  # AAPL Jan 10, NVDA Jan 12, AAPL Jan 15

    def test_by_position(self):
        store = _make_test_store()
        buys = store.by_position("BUY")
        sells = store.by_position("SELL")
        assert buys.count == 3
        assert sells.count == 2

    def test_chaining(self):
        store = _make_test_store()
        result = store.by_ticker("NVDA").by_position("BUY")
        assert result.count == 2

    def test_by_sentiment_score(self):
        store = _make_test_store()
        high = store.by_sentiment_score(min_score=4, max_score=5)
        assert high.count == 2  # AAPL(4), NVDA(5)

    def test_by_fundamental_sentiment(self):
        store = _make_test_store()
        fund_buys = store.by_fundamental_sentiment("BUY")
        assert fund_buys.count == 2  # AAPL and NVDA

    def test_apply_filter(self):
        store = _make_test_store()
        has_news = store.apply_filter(lambda s: len(s.news_keys) > 0)
        assert has_news.count == 2  # AAPL(111), NVDA(222,333)

    def test_tickers(self):
        store = _make_test_store()
        assert store.tickers == ["AAPL", "NVDA", "TSLA"]

    def test_dates(self):
        store = _make_test_store()
        assert len(store.dates) == 5

    def test_group_by_ticker(self):
        store = _make_test_store()
        groups = store.group_by_ticker()
        assert set(groups.keys()) == {"AAPL", "NVDA", "TSLA"}
        assert groups["AAPL"].count == 2
        assert groups["NVDA"].count == 2
        assert groups["TSLA"].count == 1

    def test_group_by_date(self):
        store = _make_test_store()
        groups = store.group_by_date()
        assert len(groups) == 5

    def test_sorted_by_date(self):
        store = _make_test_store()
        ordered = store.sorted_by_date()
        dates = [s.signal_date for s in ordered]
        assert dates == sorted(dates)

    def test_sorted_by_ticker(self):
        store = _make_test_store()
        ordered = store.sorted_by_ticker()
        keys = [(s.ticker, s.signal_date) for s in ordered]
        assert keys == sorted(keys)

    def test_to_dataframe(self):
        store = _make_test_store()
        df = store.to_dataframe()
        assert len(df) == 5
        assert "ticker" in df.columns
        assert "signal_date" in df.columns

    def test_empty_store(self):
        empty = SignalStore()
        assert empty.count == 0
        assert empty.tickers == []
        assert empty.to_dataframe().empty

    def test_len_and_bool(self):
        store = _make_test_store()
        assert len(store) == 5
        assert bool(store) is True
        assert bool(SignalStore()) is False


# ------------------------------------------------------------------
# Integration test (requires real CSV)
# ------------------------------------------------------------------

class TestCSVLoader:
    def test_load_real_csv(self):
        if not os.path.isfile(CSV_PATH):
            print(f"SKIPPED: {CSV_PATH} not found")
            return
        store = SignalLoader.load_from_csv(CSV_PATH)
        assert store.count > 1000, f"Expected many signals, got {store.count}"
        print(f"\n  Loaded {store.count} signals, {len(store.tickers)} tickers")

    def test_real_csv_query(self):
        if not os.path.isfile(CSV_PATH):
            return
        store = SignalLoader.load_from_csv(CSV_PATH)

        # Test grouping
        groups = store.group_by_ticker()
        assert len(groups) > 100

        # Test filtering
        buys = store.by_position("BUY")
        sells = store.by_position("SELL")
        assert buys.count + sells.count <= store.count  # some may have other positions

        # Test chaining
        first_ticker = store.tickers[0]
        ticker_buys = store.by_ticker(first_ticker).by_position("BUY")
        assert all(s.ticker == first_ticker for s in ticker_buys)
        assert all(s.signal_position == "BUY" for s in ticker_buys)


# ------------------------------------------------------------------
# Run directly
# ------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
