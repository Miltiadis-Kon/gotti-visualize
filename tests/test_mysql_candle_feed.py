import unittest
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plots.mysql_candle_feed import MySQLCandleFeed


class TestMySQLCandleFeed(unittest.TestCase):
    def setUp(self):
        self.feed = MySQLCandleFeed()

    def test_get_ticker_coverage(self):
        cov = self.feed.get_ticker_coverage("AAPL", timeframe="5 min")
        self.assertEqual(cov["ticker"], "AAPL")
        self.assertGreater(cov["count"], 1000)
        self.assertIsNotNone(cov["start_date"])
        self.assertIsNotNone(cov["end_date"])

    def test_get_candles_df(self):
        df = self.feed.get_candles_df("AAPL", timeframe="5 min", limit=100)
        self.assertEqual(len(df), 100)
        self.assertListEqual(list(df.columns), ["open", "high", "low", "close", "volume", "vwap"])
        # Check index is DatetimeIndex
        self.assertTrue(hasattr(df.index, "tz"))
        # Check first row has positive prices
        first = df.iloc[0]
        self.assertGreater(first["open"], 0)
        self.assertGreater(first["close"], 0)
        self.assertGreaterEqual(first["high"], first["low"])


if __name__ == "__main__":
    unittest.main()
