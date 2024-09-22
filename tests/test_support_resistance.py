import unittest
import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from indicators.support_resistance_window import get_ticker, support_resistance

class TestSupportResistance(unittest.TestCase):

    def test_get_ticker(self):
        # Test fetching data for a known stock symbol
        data = get_ticker('AAPL', '1mo', '1d')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('Open', data.columns)
        self.assertIn('High', data.columns)
        self.assertIn('Low', data.columns)
        self.assertIn('Close', data.columns)
        self.assertIn('Volume', data.columns)

    def test_support_resistance(self):
        # Create a sample DataFrame
        data = pd.DataFrame({
            'Low': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        })

        # Calculate support and resistance
        support, resistance = support_resistance(data, window=3)

        # Check if the support and resistance are calculated correctly
        expected_support = pd.Series([None, None, 100, 101, 102, 103, 104, 105, 106, 107])
        expected_resistance = pd.Series([None, None, 112, 113, 114, 115, 116, 117, 118, 119])

        pd.testing.assert_series_equal(support.reset_index(drop=True), expected_support, check_names=False)
        pd.testing.assert_series_equal(resistance.reset_index(drop=True), expected_resistance, check_names=False)

if __name__ == '__main__':
    unittest.main()