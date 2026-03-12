"""
Demo Interactive Strategy

Run this to verify the InteractiveKeyLevelsStrategy.
"""

from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset
from interactive_strategy_base import InteractiveKeyLevelsStrategy

class DemoInteractive(InteractiveKeyLevelsStrategy):
    def get_entry_signal(self, current_price, support_levels, resistance_levels):
        return None
        
    def get_exit_signal(self, current_price, position, entry_support, target_resistance):
        return None

def run_demo():
    print("Starting Interactive Demo...")
    
    # Run backtest for a short period
    DemoInteractive.backtest(
        YahooDataBacktesting,
        datetime(2025, 12, 1),
        datetime(2025, 12, 5), # 5 days
        parameters={
            "Ticker": Asset(symbol="PLTR", asset_type=Asset.AssetType.STOCK),
            "RECALC_FREQUENCY": "daily"
        }
    )

if __name__ == "__main__":
    run_demo()
