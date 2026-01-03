"""
Simple 5-Minute Key Levels Strategy

A simple example strategy that:
- Uses only 5-minute timeframe support/resistance levels
- Refreshes levels daily (at start of each trading day)
- Places 1:1 Risk:Reward trades
- Entry: When price touches a support level
- Take Profit: At the nearest resistance level
- Stop Loss: Equal distance below support (1:1 R:R)

This demonstrates how to extend BaseKeyLevelsStrategy with minimal code.
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_key_levels_strategy import BaseKeyLevelsStrategy


class Simple5mStrategy(BaseKeyLevelsStrategy):
    """
    Simple 5-minute key levels strategy with 1:1 Risk:Reward.
    
    Entry: Price at or near a 5m support level
    Take Profit: Nearest resistance level above entry
    Stop Loss: Same distance below entry as TP is above (1:1 R:R)
    """
    
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RISK_PERCENT": 0.02,          # Risk 2% per trade
        "MIN_IMPORTANCE": 1,           # 5m levels have importance=1
        "TIMEFRAMES": ['5m'],          # Only use 5-minute timeframe
        "PRICE_THRESHOLD": 0.3,        # Tighter threshold for 5m levels
        "RECALC_FREQUENCY": "daily",   # Refresh levels daily
        "SUPPORT_TOLERANCE": 0.003,    # 0.3% tolerance for support touch
    }
    
    def get_strategy_name(self) -> str:
        return "Simple5mStrategy"
    
    def on_strategy_start(self):
        """Custom initialization."""
        self.support_tolerance = self.parameters.get("SUPPORT_TOLERANCE", 0.003)
        self.log_message(f"[{self.get_strategy_name()}] Initialized with 1:1 R:R on 5m levels")
    
    def get_entry_signal(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Check if price is at a support level and find 1:1 R:R setup.
        
        Entry logic:
        1. Price must be within tolerance of a support level
        2. Must have a resistance level above
        3. Calculate TP at resistance, SL at equal distance below
        """
        if support_levels.empty or resistance_levels.empty:
            return None
        
        # Sort levels
        supports = support_levels.sort_values('level_price', ascending=False)
        resistances = resistance_levels.sort_values('level_price', ascending=True)
        
        # Find supports near current price
        for _, support in supports.iterrows():
            support_price = support['level_price']
            
            # Check if price is at this support (within tolerance)
            tolerance = support_price * self.support_tolerance
            if not (support_price - tolerance <= current_price <= support_price + tolerance):
                continue
            
            # Find nearest resistance above
            resistances_above = resistances[resistances['level_price'] > current_price]
            if resistances_above.empty:
                continue
            
            nearest_resistance = resistances_above.iloc[0]['level_price']
            
            # Calculate 1:1 R:R
            entry_price = current_price
            reward_distance = nearest_resistance - entry_price
            risk_distance = reward_distance  # 1:1 R:R
            
            take_profit = nearest_resistance
            stop_loss = entry_price - risk_distance
            
            # Sanity check
            if stop_loss <= 0 or take_profit <= entry_price:
                continue
            
            self.log_message(
                f"[{self.get_strategy_name()}] Entry signal found at support ${support_price:.2f}"
            )
            
            return {
                'entry_price': entry_price,
                'take_profit': round(take_profit, 2),
                'stop_loss': round(stop_loss, 2),
                'support_level': support_price,
                'resistance_level': nearest_resistance,
            }
        
        return None
    
    def get_exit_signal(
        self,
        current_price: float,
        position,
        entry_support: float,
        target_resistance: float
    ) -> Optional[str]:
        """
        Check for manual exit (backup in case bracket order didn't trigger).
        """
        trade = self.trade_tracker.get_trade(self.current_trade_id)
        if trade is None:
            return None
        
        # Manual TP check
        if current_price >= trade.take_profit:
            return "TP"
        
        # Manual SL check
        if current_price <= trade.stop_loss:
            return "SL"
        
        return None


def run_backtest(
    ticker: str = "NVDA",
    start_date: datetime = None,
    end_date: datetime = None,
    budget: float = 10000
):
    """Run backtest of Simple 5m Strategy."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Running Simple 5m Strategy backtest for {ticker}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    Simple5mStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        budget=budget,
        parameters={
            "Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK),
        }
    )


if __name__ == "__main__":
    run_backtest(
        ticker="PLTR",
        start_date=datetime(2025, 11, 1),
        end_date=datetime(2025, 12, 20),
        budget=10000
    )
