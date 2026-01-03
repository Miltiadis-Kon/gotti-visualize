"""
Multi-Timeframe Key Levels Strategy

A more sophisticated strategy that:
- Uses multiple timeframes (1D, 4H, 1H, 15m, 5m)
- Requires minimum importance level for entry
- Uses configurable Risk:Reward ratio
- Take Profit set near resistance with threshold
- Stop Loss set below support with threshold

This is the refactored version of KeyLevelsStrategy using BaseKeyLevelsStrategy.
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_key_levels_strategy import BaseKeyLevelsStrategy

load_dotenv()


class MultiTimeframeKeyLevelsStrategy(BaseKeyLevelsStrategy):
    """
    Multi-Timeframe Key Levels Trading Strategy.
    
    Uses support/resistance levels from multiple timeframes with importance scoring.
    Entry when price touches a strong support with a valid resistance target above.
    
    Key features:
    - Multi-timeframe analysis (1D, 4H, 1H, 15m, 5m)
    - Importance-weighted levels
    - Configurable R:R ratio
    - Dynamic levels refresh
    """
    
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "RISK_PERCENT": 0.02,          # Risk 2% per trade
        "MIN_IMPORTANCE": 3,           # Minimum importance for levels (3 = 1H or higher)
        "TIMEFRAMES": ['1d', '4h', '1h', '15m', '5m'],  # All timeframes
        "PRICE_THRESHOLD": 0.5,        # Threshold for merging levels
        "RECALC_FREQUENCY": "daily",   # Recalculate levels daily
        
        # Entry/Exit thresholds (from base class)
        "ENTRY_THRESHOLD": 0.005,      # 0.5% tolerance for level matching
        "EXIT_THRESHOLD": 0.01,        # 1% tolerance for TP/SL matching
        
        # Strategy-specific parameters
        "TP_THRESHOLD": 0.02,          # Take profit 2% before resistance
        "SL_THRESHOLD": 0.05,          # Stop loss 5% below support
        "MIN_RISK_REWARD": 1.5,        # Minimum R:R ratio
    }
    
    def get_strategy_name(self) -> str:
        return "MultiTFKeyLevels"
    
    def on_strategy_start(self):
        """Load strategy-specific parameters."""
        self.tp_threshold = self.parameters.get("TP_THRESHOLD", 0.02)
        self.sl_threshold = self.parameters.get("SL_THRESHOLD", 0.05)
        self.min_risk_reward = self.parameters.get("MIN_RISK_REWARD", 1.5)
        
        self.log_message(
            f"[{self.get_strategy_name()}] Initialized (LONG+SHORT) - "
            f"Min R:R={self.min_risk_reward}, Entry threshold={self.entry_threshold*100}%"
        )
    
    def get_entry_signal(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Check for LONG entry at support OR SHORT entry at resistance.
        
        LONG Entry (BUY):
        - Price at support level
        - TP at resistance above
        - SL below support
        
        SHORT Entry (SELL):
        - Price at resistance level
        - TP at support below
        - SL above resistance
        """
        if support_levels.empty or resistance_levels.empty:
            return None
        
        # Try LONG entry first
        long_signal = self._check_long_entry(current_price, support_levels, resistance_levels)
        if long_signal:
            return long_signal
        
        # Try SHORT entry
        short_signal = self._check_short_entry(current_price, support_levels, resistance_levels)
        if short_signal:
            return short_signal
        
        return None
    
    def _check_long_entry(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Check for LONG entry at support level."""
        supports = support_levels.copy()
        supports['distance'] = abs(supports['level_price'] - current_price)
        supports = supports.sort_values(['importance', 'distance'], ascending=[False, True])
        
        resistances = resistance_levels.sort_values('level_price', ascending=True)
        
        for _, support in supports.iterrows():
            support_price = support['level_price']
            support_importance = support['importance']
            
            # Check if price is at this support
            tolerance = support_price * self.entry_threshold
            if not (support_price - tolerance <= current_price <= support_price + tolerance):
                continue
            
            entry_price = current_price
            stop_loss = self._calculate_long_sl(support_price)
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share <= 0:
                continue
            
            # Find resistance that meets R:R
            resistances_above = resistances[resistances['level_price'] > entry_price]
            
            for _, resistance in resistances_above.iterrows():
                target_resistance = resistance['level_price']
                take_profit = self._calculate_long_tp(target_resistance)
                reward_per_share = take_profit - entry_price
                
                risk_reward = reward_per_share / risk_per_share
                
                if risk_reward >= self.min_risk_reward:
                    self.log_message(
                        f"[{self.get_strategy_name()}] LONG signal: "
                        f"Support ${support_price:.2f} (imp={support_importance}), "
                        f"Resistance ${target_resistance:.2f}, R:R={risk_reward:.2f}"
                    )
                    
                    return {
                        'trade_type': 'BUY',
                        'entry_price': entry_price,
                        'take_profit': round(take_profit, 2),
                        'stop_loss': round(stop_loss, 2),
                        'support_level': support_price,
                        'resistance_level': target_resistance,
                    }
            
            # Fallback with fixed R:R
            if not resistances_above.empty:
                fallback_reward = risk_per_share * self.min_risk_reward
                fallback_tp = entry_price + fallback_reward
                nearest_resistance = resistances_above.iloc[0]['level_price']
                
                return {
                    'trade_type': 'BUY',
                    'entry_price': entry_price,
                    'take_profit': round(fallback_tp, 2),
                    'stop_loss': round(stop_loss, 2),
                    'support_level': support_price,
                    'resistance_level': nearest_resistance,
                }
        
        return None
    
    def _check_short_entry(
        self,
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Check for SHORT entry at resistance level."""
        resistances = resistance_levels.copy()
        resistances['distance'] = abs(resistances['level_price'] - current_price)
        resistances = resistances.sort_values(['importance', 'distance'], ascending=[False, True])
        
        supports = support_levels.sort_values('level_price', ascending=False)
        
        for _, resistance in resistances.iterrows():
            resistance_price = resistance['level_price']
            resistance_importance = resistance['importance']
            
            # Check if price is at this resistance
            tolerance = resistance_price * self.entry_threshold
            if not (resistance_price - tolerance <= current_price <= resistance_price + tolerance):
                continue
            
            entry_price = current_price
            stop_loss = self._calculate_short_sl(resistance_price)
            risk_per_share = stop_loss - entry_price  # For shorts, SL is above entry
            
            if risk_per_share <= 0:
                continue
            
            # Find support that meets R:R
            supports_below = supports[supports['level_price'] < entry_price]
            
            for _, support in supports_below.iterrows():
                target_support = support['level_price']
                take_profit = self._calculate_short_tp(target_support)
                reward_per_share = entry_price - take_profit  # For shorts, profit when price goes down
                
                risk_reward = reward_per_share / risk_per_share
                
                if risk_reward >= self.min_risk_reward:
                    self.log_message(
                        f"[{self.get_strategy_name()}] SHORT signal: "
                        f"Resistance ${resistance_price:.2f} (imp={resistance_importance}), "
                        f"Support ${target_support:.2f}, R:R={risk_reward:.2f}"
                    )
                    
                    return {
                        'trade_type': 'SELL',
                        'entry_price': entry_price,
                        'take_profit': round(take_profit, 2),
                        'stop_loss': round(stop_loss, 2),
                        'support_level': target_support,
                        'resistance_level': resistance_price,
                    }
            
            # Fallback with fixed R:R
            if not supports_below.empty:
                fallback_reward = risk_per_share * self.min_risk_reward
                fallback_tp = entry_price - fallback_reward
                nearest_support = supports_below.iloc[0]['level_price']
                
                return {
                    'trade_type': 'SELL',
                    'entry_price': entry_price,
                    'take_profit': round(fallback_tp, 2),
                    'stop_loss': round(stop_loss, 2),
                    'support_level': nearest_support,
                    'resistance_level': resistance_price,
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
        Manual exit check (backup for bracket orders).
        Uses exit_threshold for tolerance matching.
        """
        trade = self.trade_tracker.get_trade(self.current_trade_id)
        if trade is None:
            return None
        
        # Calculate tolerance zones
        tp_tolerance = trade.take_profit * self.exit_threshold
        sl_tolerance = trade.stop_loss * self.exit_threshold
        
        if trade.trade_type == "BUY":
            # LONG position - TP is above, SL is below
            if current_price >= trade.take_profit - tp_tolerance:
                return "TP"
            if current_price <= trade.stop_loss + sl_tolerance:
                return "SL"
        else:
            # SHORT position - TP is below, SL is above
            if current_price <= trade.take_profit + tp_tolerance:
                return "TP"
            if current_price >= trade.stop_loss - sl_tolerance:
                return "SL"
        
        return None
    
    def _calculate_long_tp(self, resistance_price: float) -> float:
        """Calculate LONG TP price (threshold% before resistance)."""
        return resistance_price * (1 - self.tp_threshold)
    
    def _calculate_long_sl(self, support_price: float) -> float:
        """Calculate LONG SL price (threshold% below support)."""
        return support_price * (1 - self.sl_threshold)
    
    def _calculate_short_tp(self, support_price: float) -> float:
        """Calculate SHORT TP price (threshold% above support)."""
        return support_price * (1 + self.tp_threshold)
    
    def _calculate_short_sl(self, resistance_price: float) -> float:
        """Calculate SHORT SL price (threshold% above resistance)."""
        return resistance_price * (1 + self.sl_threshold)


def run_backtest(
    ticker: str = "NVDA",
    start_date: datetime = None,
    end_date: datetime = None,
    budget: float = 10000,
    min_importance: int = 3,
    min_risk_reward: float = 1.5
):
    """Run backtest of Multi-Timeframe Key Levels Strategy."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Running Multi-TF Key Levels backtest for {ticker}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    MultiTimeframeKeyLevelsStrategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        budget=budget,
        parameters={
            "Ticker": Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK),
            "MIN_IMPORTANCE": min_importance,
            "MIN_RISK_REWARD": min_risk_reward,
        }
    )


if __name__ == "__main__":
    run_backtest(
        ticker="PLTR",
        start_date=datetime(2025, 12, 1),
        end_date=datetime(2025, 12, 24),
        budget=10000,
        min_importance=2,
        min_risk_reward=1.5
    )
