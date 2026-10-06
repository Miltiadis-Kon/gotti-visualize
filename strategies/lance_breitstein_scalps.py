import os
import sys
import math
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_baseplate import StrategyBaseplate

class LanceBreitsteinScalping(StrategyBaseplate):
    """
    Lance Breitstein's 3 Scalping Strategies:
    1. Continuation (Breakout) Scalp
    2. Mean Reversion (Exhaustion) Scalp
    3. Making the Spread (Liquidity Provision)
    
    NOTE: True order flow scalping requires Level 2 (order book) and Time & Sales (tape) data.
    This implementation uses price-action and volume proxies suitable for standard OHLCV 
    backtesting where tick-level Level 2 data isn't readily available.
    """
    
    parameters = {
        "Ticker": "NVDA",
        "Plot": True,
        "TradingStyle": "scalping",
        "RiskPct": 0.01,
        
        # Strategy selection
        "Strategy_Type": "breakout", # "breakout", "exhaustion", "spread"
        
        # Proxy settings
        "InPlay_Volume_Mult": 3.0,   # Volume must be X times the average to be "in play"
        "Key_Level_Proximity": 0.05, # Distance in $ to a whole number or key level
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "1M" # 1-minute chart for scalping proxy
        self.avg_volume = None
        self.state = "flat"
        self.entry_price_mem = None

    def setup(self):
        ticker = self.parameters.get("Ticker")
        bars = self.get_historical_prices(ticker, 50, "minute")
        if bars is None or len(bars.df) < 20:
            return False
            
        self.df = bars.df.copy()
        
        # Calculate recent average volume (e.g. 20 periods)
        self.avg_volume = self.df['volume'].rolling(20).mean().iloc[-1]
        self.latest = self.df.iloc[-1]
        self.prev = self.df.iloc[-2]
        
        return True

    def _is_in_play(self):
        """Check if stock has high relative volume (In-Play requirement)"""
        if not self.avg_volume or self.avg_volume == 0:
            return False
        return self.latest['volume'] > (self.avg_volume * self.parameters["InPlay_Volume_Mult"])

    def _is_near_key_level(self, price):
        """Check if price is near a major psychological level (e.g., $50, $25, whole numbers)"""
        # Find nearest whole number
        nearest_whole = round(price)
        # Check if within proximity
        return abs(price - nearest_whole) <= self.parameters["Key_Level_Proximity"]

    def get_entry_signal(self):
        if not self._is_in_play():
            return 0
            
        close = self.latest['close']
        high = self.latest['high']
        low = self.latest['low']
        
        strategy = self.parameters["Strategy_Type"]

        if strategy == "breakout":
            # Strategy 1: Continuation Breakout
            # Proxy: Tight consolidation just below a whole number, followed by a volume spike pushing through it.
            nearest_whole = math.ceil(close)
            consolidation_below = (self.prev['high'] < nearest_whole) and (self.prev['high'] >= nearest_whole - self.parameters["Key_Level_Proximity"])
            breakout = close > nearest_whole
            
            if consolidation_below and breakout:
                self.entry_setup = "breakout_long"
                self.key_level = nearest_whole
                return 1

        elif strategy == "exhaustion":
            # Strategy 2: Mean Reversion (Exhaustion)
            # Proxy: Violent, sharp parabolic move into support with volume climax.
            # E.g., 3 consecutive large red candles, ending near a whole number support.
            nearest_whole = math.floor(close)
            near_support = abs(close - nearest_whole) <= self.parameters["Key_Level_Proximity"]
            
            # Simple parabolic definition: rapid drop
            rapid_drop = self.prev['close'] < self.df.iloc[-3]['close'] and close < self.prev['close']
            vol_climax = self.latest['volume'] > self.prev['volume'] * 2
            
            if near_support and rapid_drop and vol_climax:
                self.entry_setup = "exhaustion_long"
                self.key_level = nearest_whole
                return 1

        elif strategy == "spread":
            # Strategy 3: Making the Spread
            # Proxy: In high volatility, assume we capture the spread on both sides.
            # Not easily backtestable with standard market orders, returning 0.
            pass

        return 0

    def get_position_sizing(self):
        signal = self.get_entry_signal()
        if signal == 0:
            return 0
            
        price = self.latest['close']
        cash = self.cash * self.risk_percent
        qty = int(cash / price)
        
        self.state = "long" if signal == 1 else "short"
        self.entry_price_mem = price
        return qty if signal == 1 else -qty

    def get_stop_loss(self, entry: float):
        if self.parameters["Strategy_Type"] == "breakout":
            # Tight invalidation just below the broken key level
            return self.key_level - 0.02
        elif self.parameters["Strategy_Type"] == "exhaustion":
            # Directly behind the support level
            return self.key_level - 0.05
        return None

    def get_take_profit(self, entry: float):
        if self.parameters["Strategy_Type"] == "breakout":
            # Quick scalp of 10-30 cents for the emotional surge
            return entry + 0.20
        elif self.parameters["Strategy_Type"] == "exhaustion":
            # Scalp the immediate snapback bounce
            return entry + 0.30
        return None

