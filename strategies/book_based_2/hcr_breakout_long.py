"""
Strategy: Bulkowski's Breakout Day Trading Setup (Long)
Source:   https://thepatternsite.com/hcrl.html
Book:     "Swing and Day Trading: Evolution of a Trader" by Thomas Bulkowski

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Identify a Horizontal Consolidation Region (HCR) - a tight, relatively flat price range.
    3. Buy when price breaks 1 penny ($0.01) above the highest high of the HCR.
    4. Initial Stop Loss: 1 penny below the lowest low of the HCR.
    5. Trailing Stop: Move the stop up to 1 penny below the prior 5-min candle's low as price climbs.
"""

from lumibot.strategies import Strategy
from lumibot.entities import Asset
import pandas as pd

class HCRBreakoutLong(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "ConsolidationBars": 15,    
        "MaxRangePct": 0.003,       
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.hcr_high = None
        self.hcr_low = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 200, "minute")
        if bars is None or len(bars.df) < 50: return
            
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
        if len(df) < 15: return
            
        if len(df) < self.parameters["ConsolidationBars"] + 1: return

        current_price = self.get_last_price(symbol)
        prior_candle = df.iloc[-2]  # prior completed 5m candle
        
        pos = self.get_position(symbol)
        
        if pos is None:
            window = df.iloc[-self.parameters["ConsolidationBars"]-1:-1]
            highest_high = window['high'].max()
            lowest_low = window['low'].min()
            range_pct = (highest_high - lowest_low) / lowest_low
            
            if range_pct <= self.parameters["MaxRangePct"]:
                self.hcr_high = highest_high
                self.hcr_low = lowest_low
                
            if self.hcr_high is not None and current_price >= (self.hcr_high + 0.01):
                qty = (10000.0 * self.risk_percent) / max(0.01, (current_price - (self.hcr_low - 0.01)))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
        else:
            trailing_stop_price = prior_candle['low'] - 0.01
            initial_stop_price = self.hcr_low - 0.01 if self.hcr_low else trailing_stop_price
            stop_price = max(trailing_stop_price, initial_stop_price)
            
            if current_price <= stop_price:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
                self.hcr_high = None
                self.hcr_low = None
