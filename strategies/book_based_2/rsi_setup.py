"""
Strategy: Bulkowski's RSI Setup
Source:   https://thepatternsite.com/rsisetup.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. RSI(14) falls below 30 (oversold) and crosses back above 30.
    3. Buy when RSI crosses above 30.
    4. Stop loss below the local swing low.
"""

from lumibot.strategies import Strategy
from lumibot.entities import Asset
import pandas_ta as ta

class RSISetup(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.stop_price = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 200, "minute")
        if bars is None or len(bars.df) < 50: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
        if len(df) < 15: return
            
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        df.ta.rsi(length=14, append=True)
        if 'RSI_14' not in df.columns: return
        
        rsi_prior = df['RSI_14'].iloc[-3]
        rsi_current = df['RSI_14'].iloc[-2]
        
        if pos is None:
            if rsi_prior < 30 and rsi_current >= 30:
                self.stop_price = df['low'].iloc[-3:-1].min() - 0.01
                qty = (10000.0 * self.risk_percent) / max(0.01, (current_price - self.stop_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
        else:
            if current_price <= self.stop_price or rsi_current >= 70:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
