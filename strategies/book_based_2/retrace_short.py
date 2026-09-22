"""
Strategy: Bulkowski's Retrace (Short)
Source:   https://thepatternsite.com/retraces.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Uptrend of at least 3 consecutive green candles.
    3. Sell short when price crosses 1 penny below the prior 5-minute candle's low.
    4. Stop loss 1 penny above the swing high.
"""

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class RetraceShort(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.trigger_low = None
        self.swing_high = None
        
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
        
        if pos is None:
            c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
            
            # 3 consecutive green candles
            if c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open']:
                self.trigger_low = c3['low'] - 0.01
                self.swing_high = c3['high'] + 0.01
                
            if self.trigger_low is not None and current_price <= self.trigger_low:
                qty = (10000.0 * self.risk_percent) / max(0.01, (self.swing_high - current_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "sell")
                    self.submit_order(order)
                    self.trigger_low = None
        else:
            if current_price >= self.swing_high or current_price <= (pos.avg_price * 0.99):
                order = self.create_order(symbol, abs(pos.quantity), "buy")
                self.submit_order(order)
