"""
Strategy: Bulkowski's Fibonacci Retrace Swing (Day Trade)
Source:   https://thepatternsite.com/Daytrade.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Straight-line run: At least 3-5 consecutive candles in one direction (e.g., up) with no overlapping real bodies.
    3. Retrace: Wait for the price to retrace 38% to 62% of the run.
    4. Entry: Buy when the price reverses (closes higher) after hitting the retrace zone.
    5. Stop: 1 penny below the swing low.
"""

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class FibRetraceSwing(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.swing_high = None
        self.swing_low = None
        self.stop_price = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 200, "minute")
        if bars is None or len(bars.df) < 50: return
        
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
        if len(df) < 15: return
            
        if len(df) < 10: return
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # Check for 3 straight green candles followed by a red retrace
            c1, c2, c3, c4 = df.iloc[-5], df.iloc[-4], df.iloc[-3], df.iloc[-2]
            
            # Straight-line uprun (3 consecutive higher closes)
            if c1['close'] < c2['close'] < c3['close']:
                self.swing_low = min(c1['low'], c2['low'], c3['low'])
                self.swing_high = c3['high']
                
            if self.swing_high and self.swing_low:
                move = self.swing_high - self.swing_low
                retrace_38 = self.swing_high - (move * 0.382)
                retrace_62 = self.swing_high - (move * 0.618)
                
                # If we entered the retrace zone and just closed green
                if retrace_62 <= current_price <= retrace_38 and df.iloc[-1]['close'] > df.iloc[-1]['open']:
                    self.stop_price = self.swing_low - 0.01
                    qty = (10000.0 * self.risk_percent) / max(0.01, (current_price - self.stop_price))
                    if qty >= 1:
                        order = self.create_order(symbol, int(qty), "buy")
                        self.submit_order(order)
        else:
            if current_price <= self.stop_price or current_price >= self.swing_high:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
                self.swing_high = None
                self.swing_low = None
