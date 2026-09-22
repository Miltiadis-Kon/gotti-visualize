import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class RetraceLongSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"): symbol = symbol.symbol
        
        bars = self.get_historical_prices(symbol, 40, "day")
        if bars is None or len(bars.df) < 10: return
            
        df = bars.df
        df.ta.atr(length=14, append=True)
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # 3 Consecutive Down Days
            c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
            if (c1['close'] < c1['open'] and 
                c2['close'] < c2['open'] and 
                c3['close'] < c3['open']):
                
                # Buy if today breaks above yesterday's high
                if current_price > c3['high']:
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
                    self.stop_price = current_price - (atr * self.parameters["ATR_Multiplier"])
        else:
            new_stop = current_price - (atr * self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
            if current_price <= self.stop_price:
                self.sell_all()
