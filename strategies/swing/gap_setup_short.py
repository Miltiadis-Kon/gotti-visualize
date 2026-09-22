import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class GapSetupShortSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "GapPct": 0.02, # 2% gap up
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"): symbol = symbol.symbol
        
        bars = self.get_historical_prices(symbol, 40, "day")
        if bars is None or len(bars.df) < 5: return
            
        df = bars.df
        df.ta.atr(length=14, append=True)
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            yest_close = df['close'].iloc[-2]
            today_open = df['open'].iloc[-1]
            gap = (today_open - yest_close) / yest_close
            
            if gap >= self.parameters["GapPct"] and current_price < today_open:
                qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                order = self.create_order(symbol, int(qty), "sell")
                self.submit_order(order)
                self.target_price = yest_close
                self.stop_price = today_open + (atr * 0.5)
        else:
            if current_price <= getattr(self, "target_price", 0) or current_price >= getattr(self, "stop_price", float('inf')):
                self.sell_all()
