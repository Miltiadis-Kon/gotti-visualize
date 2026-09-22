import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class HCRBreakoutShortSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "ConsolidationDays": 15,
        "MaxRangePct": 0.05,
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"): symbol = symbol.symbol
        
        bars = self.get_historical_prices(symbol, 40, "day")
        if bars is None or len(bars.df) < 20: return
            
        df = bars.df
        df.ta.atr(length=14, append=True)
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            period = df.iloc[-self.parameters["ConsolidationDays"]:]
            highest = period['high'].max()
            lowest = period['low'].min()
            range_pct = (highest - lowest) / lowest
            
            if range_pct <= self.parameters["MaxRangePct"]:
                # Breakdown entry
                if current_price < (lowest * 0.998):
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                    order = self.create_order(symbol, int(qty), "sell")
                    self.submit_order(order)
                    self.stop_price = current_price + (atr * self.parameters["ATR_Multiplier"])
        else:
            new_stop = current_price + (atr * self.parameters["ATR_Multiplier"])
            if not hasattr(self, "stop_price") or new_stop < self.stop_price:
                self.stop_price = new_stop
                
            if current_price >= self.stop_price:
                self.sell_all()
