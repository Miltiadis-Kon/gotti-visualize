import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class RSISetupSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "RsiLength": 14,
        "RsiOversold": 30,
        "RsiOverbought": 70,
        "RiskPct": 0.02,
        "ATR_Multiplier": 2.0,
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"): symbol = symbol.symbol
        
        bars = self.get_historical_prices(symbol, 60, "day")
        if bars is None or len(bars.df) < 20: return
            
        df = bars.df
        df.ta.rsi(length=self.parameters["RsiLength"], append=True)
        df.ta.atr(length=14, append=True)
        
        rsi_col = f"RSI_{self.parameters['RsiLength']}"
        if rsi_col not in df.columns: return
        
        current_rsi = df[rsi_col].iloc[-1]
        prev_rsi = df[rsi_col].iloc[-2]
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # Cross above oversold threshold
            if prev_rsi < self.parameters["RsiOversold"] and current_rsi >= self.parameters["RsiOversold"]:
                qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                order = self.create_order(symbol, int(qty), "buy")
                self.submit_order(order)
                self.stop_price = current_price - (atr * self.parameters["ATR_Multiplier"])
        else:
            # Trailing stop or RSI overbought target
            new_stop = current_price - (atr * self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
                
            if current_price <= self.stop_price or current_rsi >= self.parameters["RsiOverbought"]:
                self.sell_all()
