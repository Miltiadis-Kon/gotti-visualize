import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class RetraceLongSwing(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": "NVDA",
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
        "TradingStyle": "swing_trading",
        "TrailingStop": True,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "1D"
        self.stop_price = None
        self.target_price = None
        self.current_trailing_stop = None

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
                    # Custom calculation
                    custom_sl = current_price - (atr * self.parameters["ATR_Multiplier"])
                    
                    # Pass through ATR optimization function BEFORE order submission
                    risk_levels = self.optimize_sl_tp(
                        entry_price=current_price,
                        stop_loss=custom_sl,
                        take_profit=None,
                        side="buy",
                        atr=atr,
                        trading_style=self.parameters.get("TradingStyle", "swing_trading"),
                        sl_multiplier=self.parameters["ATR_Multiplier"],
                        trailing_stop=True,
                    )
                    self.stop_price = risk_levels.stop_loss
                    self.current_trailing_stop = self.stop_price
                    
                    risk_dist = max(0.01, current_price - self.stop_price)
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / risk_dist
                    if qty >= 1:
                        order = self.create_order(symbol, int(qty), "buy")
                        self.submit_order(order)
        else:
            new_stop = self.get_trailing_stop(current_price, side="buy", sl_multiplier=self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
            if self.stop_price and current_price <= self.stop_price:
                self.sell_all()
                self.stop_price = None
                self.current_trailing_stop = None
