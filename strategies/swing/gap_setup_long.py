import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class GapSetupLongSwing(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": "NVDA",
        "GapPct": -0.02, # 2% gap down
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
        "TradingStyle": "swing_trading",
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
        if bars is None or len(bars.df) < 5: return
            
        df = bars.df
        df.ta.atr(length=14, append=True)
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # Check if today opened significantly lower than yesterday's close
            yest_close = df['close'].iloc[-2]
            today_open = df['open'].iloc[-1]
            gap = (today_open - yest_close) / yest_close
            
            # If gap down > 2% and price is pushing up past the open (gap fade)
            if gap <= self.parameters["GapPct"] and current_price > today_open:
                # Custom calculation
                custom_tp = yest_close # Target the gap fill
                custom_sl = today_open - (atr * 0.5) # Stop below today's open
                
                # Pass through ATR optimization function BEFORE order submission
                risk_levels = self.optimize_sl_tp(
                    entry_price=current_price,
                    stop_loss=custom_sl,
                    take_profit=custom_tp,
                    side="buy",
                    atr=atr,
                    trading_style=self.parameters.get("TradingStyle", "swing_trading"),
                )
                self.target_price = risk_levels.take_profit
                self.stop_price = risk_levels.stop_loss
                
                risk_dist = max(0.01, current_price - self.stop_price)
                qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / risk_dist
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
        else:
            if (self.target_price and current_price >= self.target_price) or (self.stop_price and current_price <= self.stop_price):
                self.sell_all()
                self.target_price = None
                self.stop_price = None
                self.current_trailing_stop = None
