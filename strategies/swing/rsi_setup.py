import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class RSISetupSwing(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": "NVDA",
        "RsiLength": 14,
        "RsiOversold": 30,
        "RsiOverbought": 70,
        "RiskPct": 0.02,
        "ATR_Multiplier": 2.0,
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
            # Trailing stop or RSI overbought target
            new_stop = self.get_trailing_stop(current_price, side="buy", sl_multiplier=self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
                
            if (self.stop_price and current_price <= self.stop_price) or current_rsi >= self.parameters["RsiOverbought"]:
                self.sell_all()
                self.stop_price = None
                self.current_trailing_stop = None
