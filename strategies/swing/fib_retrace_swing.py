import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class FibRetraceSwing(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": "NVDA",
        "TrendDays": 10,
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
        
        bars = self.get_historical_prices(symbol, 40, "day")
        if bars is None or len(bars.df) < 20: return
            
        df = bars.df
        df.ta.atr(length=14, append=True)
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # Measure a 10-day swing range
            period = df.iloc[-self.parameters["TrendDays"]:]
            highest = period['high'].max()
            lowest = period['low'].min()
            
            # Fibonacci zones (38.2% to 61.8% pullback from the high)
            range_diff = highest - lowest
            fib_38 = highest - (0.382 * range_diff)
            fib_61 = highest - (0.618 * range_diff)
            
            # Check if current price is in the golden pocket AND today's candle is green (bullish reversal)
            is_green = df['close'].iloc[-1] > df['open'].iloc[-1]
            if fib_61 <= current_price <= fib_38 and is_green:
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
