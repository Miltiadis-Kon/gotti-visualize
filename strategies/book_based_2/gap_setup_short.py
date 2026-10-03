"""
Strategy: Bulkowski's Gap Setup (Short)
Source:   https://thepatternsite.com/gapsetups.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Price gaps UP at the open (Today's open > Yesterday's close).
    3. Sell short when price crosses 1 penny below the low of the first 5-min candle.
    4. Stop loss 1 penny above the high of the first 5-min candle.
    5. Target: Gap fill (Yesterday's close).
"""

from lumibot.entities import Asset
import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class OpeningGapShort(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
        "TradingStyle": "day_trading",
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.first_candle_high = None
        self.first_candle_low = None
        self.yesterday_close = None
        self.day_traded = None
        self.stop_price = None
        self.target_price = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"):
            symbol = symbol.symbol
        current_time = self.get_datetime()
        
        if self.day_traded != current_time.date():
            self.first_candle_high = None
            self.first_candle_low = None
            self.yesterday_close = None
            self.stop_price = None
            self.target_price = None
            
            daily_bars = self.get_historical_prices(symbol, 2, "day")
            if daily_bars is not None and len(daily_bars.df) >= 2:
                self.yesterday_close = daily_bars.df.iloc[-2]['close']

        bars = self.get_historical_prices(symbol, 200, "minute")
        if bars is None or len(bars.df) < 50: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
        if len(df) < 15: return
            
        current_price = self.get_last_price(symbol)
        
        today_bars = df[df.index.date == current_time.date()]
        if len(today_bars) >= 1 and self.first_candle_high is None:
            first_candle = today_bars.iloc[0]
            if self.yesterday_close and first_candle['open'] > self.yesterday_close:
                self.first_candle_high = first_candle['high']
                self.first_candle_low = first_candle['low']
                
        pos = self.get_position(symbol)
        if pos is None:
            if self.day_traded != current_time.date() and self.first_candle_low is not None:
                if current_price <= self.first_candle_low - 0.01:
                    # Custom calculation
                    custom_sl = self.first_candle_high + 0.01
                    custom_tp = self.yesterday_close
                    
                    # Compute 5-min ATR locally
                    df.ta.atr(length=14, append=True)
                    atr_col = [c for c in df.columns if c.startswith('ATRr')]
                    atr = df[atr_col[0]].iloc[-2] if atr_col else None
                    
                    # Pass through ATR optimization function
                    risk_levels = self.optimize_sl_tp(
                        entry_price=current_price,
                        stop_loss=custom_sl,
                        take_profit=custom_tp,
                        side="sell",
                        atr=atr,
                        trading_style=self.parameters.get("TradingStyle", "day_trading"),
                    )
                    self.stop_price = risk_levels.stop_loss
                    self.target_price = risk_levels.take_profit
                    
                    risk_dist = max(0.01, self.stop_price - current_price)
                    risk_budget = self.get_portfolio_value() * self.risk_percent
                    qty = risk_budget / risk_dist
                    if qty >= 1:
                        order = self.create_order(symbol, int(qty), "sell_short")
                        self.submit_order(order)
                        self.day_traded = current_time.date()
        else:
            sl = getattr(self, "stop_price", self.first_candle_high + 0.01 if self.first_candle_high else None)
            tp = getattr(self, "target_price", self.yesterday_close)
            if (sl and current_price >= sl) or (tp and current_price <= tp):
                order = self.create_order(symbol, abs(pos.quantity), "buy")
                self.submit_order(order)
                self.stop_price = None
                self.target_price = None
