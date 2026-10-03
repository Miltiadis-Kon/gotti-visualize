"""
Strategy: Bulkowski's RSI Setup
Source:   https://thepatternsite.com/rsisetup.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. RSI(14) falls below 30 (oversold) and crosses back above 30.
    3. Buy when RSI crosses above 30.
    4. Stop loss below the local swing low.
"""

from lumibot.entities import Asset
import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class RSISetup(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
        "TradingStyle": "day_trading",
        "ATR_Multiplier": 1.5,
        "TrailingStop": False,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.stop_price = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"):
            symbol = symbol.symbol
        bars = self.get_historical_prices(symbol, 200, "minute")
        if bars is None or len(bars.df) < 50: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
        if len(df) < 15: return
            
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        df.ta.rsi(length=14, append=True)
        if 'RSI_14' not in df.columns: return
        
        rsi_prior = df['RSI_14'].iloc[-3]
        rsi_current = df['RSI_14'].iloc[-2]
        
        if pos is None:
            if rsi_prior < 30 and rsi_current >= 30:
                custom_sl = df['low'].iloc[-3:-1].min() - 0.01
                # Compute 5-min ATR locally from the bars already fetched
                df.ta.atr(length=14, append=True)
                atr_col = [c for c in df.columns if c.startswith('ATRr')]
                atr = df[atr_col[0]].iloc[-2] if atr_col else None
                risk_levels = self.optimize_sl_tp(
                    entry_price=current_price,
                    stop_loss=custom_sl,
                    take_profit=None,
                    side="buy",
                    atr=atr,
                    trading_style=self.parameters.get("TradingStyle", "day_trading"),
                    sl_multiplier=self.parameters.get("ATR_Multiplier", 1.5),
                    trailing_stop=self.parameters.get("TrailingStop", False),
                )
                self.stop_price = risk_levels.stop_loss
                risk_budget = self.get_portfolio_value() * self.risk_percent
                qty = risk_budget / max(0.01, (current_price - self.stop_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
        else:
            if self.parameters.get("TrailingStop", False):
                new_stop = self.get_trailing_stop(current_price, side="buy", sl_multiplier=self.parameters.get("ATR_Multiplier", 1.5))
                if new_stop > getattr(self, "stop_price", 0):
                    self.stop_price = new_stop
                    
            if self.stop_price is not None and (current_price <= self.stop_price or rsi_current >= 70):
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)

