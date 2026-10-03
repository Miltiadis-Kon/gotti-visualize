"""
Strategy: Bulkowski's Breakout Day Trading Setup (Short)
Source:   https://thepatternsite.com/hcrs.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Identify a Horizontal Consolidation Region (HCR).
    3. Sell Short when price breaks 1 penny ($0.01) below the lowest low of the HCR.
    4. Initial Stop Loss: 1 penny above the highest high of the HCR.
    5. Trailing Stop: Move the stop down to 1 penny above the prior 5-min candle's high.
"""

from lumibot.entities import Asset
import pandas as pd
import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class HCRBreakoutShort(StrategyBaseplate):
    parameters = {
        **StrategyBaseplate.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "ConsolidationBars": 15,
        "MaxRangePct": 0.003,
        "Plot": False,
        "TradingStyle": "day_trading",
        "TrailingStop": True,
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.hcr_high = None
        self.hcr_low = None
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
            
        if len(df) < self.parameters["ConsolidationBars"] + 1: return

        current_price = self.get_last_price(symbol)
        prior_candle = df.iloc[-2]
        
        pos = self.get_position(symbol)
        
        if pos is None:
            window = df.iloc[-self.parameters["ConsolidationBars"]-1:-1]
            highest_high = window['high'].max()
            lowest_low = window['low'].min()
            range_pct = (highest_high - lowest_low) / lowest_low
            
            if range_pct <= self.parameters["MaxRangePct"]:
                self.hcr_high = highest_high
                self.hcr_low = lowest_low
                
            if self.hcr_low is not None and current_price <= (self.hcr_low - 0.01):
                custom_sl = self.hcr_high + 0.01
                
                # Compute 5-min ATR locally
                df.ta.atr(length=14, append=True)
                atr_col = [c for c in df.columns if c.startswith('ATRr')]
                atr = df[atr_col[0]].iloc[-2] if atr_col else None
                
                risk_levels = self.optimize_sl_tp(
                    entry_price=current_price,
                    stop_loss=custom_sl,
                    take_profit=None,
                    side="sell",
                    atr=atr,
                    trading_style=self.parameters.get("TradingStyle", "day_trading"),
                    trailing_stop=True,
                )
                self.stop_price = risk_levels.stop_loss
                self.current_trailing_stop = self.stop_price
                
                risk_budget = self.get_portfolio_value() * self.risk_percent
                qty = risk_budget / max(0.01, (self.stop_price - current_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "sell_short")
                    self.submit_order(order)
                    self.hcr_high = None
                    self.hcr_low = None
        else:
            trailing_stop_price = prior_candle['high'] + 0.01
            initial_stop_price = getattr(self, "stop_price", self.hcr_high + 0.01 if self.hcr_high else trailing_stop_price)
            stop_price = min(trailing_stop_price, initial_stop_price)
            
            if current_price >= stop_price:
                order = self.create_order(symbol, abs(pos.quantity), "buy")
                self.submit_order(order)
                self.hcr_high = None
                self.hcr_low = None
                self.stop_price = None
                self.current_trailing_stop = None
