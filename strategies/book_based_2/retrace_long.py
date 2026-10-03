"""
Strategy: Bulkowski's Retrace (Long)
Source:   https://thepatternsite.com/retracel.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Downtrend of at least 3 consecutive red candles.
    3. Buy when price crosses 1 penny above the prior 5-minute candle's high.
    4. Stop loss 1 penny below the swing low.
"""

from lumibot.entities import Asset
import pandas_ta as ta
from strategies.strat_baseplate import StrategyBaseplate


class RetraceLong(StrategyBaseplate):
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
        self.trigger_high = None
        self.swing_low = None
        self.stop_price = None
        self.target_price = None
        
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
        
        if pos is None:
            c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
            
            # 3 consecutive red candles
            if c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open']:
                self.trigger_high = c3['high'] + 0.01
                self.swing_low = c3['low'] - 0.01
                
            if self.trigger_high is not None and current_price >= self.trigger_high:
                custom_sl = self.swing_low
                custom_tp = current_price * 1.01  # 1% day trade target
                
                # Compute 5-min ATR locally
                df.ta.atr(length=14, append=True)
                atr_col = [c for c in df.columns if c.startswith('ATRr')]
                atr = df[atr_col[0]].iloc[-2] if atr_col else None
                
                risk_levels = self.optimize_sl_tp(
                    entry_price=current_price,
                    stop_loss=custom_sl,
                    take_profit=custom_tp,
                    side="buy",
                    atr=atr,
                    trading_style=self.parameters.get("TradingStyle", "day_trading"),
                )
                self.stop_price = risk_levels.stop_loss
                self.target_price = risk_levels.take_profit
                
                risk_budget = self.get_portfolio_value() * self.risk_percent
                qty = risk_budget / max(0.01, (current_price - self.stop_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
                    self.trigger_high = None
                    self.swing_low = None
        else:
            sl = getattr(self, "stop_price", self.swing_low)
            tp = getattr(self, "target_price", pos.avg_price * 1.01 if hasattr(pos, 'avg_price') else None)
            if (sl and current_price <= sl) or (tp and current_price >= tp):
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
                self.stop_price = None
                self.target_price = None
