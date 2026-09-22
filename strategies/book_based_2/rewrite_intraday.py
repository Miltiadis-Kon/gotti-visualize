import os

base_dir = r"E:\repos\gotti-visualize\strategies\book_based_2"

hcrl = """\
\"\"\"
Strategy: Bulkowski's Breakout Day Trading Setup (Long)
Source:   https://thepatternsite.com/hcrl.html
Book:     "Swing and Day Trading: Evolution of a Trader" by Thomas Bulkowski

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Identify a Horizontal Consolidation Region (HCR) - a tight, relatively flat price range.
    3. Buy when price breaks 1 penny ($0.01) above the highest high of the HCR.
    4. Initial Stop Loss: 1 penny below the lowest low of the HCR.
    5. Trailing Stop: Move the stop up to 1 penny below the prior 5-min candle's low as price climbs.
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset
import pandas as pd

class HCRBreakoutLong(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "ConsolidationBars": 15,    
        "MaxRangePct": 0.003,       
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.hcr_high = None
        self.hcr_low = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 30, "minute")
        if bars is None or len(bars.df) < 20: return
            
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
        if len(df) < self.parameters["ConsolidationBars"] + 1: return

        current_price = self.get_last_price(symbol)
        prior_candle = df.iloc[-2]  # prior completed 5m candle
        
        pos = self.get_position(symbol)
        
        if pos is None:
            window = df.iloc[-self.parameters["ConsolidationBars"]-1:-1]
            highest_high = window['high'].max()
            lowest_low = window['low'].min()
            range_pct = (highest_high - lowest_low) / lowest_low
            
            if range_pct <= self.parameters["MaxRangePct"]:
                self.hcr_high = highest_high
                self.hcr_low = lowest_low
                
            if self.hcr_high is not None and current_price >= (self.hcr_high + 0.01):
                qty = (self.portfolio_value * self.risk_percent) / max(0.01, (current_price - (self.hcr_low - 0.01)))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
        else:
            trailing_stop_price = prior_candle['low'] - 0.01
            initial_stop_price = self.hcr_low - 0.01 if self.hcr_low else trailing_stop_price
            stop_price = max(trailing_stop_price, initial_stop_price)
            
            if current_price <= stop_price:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
                self.hcr_high = None
                self.hcr_low = None
"""

hcrs = """\
\"\"\"
Strategy: Bulkowski's Breakout Day Trading Setup (Short)
Source:   https://thepatternsite.com/hcrs.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Identify a Horizontal Consolidation Region (HCR).
    3. Sell Short when price breaks 1 penny ($0.01) below the lowest low of the HCR.
    4. Initial Stop Loss: 1 penny above the highest high of the HCR.
    5. Trailing Stop: Move the stop down to 1 penny above the prior 5-min candle's high.
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class HCRBreakoutShort(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "ConsolidationBars": 15,
        "MaxRangePct": 0.003,
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.hcr_high = None
        self.hcr_low = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 30, "minute")
        if bars is None or len(bars.df) < 20: return
            
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
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
                qty = (self.portfolio_value * self.risk_percent) / max(0.01, ((self.hcr_high + 0.01) - current_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "sell")
                    self.submit_order(order)
        else:
            trailing_stop_price = prior_candle['high'] + 0.01
            initial_stop_price = self.hcr_high + 0.01 if self.hcr_high else trailing_stop_price
            stop_price = min(trailing_stop_price, initial_stop_price)
            
            if current_price >= stop_price:
                order = self.create_order(symbol, abs(pos.quantity), "buy")
                self.submit_order(order)
                self.hcr_high = None
                self.hcr_low = None
"""

fib = """\
\"\"\"
Strategy: Bulkowski's Fibonacci Retrace Swing (Day Trade)
Source:   https://thepatternsite.com/Daytrade.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Straight-line run: At least 3-5 consecutive candles in one direction (e.g., up) with no overlapping real bodies.
    3. Retrace: Wait for the price to retrace 38% to 62% of the run.
    4. Entry: Buy when the price reverses (closes higher) after hitting the retrace zone.
    5. Stop: 1 penny below the swing low.
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class FibRetraceSwing(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.swing_high = None
        self.swing_low = None
        self.stop_price = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 20, "minute")
        if bars is None or len(bars.df) < 10: return
        
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
        if len(df) < 10: return
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # Check for 3 straight green candles followed by a red retrace
            c1, c2, c3, c4 = df.iloc[-5], df.iloc[-4], df.iloc[-3], df.iloc[-2]
            
            # Straight-line uprun (3 consecutive higher closes)
            if c1['close'] < c2['close'] < c3['close']:
                self.swing_low = min(c1['low'], c2['low'], c3['low'])
                self.swing_high = c3['high']
                
            if self.swing_high and self.swing_low:
                move = self.swing_high - self.swing_low
                retrace_38 = self.swing_high - (move * 0.382)
                retrace_62 = self.swing_high - (move * 0.618)
                
                # If we entered the retrace zone and just closed green
                if retrace_62 <= current_price <= retrace_38 and df.iloc[-1]['close'] > df.iloc[-1]['open']:
                    self.stop_price = self.swing_low - 0.01
                    qty = (self.portfolio_value * self.risk_percent) / max(0.01, (current_price - self.stop_price))
                    if qty >= 1:
                        order = self.create_order(symbol, int(qty), "buy")
                        self.submit_order(order)
        else:
            if current_price <= self.stop_price or current_price >= self.swing_high:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
                self.swing_high = None
                self.swing_low = None
"""

gap_long = """\
\"\"\"
Strategy: Bulkowski's Gap Setup (Long)
Source:   https://thepatternsite.com/gapsetupl.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Price gaps DOWN at the open (Today's open < Yesterday's close).
    3. Buy when price crosses 1 penny above the high of the first 5-min candle.
    4. Stop loss 1 penny below the low of the first 5-min candle (or current day low).
    5. Target: Gap fill (Yesterday's close).
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class OpeningGapLong(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.first_candle_high = None
        self.first_candle_low = None
        self.yesterday_close = None
        self.day_traded = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        current_time = self.get_datetime()
        
        # Reset daily variables on new day
        if self.day_traded != current_time.date():
            self.first_candle_high = None
            self.first_candle_low = None
            self.yesterday_close = None
            
            # Get yesterday's close
            daily_bars = self.get_historical_prices(symbol, 2, "day")
            if daily_bars is not None and len(daily_bars.df) >= 2:
                self.yesterday_close = daily_bars.df.iloc[-2]['close']

        bars = self.get_historical_prices(symbol, 10, "minute")
        if bars is None or len(bars.df) < 5: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
        current_price = self.get_last_price(symbol)
        
        # Identify first 5-min candle of the day
        today_bars = df[df.index.date == current_time.date()]
        if len(today_bars) >= 1 and self.first_candle_high is None:
            first_candle = today_bars.iloc[0]
            if self.yesterday_close and first_candle['open'] < self.yesterday_close:
                self.first_candle_high = first_candle['high']
                self.first_candle_low = first_candle['low']
                
        pos = self.get_position(symbol)
        if pos is None:
            if self.day_traded != current_time.date() and self.first_candle_high is not None:
                if current_price >= self.first_candle_high + 0.01:
                    qty = (self.portfolio_value * self.risk_percent) / max(0.01, (current_price - (self.first_candle_low - 0.01)))
                    if qty >= 1:
                        order = self.create_order(symbol, int(qty), "buy")
                        self.submit_order(order)
                        self.day_traded = current_time.date()
        else:
            if current_price <= (self.first_candle_low - 0.01) or current_price >= self.yesterday_close:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
"""

gap_short = """\
\"\"\"
Strategy: Bulkowski's Gap Setup (Short)
Source:   https://thepatternsite.com/gapsetups.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Price gaps UP at the open (Today's open > Yesterday's close).
    3. Sell short when price crosses 1 penny below the low of the first 5-min candle.
    4. Stop loss 1 penny above the high of the first 5-min candle.
    5. Target: Gap fill (Yesterday's close).
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class OpeningGapShort(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.first_candle_high = None
        self.first_candle_low = None
        self.yesterday_close = None
        self.day_traded = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        current_time = self.get_datetime()
        
        if self.day_traded != current_time.date():
            self.first_candle_high = None
            self.first_candle_low = None
            self.yesterday_close = None
            
            daily_bars = self.get_historical_prices(symbol, 2, "day")
            if daily_bars is not None and len(daily_bars.df) >= 2:
                self.yesterday_close = daily_bars.df.iloc[-2]['close']

        bars = self.get_historical_prices(symbol, 10, "minute")
        if bars is None or len(bars.df) < 5: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
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
                    qty = (self.portfolio_value * self.risk_percent) / max(0.01, ((self.first_candle_high + 0.01) - current_price))
                    if qty >= 1:
                        order = self.create_order(symbol, int(qty), "sell")
                        self.submit_order(order)
                        self.day_traded = current_time.date()
        else:
            if current_price >= (self.first_candle_high + 0.01) or current_price <= self.yesterday_close:
                order = self.create_order(symbol, abs(pos.quantity), "buy")
                self.submit_order(order)
"""

retrace_long = """\
\"\"\"
Strategy: Bulkowski's Retrace (Long)
Source:   https://thepatternsite.com/retracel.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Downtrend of at least 3 consecutive red candles.
    3. Buy when price crosses 1 penny above the prior 5-minute candle's high.
    4. Stop loss 1 penny below the swing low.
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class RetraceLong(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.trigger_high = None
        self.swing_low = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 15, "minute")
        if bars is None or len(bars.df) < 10: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
            
            # 3 consecutive red candles
            if c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open']:
                self.trigger_high = c3['high'] + 0.01
                self.swing_low = c3['low'] - 0.01
                
            if self.trigger_high is not None and current_price >= self.trigger_high:
                qty = (self.portfolio_value * self.risk_percent) / max(0.01, (current_price - self.swing_low))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
                    self.trigger_high = None
        else:
            if current_price <= self.swing_low or current_price >= (pos.avg_price * 1.01): # 1% trailing/take profit for day trade
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
"""

retrace_short = """\
\"\"\"
Strategy: Bulkowski's Retrace (Short)
Source:   https://thepatternsite.com/retraces.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. Uptrend of at least 3 consecutive green candles.
    3. Sell short when price crosses 1 penny below the prior 5-minute candle's low.
    4. Stop loss 1 penny above the swing high.
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset

class RetraceShort(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.trigger_low = None
        self.swing_high = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 15, "minute")
        if bars is None or len(bars.df) < 10: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
            
            # 3 consecutive green candles
            if c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open']:
                self.trigger_low = c3['low'] - 0.01
                self.swing_high = c3['high'] + 0.01
                
            if self.trigger_low is not None and current_price <= self.trigger_low:
                qty = (self.portfolio_value * self.risk_percent) / max(0.01, (self.swing_high - current_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "sell")
                    self.submit_order(order)
                    self.trigger_low = None
        else:
            if current_price >= self.swing_high or current_price <= (pos.avg_price * 0.99):
                order = self.create_order(symbol, abs(pos.quantity), "buy")
                self.submit_order(order)
"""

rsi = """\
\"\"\"
Strategy: Bulkowski's RSI Setup
Source:   https://thepatternsite.com/rsisetup.html

EXACT RULES (5-Minute Timeframe):
    1. 5-minute chart.
    2. RSI(14) falls below 30 (oversold) and crosses back above 30.
    3. Buy when RSI crosses above 30.
    4. Stop loss below the local swing low.
\"\"\"

from lumibot.strategies import Strategy
from lumibot.entities import Asset
import pandas_ta as ta

class RSISetup(Strategy):
    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": False,
    }

    def initialize(self):
        self.sleeptime = "5M"
        self.risk_percent = 0.02
        self.stop_price = None
        
    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"].symbol
        bars = self.get_historical_prices(symbol, 40, "minute")
        if bars is None or len(bars.df) < 30: return
        df = bars.df
        if len(df) > 1 and (df.index[-1] - df.index[-2]).total_seconds() <= 60:
            df = df.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        df.ta.rsi(length=14, append=True)
        if 'RSI_14' not in df.columns: return
        
        rsi_prior = df['RSI_14'].iloc[-3]
        rsi_current = df['RSI_14'].iloc[-2]
        
        if pos is None:
            if rsi_prior < 30 and rsi_current >= 30:
                self.stop_price = df['low'].iloc[-3:-1].min() - 0.01
                qty = (self.portfolio_value * self.risk_percent) / max(0.01, (current_price - self.stop_price))
                if qty >= 1:
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
        else:
            if current_price <= self.stop_price or rsi_current >= 70:
                order = self.create_order(symbol, pos.quantity, "sell")
                self.submit_order(order)
"""

# Map contents to file names
files = {
    "hcr_breakout_long.py": hcrl,
    "hcr_breakout_short.py": hcrs,
    "fib_retrace_swing.py": fib,
    "gap_setup_long.py": gap_long,
    "gap_setup_short.py": gap_short,
    "retrace_long.py": retrace_long,
    "retrace_short.py": retrace_short,
    "rsi_setup.py": rsi
}

for filename, content in files.items():
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
print("All 8 files rewritten successfully with strict 5-minute rules.")
