import os

base_dir = r"E:\repos\gotti-visualize\strategies\swing"
os.makedirs(base_dir, exist_ok=True)

# 1. HCR Breakout Long (Swing)
hcr_long = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class HCRBreakoutLongSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "ConsolidationDays": 15,
        "MaxRangePct": 0.05,  # 5% max variance over 15 days for a swing consolidation
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = self.parameters["Ticker"]
        if hasattr(symbol, "symbol"): symbol = symbol.symbol
        
        bars = self.get_historical_prices(symbol, 40, "day")
        if bars is None or len(bars.df) < 20:
            return
            
        df = bars.df
        df.ta.atr(length=14, append=True)
        atr = df['ATRr_14'].iloc[-1]
        
        current_price = self.get_last_price(symbol)
        pos = self.get_position(symbol)
        
        if pos is None:
            # Check for consolidation in the last N days
            period = df.iloc[-self.parameters["ConsolidationDays"]:]
            highest = period['high'].max()
            lowest = period['low'].min()
            
            range_pct = (highest - lowest) / lowest
            
            if range_pct <= self.parameters["MaxRangePct"]:
                # Breakout entry: current price clears the consolidation high by a small percentage
                if current_price > (highest * 1.002):
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
                    self.stop_price = current_price - (atr * self.parameters["ATR_Multiplier"])
        else:
            # Trailing stop using ATR
            new_stop = current_price - (atr * self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
                
            if current_price <= self.stop_price:
                self.sell_all()
"""

# 2. HCR Breakout Short (Swing)
hcr_short = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class HCRBreakoutShortSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "ConsolidationDays": 15,
        "MaxRangePct": 0.05,
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

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
            period = df.iloc[-self.parameters["ConsolidationDays"]:]
            highest = period['high'].max()
            lowest = period['low'].min()
            range_pct = (highest - lowest) / lowest
            
            if range_pct <= self.parameters["MaxRangePct"]:
                # Breakdown entry
                if current_price < (lowest * 0.998):
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                    order = self.create_order(symbol, int(qty), "sell")
                    self.submit_order(order)
                    self.stop_price = current_price + (atr * self.parameters["ATR_Multiplier"])
        else:
            new_stop = current_price + (atr * self.parameters["ATR_Multiplier"])
            if not hasattr(self, "stop_price") or new_stop < self.stop_price:
                self.stop_price = new_stop
                
            if current_price >= self.stop_price:
                self.sell_all()
"""

# 3. Fib Retrace Swing (Swing)
fib = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class FibRetraceSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "TrendDays": 10,
        "RiskPct": 0.02,
        "ATR_Multiplier": 2.0,
    }

    def initialize(self):
        self.sleeptime = "1D"

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
                qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                order = self.create_order(symbol, int(qty), "buy")
                self.submit_order(order)
                self.stop_price = current_price - (atr * self.parameters["ATR_Multiplier"])
        else:
            new_stop = current_price - (atr * self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
            if current_price <= self.stop_price:
                self.sell_all()
"""

# 4. Gap Setup Long (Swing)
gap_long = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class GapSetupLongSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "GapPct": -0.02, # 2% gap down
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

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
                qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                order = self.create_order(symbol, int(qty), "buy")
                self.submit_order(order)
                self.target_price = yest_close # Target the gap fill
                self.stop_price = today_open - (atr * 0.5) # Tight stop below today's open
        else:
            if current_price >= getattr(self, "target_price", float('inf')) or current_price <= getattr(self, "stop_price", 0):
                self.sell_all()
"""

# 5. Gap Setup Short (Swing)
gap_short = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class GapSetupShortSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "GapPct": 0.02, # 2% gap up
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

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
            yest_close = df['close'].iloc[-2]
            today_open = df['open'].iloc[-1]
            gap = (today_open - yest_close) / yest_close
            
            if gap >= self.parameters["GapPct"] and current_price < today_open:
                qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                order = self.create_order(symbol, int(qty), "sell")
                self.submit_order(order)
                self.target_price = yest_close
                self.stop_price = today_open + (atr * 0.5)
        else:
            if current_price <= getattr(self, "target_price", 0) or current_price >= getattr(self, "stop_price", float('inf')):
                self.sell_all()
"""

# 6. Retrace Long (Swing)
retrace_long = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class RetraceLongSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

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
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                    order = self.create_order(symbol, int(qty), "buy")
                    self.submit_order(order)
                    self.stop_price = current_price - (atr * self.parameters["ATR_Multiplier"])
        else:
            new_stop = current_price - (atr * self.parameters["ATR_Multiplier"])
            if new_stop > getattr(self, "stop_price", 0):
                self.stop_price = new_stop
            if current_price <= self.stop_price:
                self.sell_all()
"""

# 7. Retrace Short (Swing)
retrace_short = """import pandas_ta as ta
from lumibot.strategies.strategy import Strategy

class RetraceShortSwing(Strategy):
    parameters = {
        "Ticker": "NVDA",
        "RiskPct": 0.02,
        "ATR_Multiplier": 1.5,
    }

    def initialize(self):
        self.sleeptime = "1D"

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
            # 3 Consecutive Up Days
            c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
            if (c1['close'] > c1['open'] and 
                c2['close'] > c2['open'] and 
                c3['close'] > c3['open']):
                
                # Short if today breaks below yesterday's low
                if current_price < c3['low']:
                    qty = (self.get_portfolio_value() * self.parameters["RiskPct"]) / max(1, (atr * self.parameters["ATR_Multiplier"]))
                    order = self.create_order(symbol, int(qty), "sell")
                    self.submit_order(order)
                    self.stop_price = current_price + (atr * self.parameters["ATR_Multiplier"])
        else:
            new_stop = current_price + (atr * self.parameters["ATR_Multiplier"])
            if not hasattr(self, "stop_price") or new_stop < self.stop_price:
                self.stop_price = new_stop
            if current_price >= self.stop_price:
                self.sell_all()
"""

# 8. RSI Setup (Swing)
rsi_setup = """import pandas_ta as ta
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
"""

files = {
    "hcr_breakout_long.py": hcr_long,
    "hcr_breakout_short.py": hcr_short,
    "fib_retrace_swing.py": fib,
    "gap_setup_long.py": gap_long,
    "gap_setup_short.py": gap_short,
    "retrace_long.py": retrace_long,
    "retrace_short.py": retrace_short,
    "rsi_setup.py": rsi_setup
}

for filename, content in files.items():
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print(f"Successfully generated 8 Swing strategies in {base_dir}")
