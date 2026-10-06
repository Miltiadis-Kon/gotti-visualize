import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

# Add parent directory (strategies) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None
from plot_mixin import PlottableStrategyMixin


class SneakyPivotStrategy(Strategy, PlottableStrategyMixin):
    """
    The Sneaky Pivot Strategy (The Rumers)
    
    Timeframe: 15-minute (15m) chart.
    
    Key Levels:
    - Range High (RH): Previous day's high
    - Range Low (RL): Previous day's low
    - Swing High (SH): Next major swing high above RH
    - Swing Low (SL): Next major swing low below RL
    
    Entry Logic (Morning Reversal at 10:00 AM):
    - Candle 1 (0-15m): Pushes into the key level.
    - Candle 2 (15-30m): Sneaky candle (stabilizing / reversal / long wicks).
    - Candle 3 (30-45m): Triggers entry when price crosses the high/low of Candle 2.
    """

    parameters = {
        "Ticker": Asset(symbol="PLTR", asset_type=Asset.AssetType.STOCK),
        "Plot": True,
        
        "RiskPct": 0.02, # 2% risk per trade
        "AtrMultiplier": 0.5, # Use 0.5 * ATR(14) for boundary testing tolerance
        "LookbackDays": 20, # Days to look back for swing highs/lows
    }

    def initialize(self):
        self.sleeptime = "15M"
        self.will_plot = self.parameters["Plot"]
        
        self.rh = None
        self.rl = None
        self.sh = None
        self.sl = None
        self.atr_val = None
        
        self.traded_today = False

    def before_market_opens(self):
        """
        Calculate the 4 Magic Lines + ATR
        """
        ticker = self.parameters["Ticker"]
        self.traded_today = False
        
        bars = self.get_historical_prices(ticker, self.parameters["LookbackDays"], "day")
        if bars is None or len(bars.df) < 15:
            return
            
        df = bars.df.copy()
        
        # Calculate ATR for proximity buffer
        import pandas_ta as ta
        df.ta.atr(length=14, append=True)
        atr_col = [c for c in df.columns if 'ATR' in c]
        if atr_col:
            self.atr_val = df[atr_col[0]].iloc[-1]
        else:
            self.atr_val = df['close'].iloc[-1] * 0.02 # fallback to 2%
            
        yesterday = df.iloc[-1]
        
        # Range High & Low
        self.rh = yesterday['high']
        self.rl = yesterday['low']
        
        self.sh = None
        self.sl = None
        
        for i in range(len(df) - 2, 0, -1):
            curr_high = df['high'].iloc[i]
            prev_high = df['high'].iloc[i-1]
            next_high = df['high'].iloc[i+1]
            
            curr_low = df['low'].iloc[i]
            prev_low = df['low'].iloc[i-1]
            next_low = df['low'].iloc[i+1]
            
            if curr_high > prev_high and curr_high > next_high:
                if self.sh is None and curr_high > self.rh:
                    self.sh = curr_high
                    
            if curr_low < prev_low and curr_low < next_low:
                if self.sl is None and curr_low < self.rl:
                    self.sl = curr_low
                    
            if self.sh is not None and self.sl is not None:
                break
                
        if self.sh is None:
            self.sh = self.rh + self.atr_val
        if self.sl is None:
            self.sl = self.rl - self.atr_val
            
        self.log_message(f"[{self.get_datetime().date()}] Levels | SH: {self.sh:.2f} | RH: {self.rh:.2f} || RL: {self.rl:.2f} | SL: {self.sl:.2f} | ATR Buffer: {self.atr_val * self.parameters['AtrMultiplier']:.2f}")

    def on_trading_iteration(self):
        """
        Executes on 15m intervals.
        Expanded entry window: scan for sneaky candles between 09:45 and 11:00.
        """
        current_time = self.get_datetime()
        
        if self.traded_today:
            return
            
        ticker = self.parameters["Ticker"]
        
        # Fetch up to 120 minutes of 1-minute bars to safely resample the morning
        bars = self.get_historical_prices(ticker, 120, "minute")
        if bars is None or len(bars.df) < 15:
            return
            
        df_1m = bars.df.copy()
        df = df_1m.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        if len(df) < 2:
            return
            
        last_date = df.index[-1].date()
        today_bars = df[df.index.date == last_date]
            
        # Expanded window: Evaluate between bar 2 (09:45) and bar 6 (11:00)
        if len(today_bars) < 2 or len(today_bars) > 6:
            return
            
        # First candle boundaries (Discretionary cheat)
        c1 = today_bars.iloc[0]
        c1_high = c1['high']
        c1_low = c1['low']
        
        # The latest closed candle is our potential "sneaky candle"
        sneaky_c = today_bars.iloc[-1]
        
        self._evaluate_sneaky_candle(sneaky_c, c1_high, c1_low)
        
    def _evaluate_sneaky_candle(self, c, c1_high, c1_low):
        if self.rh is None or self.rl is None or self.atr_val is None:
            return
            
        buffer = self.atr_val * self.parameters["AtrMultiplier"]
        
        def tested_level(low, high, level):
            return low <= level + buffer and high >= level - buffer

        # Check Bottom Lines (Buy Side Force)
        tested_bottom = tested_level(c['low'], c['high'], self.rl) or \
                        tested_level(c['low'], c['high'], self.sl) or \
                        tested_level(c['low'], c['high'], c1_low)
                        
        # Check Top Lines (Sell Side Force)
        tested_top = tested_level(c['low'], c['high'], self.rh) or \
                     tested_level(c['low'], c['high'], self.sh) or \
                     tested_level(c['low'], c['high'], c1_high)
        
        # LONG SETUP
        if tested_bottom:
            is_green = c['close'] > c['open']
            body_size = abs(c['close'] - c['open'])
            lower_wick = min(c['close'], c['open']) - c['low']
            has_long_wick = lower_wick > body_size
            
            if is_green or has_long_wick:
                self.log_message("LONG SETUP: Sneaky Candle printed at support.")
                sl_price = c['low'] * 0.999
                # Target the opposite side of the range (C1 high or RH)
                tp_price = max(self.rh, c1_high)
                entry_trigger = c['high']
                
                self._place_trigger_order("buy", entry_trigger, sl_price, tp_price)
                self.traded_today = True
                return

        # SHORT SETUP
        if tested_top:
            is_red = c['close'] < c['open']
            body_size = abs(c['close'] - c['open'])
            upper_wick = c['high'] - max(c['close'], c['open'])
            has_long_wick = upper_wick > body_size
            
            if is_red or has_long_wick:
                self.log_message("SHORT SETUP: Sneaky Candle printed at resistance.")
                sl_price = c['high'] * 1.001
                # Target the opposite side of the range
                tp_price = min(self.rl, c1_low)
                entry_trigger = c['low']
                
                self._place_trigger_order("sell", entry_trigger, sl_price, tp_price)
                self.traded_today = True
                return

    def _place_trigger_order(self, side: str, trigger_price: float, sl_price: float, tp_price: float):
        # Position sizing
        risk_per_share = abs(trigger_price - sl_price)
        if risk_per_share <= 0:
            return
            
        portfolio_value = self.get_portfolio_value()
        risk_budget = portfolio_value * self.parameters["RiskPct"]
        quantity = int(risk_budget // risk_per_share)
        
        if quantity <= 0:
            return
            
        self.log_message(
            f"Placing Bracket STOP order for {side.upper()}. "
            f"Trigger at {trigger_price:.2f}, SL at {sl_price:.2f}, TP at {tp_price:.2f}"
        )
        
        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=quantity,
            side=side,
            stop_price=round(trigger_price, 2), # This makes it a Stop Order (entry trigger)
            take_profit_price=round(tp_price, 2),
            stop_loss_price=round(sl_price, 2),
            time_in_force="day" # Cancel at end of day if not triggered
        )
        
        self.submit_order(order)

    def on_strategy_end(self):
        if getattr(self, "will_plot", False) and getattr(self, "is_backtesting", False):
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'logs', 'charts',
                f"{self.parameters.get('Ticker', 'chart')}_sneaky_pivot_chart.html"
            )
            try:
                self.save_plot_html(output_path)
            except Exception as e:
                pass
        return super().on_strategy_end()


if __name__ == "__main__":
    backtesting_end = datetime.now()
    backtesting_start = backtesting_end - timedelta(days=180)
    budget = 10000
    
    print("Running Backtest for Sneaky Pivot Strategy...")
    SneakyPivotStrategy.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        budget=budget,
        parameters={"Ticker": Asset(symbol="PLTR", asset_type=Asset.AssetType.STOCK)}
    )
