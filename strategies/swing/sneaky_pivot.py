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
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": True,
        
        "RiskPct": 0.02, # 2% risk per trade
        "ProximityPct": 0.003, # 0.3% proximity to level to consider it 'tested'
        "LookbackDays": 20, # Days to look back for swing highs/lows
    }

    def initialize(self):
        # We need to assess 15m candles
        self.sleeptime = "15M"
        self.will_plot = self.parameters["Plot"]
        
        self.rh = None
        self.rl = None
        self.sh = None
        self.sl = None
        
        self.traded_today = False

    def before_market_opens(self):
        """
        Calculate the 4 Magic Lines: RH, RL, SH, SL
        """
        ticker = self.parameters["Ticker"]
        self.traded_today = False
        
        # Fetch daily data for swing levels
        bars = self.get_historical_prices(ticker, self.parameters["LookbackDays"], "day")
        if bars is None or len(bars.df) < 5:
            return
            
        df = bars.df.copy()
        yesterday = df.iloc[-1]
        
        # Range High & Low
        self.rh = yesterday['high']
        self.rl = yesterday['low']
        
        # Find Swing High (most recent local maximum > RH)
        # Find Swing Low (most recent local minimum < RL)
        self.sh = None
        self.sl = None
        
        # Simple swing point detection over the lookback window (excluding yesterday)
        for i in range(len(df) - 2, 0, -1):
            curr_high = df['high'].iloc[i]
            prev_high = df['high'].iloc[i-1]
            next_high = df['high'].iloc[i+1]
            
            curr_low = df['low'].iloc[i]
            prev_low = df['low'].iloc[i-1]
            next_low = df['low'].iloc[i+1]
            
            # Is it a swing high?
            if curr_high > prev_high and curr_high > next_high:
                if self.sh is None and curr_high > self.rh:
                    self.sh = curr_high
                    
            # Is it a swing low?
            if curr_low < prev_low and curr_low < next_low:
                if self.sl is None and curr_low < self.rl:
                    self.sl = curr_low
                    
            if self.sh is not None and self.sl is not None:
                break
                
        # Fallbacks if no swing high/low found
        if self.sh is None:
            self.sh = self.rh * 1.05
        if self.sl is None:
            self.sl = self.rl * 0.95
            
        self.log_message(f"[{self.get_datetime().date()}] Levels | SH: {self.sh:.2f} | RH: {self.rh:.2f} || RL: {self.rl:.2f} | SL: {self.sl:.2f}")

    def on_trading_iteration(self):
        """
        Executes on 15m intervals.
        We only look for entries exactly at 10:00 AM (market time).
        Candle 1: 9:30 - 9:45
        Candle 2: 9:45 - 10:00
        At 10:00 AM, both are closed, we evaluate and place conditional orders.
        """
        current_time = self.get_datetime()
        
        # Assuming market opens at 09:30 AM EST. 
        # In Lumibot backtesting, timezones can vary, but typically starts at 09:30.
        # Check if it is the 10:00 AM iteration.
        if self.traded_today:
            return
            
        ticker = self.parameters["Ticker"]
        # Fetch last 30 minutes of 1-minute bars
        bars = self.get_historical_prices(ticker, 30, "minute")
        if bars is None or len(bars.df) < 15:
            return
            
        # Resample to 15-minute candles
        df_1m = bars.df.copy()
        df = df_1m.resample("15min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        # Verify these two bars are from today
        if len(df) < 2 or df.index[-1].date() != current_time.date() or df.index[-2].date() != current_time.date():
            return
            
        today_bars = df[df.index.date == current_time.date()]
        if len(today_bars) != 2:
            return
            
        c1 = df.iloc[0] # 9:30 - 9:45
        c2 = df.iloc[1] # 9:45 - 10:00
        
        self._evaluate_setups(c1, c2)
        self.traded_today = True # Prevent multiple evaluations per day
        
    def _evaluate_setups(self, c1, c2):
        if self.rh is None or self.rl is None:
            return
            
        proximity_pct = self.parameters["ProximityPct"]
        
        # Helper to check if a level was tested
        def tested_level(low, high, level):
            return low <= level * (1 + proximity_pct) and high >= level * (1 - proximity_pct)
            
        c1_low_tested_rl = tested_level(c1['low'], c1['high'], self.rl)
        c1_low_tested_sl = tested_level(c1['low'], c1['high'], self.sl)
        
        c1_high_tested_rh = tested_level(c1['low'], c1['high'], self.rh)
        c1_high_tested_sh = tested_level(c1['low'], c1['high'], self.sh)
        
        # ----------------------------------------------------
        # LONG SETUP (Buying at Lows)
        # ----------------------------------------------------
        if c1_low_tested_rl or c1_low_tested_sl:
            # Check Candle 2 (Sneaky Candle) stability
            # Stabilizing: Green candle OR long lower wick
            is_green = c2['close'] > c2['open']
            body_size = abs(c2['close'] - c2['open'])
            lower_wick = min(c2['close'], c2['open']) - c2['low']
            has_long_wick = lower_wick > body_size
            
            if is_green or has_long_wick:
                self.log_message("LONG SETUP: Pushed into Buy Zone, Sneaky Candle printed.")
                
                # Stop Loss: Just below the tested support zone (low of C1 or C2)
                sl_price = min(c1['low'], c2['low']) * 0.999
                
                # Take Profit: Target opposite side of range (RH)
                tp_price = self.rh
                
                # Trigger Entry: Stop order exactly when price crosses above C2 high
                entry_trigger = c2['high']
                
                self._place_trigger_order("buy", entry_trigger, sl_price, tp_price)
                return

        # ----------------------------------------------------
        # SHORT SETUP (Selling at Highs)
        # ----------------------------------------------------
        if c1_high_tested_rh or c1_high_tested_sh:
            # Check Candle 2 (Sneaky Candle) stability
            # Stabilizing: Red candle OR long upper wick
            is_red = c2['close'] < c2['open']
            body_size = abs(c2['close'] - c2['open'])
            upper_wick = c2['high'] - max(c2['close'], c2['open'])
            has_long_wick = upper_wick > body_size
            
            if is_red or has_long_wick:
                self.log_message("SHORT SETUP: Pushed into Sell Zone, Sneaky Candle printed.")
                
                # Stop Loss: Just above the tested resistance zone
                sl_price = max(c1['high'], c2['high']) * 1.001
                
                # Take Profit: Target opposite side of range (RL)
                tp_price = self.rl
                
                # Trigger Entry: Stop order exactly when price crosses below C2 low
                entry_trigger = c2['low']
                
                self._place_trigger_order("sell", entry_trigger, sl_price, tp_price)
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
