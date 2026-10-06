import os
import sys
import pandas as pd
import pandas_ta as ta
from datetime import datetime, time, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strat_baseplate import StrategyBaseplate

class PBInvestingVWAPEMA(StrategyBaseplate):
    """
    PBInvesting Day Trading Strategy: VWAP + 8 EMA Confluence
    As defined in "The ONLY 2 Indicators I use to make $4351/Day Trading"
    
    Timeframe: 5-minute chart
    Indicators:
      - VWAP (Standard session)
      - 8 EMA
      - Pre-Market High (PMH) & Pre-Market Low (PML)
    """
    
    parameters = {
        "Ticker": "AAPL",
        "Plot": True,
        "TradingStyle": "day_trading",
        "RiskPct": 0.02,
        # Strategy specific params
        "EMA_Length": 8,
        "Setup_Type": "vwap_retest", # "pm_retest", "vwap_retest", "ema_scalp"
    }

    def initialize(self):
        super().initialize()
        self.sleeptime = "5M"
        self.pmh = None
        self.pml = None
        self.hod = None
        self.lod = None
        self.state = "flat"  # flat, long, short
        
        # State variables for trailing stops
        self.entry_setup = None 
        self.vwap_at_entry = None
        self.retest_candle_low = None
        self.retest_candle_high = None

    def _calculate_indicators(self, df):
        # Calculate 8 EMA
        df['ema8'] = ta.ema(df['close'], length=self.parameters["EMA_Length"])
        
        # Calculate session VWAP
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        return df

    def _update_levels(self, df):
        """Update PMH, PML, HOD, LOD based on 5m data."""
        if df.empty:
            return
            
        current_date = df.index[-1].date()
        today_data = df[df.index.date == current_date]
        
        if today_data.empty:
            return
            
        # Separate Pre-market (before 09:30) and Regular Trading Hours
        # Assuming index is timezone-aware or local market time
        pm_data = today_data[today_data.index.time < time(9, 30)]
        rth_data = today_data[today_data.index.time >= time(9, 30)]
        
        if not pm_data.empty:
            self.pmh = pm_data['high'].max()
            self.pml = pm_data['low'].min()
            
        if not rth_data.empty:
            self.hod = rth_data['high'].max()
            self.lod = rth_data['low'].min()

    def setup(self):
        ticker = self.parameters.get("Ticker")
        bars = self.get_historical_prices(ticker, 100, "minute") 
        if bars is None or len(bars.df) < 20:
            return False
            
        df = bars.df.copy()
        
        # Ensure we're using 5-min resolution 
        if len(df) > 500:
            df = df.resample('5T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            
        df = self._calculate_indicators(df)
        self._update_levels(df)
        
        self.latest_data = df.iloc[-1]
        self.prev_data = df.iloc[-2]
        return True

    def get_entry_signal(self):
        """
        Determine entry signals based on the 3 setups.
        Returns: 1 for long, -1 for short, 0 for none.
        """
        if self.pmh is None or self.pml is None:
            return 0
            
        close = self.latest_data['close']
        vwap = self.latest_data['vwap']
        ema = self.latest_data['ema8']
        
        # Core Directional Bias Rule
        can_long = close > vwap
        can_short = close < vwap
        
        if not can_long and not can_short:
            return 0

        # Setup 2: VWAP Retest (Simplified)
        if self.parameters["Setup_Type"] == "vwap_retest":
            if can_long:
                pulled_back = self.prev_data['low'] <= self.prev_data['vwap'] * 1.002
                bounce_confirmed = close > ema and self.latest_data['open'] < close
                if pulled_back and bounce_confirmed:
                    self.entry_setup = "vwap_retest_long"
                    self.vwap_at_entry = vwap
                    return 1
            elif can_short:
                rallied_up = self.prev_data['high'] >= self.prev_data['vwap'] * 0.998
                rejection_confirmed = close < ema and self.latest_data['open'] > close
                if rallied_up and rejection_confirmed:
                    self.entry_setup = "vwap_retest_short"
                    self.vwap_at_entry = vwap
                    return -1

        # Setup 1: PM Level Confluence
        elif self.parameters["Setup_Type"] == "pm_retest":
            if can_long and self.prev_data['high'] > self.pmh and self.latest_data['low'] <= self.pmh * 1.001:
                if close > self.latest_data['open']:
                    self.entry_setup = "pm_retest_long"
                    self.retest_candle_low = self.latest_data['low']
                    return 1
            elif can_short and self.prev_data['low'] < self.pml and self.latest_data['high'] >= self.pml * 0.999:
                if close < self.latest_data['open']:
                    self.entry_setup = "pm_retest_short"
                    self.retest_candle_high = self.latest_data['high']
                    return -1
                    
        # Setup 3: 8 EMA Scalp Retest
        elif self.parameters["Setup_Type"] == "ema_scalp":
            if can_long and self.prev_data['low'] <= ema * 1.001 and close > ema:
                self.entry_setup = "ema_scalp_long"
                return 1
            elif can_short and self.prev_data['high'] >= ema * 0.999 and close < ema:
                self.entry_setup = "ema_scalp_short"
                return -1

        return 0

    def get_position_sizing(self):
        signal = self.get_entry_signal()
        if signal == 0:
            return 0
            
        ticker = self.parameters.get("Ticker")
        price = self.latest_data['close']
        cash = self.cash * self.risk_percent
        qty = int(cash / price)
        
        self.state = "long" if signal == 1 else "short"
        return qty if signal == 1 else -qty

    def get_stop_loss(self, entry: float):
        if "vwap_retest" in str(self.entry_setup):
            if self.state == "long":
                return self.vwap_at_entry * 0.998
            else:
                return self.vwap_at_entry * 1.002
        elif "pm_retest" in str(self.entry_setup):
            if self.state == "long":
                return min(self.retest_candle_low, self.pmh * 0.998)
            else:
                return max(self.retest_candle_high, self.pml * 1.002)
        elif "ema_scalp" in str(self.entry_setup):
            if self.state == "long":
                return entry * 0.995 
            else:
                return entry * 1.005
                
        return None

    def get_take_profit(self, entry: float):
        if self.state == "long" and self.hod and self.hod > entry:
            return self.hod
        elif self.state == "short" and self.lod and self.lod < entry:
            return self.lod
        return None

    def on_trading_iteration(self):
        super().on_trading_iteration()
        
        # Manage Trailing 8 EMA Exit
        ticker = self.parameters.get("Ticker")
        pos = self.get_position(ticker)
        if pos:
            close = self.latest_data['close']
            ema = self.latest_data['ema8']
            
            # Long: Exit immediately when 5-min candle closes below 8 EMA
            if pos.quantity > 0 and close < ema:
                self.sell_all()
                self.state = "flat"
                
            # Short: Exit immediately when 5-min candle closes above 8 EMA
            elif pos.quantity < 0 and close > ema:
                self.sell_all()
                self.state = "flat"
