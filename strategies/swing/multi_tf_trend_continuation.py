import sys
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

# Add parent directory (strategies) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lumibot.tools.helpers
lumibot.tools.helpers.print_progress_bar = lambda *args, **kwargs: None
from plot_mixin import PlottableStrategyMixin


class MultiTimeframeTrendContinuationStrategy(Strategy, PlottableStrategyMixin):
    """
    Lance Breitstein's Trend-Trading Framework

    Primary Setup: Multi-Timeframe Trend Breakout / Continuation
    - Alignment across Intraday, Daily, and Weekly trends.
    - Consolidation beneath major resistance with higher lows.
    - Above VWAP and unaffected benchmark price (e.g., prior day close).
    - Entry on breakout with volume surge (catalyst).
    - Trailing stop using prior bar lows.

    Secondary Setup: Counter-Trend Capitulation Bounce
    - Extreme parabolic flush on massive volume.
    - First candle to break prior bar high triggers long entry.
    - Hard stop at the flush low.
    """

    parameters = {
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "Plot": True,
        
        # Risk / Size
        "RiskPct": 0.02, # 2% per trade
        "MaxPositions": 2,

        # Trend & Volume parameters
        "VolumeMultiplier": 3.0, # Volume must be 3x average for catalyst/capitulation
        "CapitulationDropPct": 0.05, # Intraday drop required for capitulation setup
        
        # Timeframes
        "IntradayLookback": 60, # Bars for intraday consolidation / VWAP
        "DailyLookback": 50, # Bars for daily trend
        "WeeklyLookback": 50, # Bars for weekly trend
    }

    def initialize(self):
        # 5-minute bars for intraday execution
        self.sleeptime = "5M"
        self.will_plot = self.parameters["Plot"]
        
        self.daily_trend = 0
        self.weekly_trend = 0
        self.yesterday_close = None
        
        # Tracking states
        self.active_trades = {}
        
    def before_market_opens(self):
        """
        Assess Daily and Weekly trends, and anchor the unaffected reference price.
        """
        ticker = self.parameters["Ticker"]
        
        # 1. Fetch Daily Data
        daily_bars = self.get_historical_prices(ticker, self.parameters["DailyLookback"], "day")
        if daily_bars is not None and len(daily_bars.df) > 10:
            df_d = daily_bars.df.copy()
            df_d.ta.ema(length=20, append=True)
            df_d.ta.ema(length=50, append=True)
            
            # Simple trend check: EMA20 > EMA50 and Price > EMA20
            last_d = df_d.iloc[-1]
            if last_d['close'] > last_d['EMA_20'] and last_d['EMA_20'] > last_d['EMA_50']:
                self.daily_trend = 1 # Uptrend
            elif last_d['close'] < last_d['EMA_20'] and last_d['EMA_20'] < last_d['EMA_50']:
                self.daily_trend = -1 # Downtrend
            else:
                self.daily_trend = 0
                
            self.yesterday_close = last_d['close']
            
        # 2. Fetch Weekly Data (Approximated by slicing daily or direct if available)
        # We will use simple daily proxy for weekly (EMA 100 vs EMA 200) for simplicity in this base
        if daily_bars is not None and len(daily_bars.df) > 50:
            df_w = daily_bars.df.copy()
            df_w.ta.ema(length=20, append=True) # Weekly 20 approx
            # Real weekly would require weekly bars, but we use higher daily EMAs
            df_w.ta.ema(length=50, append=True) 
            last_w = df_w.iloc[-1]
            if last_w['close'] > last_w['EMA_50']:
                self.weekly_trend = 1
            elif last_w['close'] < last_w['EMA_50']:
                self.weekly_trend = -1
            else:
                self.weekly_trend = 0
                
        self.log_message(f"[{self.get_datetime().date()}] Daily Trend: {self.daily_trend}, Weekly Trend: {self.weekly_trend}")

    def on_trading_iteration(self):
        ticker = self.parameters["Ticker"]
        
        # Fetch intraday data (e.g. 5-min bars)
        bars = self.get_historical_prices(ticker, self.parameters["IntradayLookback"], "minute")
        if bars is None or len(bars.df) < 20:
            return
            
        df = bars.df.copy()
        
        # Calculate VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume moving average
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        
        current_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        current_price = current_bar['close']
        
        # Manage active trailing stops
        self._manage_active_trades(df, current_price)
        
        # If we have reached max positions, skip new entries
        if len(self.active_trades) >= self.parameters["MaxPositions"]:
            return

        # ==========================================
        # Primary Setup: Trend Breakout
        # ==========================================
        # Check alignment
        if self.daily_trend == 1 and self.weekly_trend == 1:
            # Anchor levels check
            above_vwap = current_price > current_bar['vwap']
            above_ref = self.yesterday_close is not None and current_price > self.yesterday_close
            
            # Consolidation resistance (e.g. recent high over last 10 bars)
            recent_high = df['high'].iloc[-12:-2].max()
            is_breakout = prev_bar['close'] <= recent_high and current_price > recent_high
            
            # Volume surge
            vol_surge = current_bar['volume'] > (self.parameters["VolumeMultiplier"] * prev_bar['vol_ma'])
            
            if above_vwap and above_ref and is_breakout and vol_surge:
                self.log_message("PRIMARY SETUP TRIGGERED: Multi-TF Trend Breakout")
                # Stop loss: below the breakout level or VWAP
                sl_price = min(recent_high, current_bar['vwap'])
                
                self._execute_entry(
                    side="buy", 
                    price=current_price, 
                    sl_price=sl_price, 
                    setup_name="Primary_Breakout"
                )
                return # Skip secondary check if primary executed

        # ==========================================
        # Secondary Setup: Capitulation Bounce
        # ==========================================
        # Look for extreme parabolic flush (drop over recent bars)
        recent_high_intraday = df['high'].max()
        flush_drop = (recent_high_intraday - current_price) / recent_high_intraday
        
        vol_surge_cap = current_bar['volume'] > (self.parameters["VolumeMultiplier"] * prev_bar['vol_ma'])
        
        if flush_drop > self.parameters["CapitulationDropPct"] and vol_surge_cap:
            # Wait for first candle that breaks prior bar high
            reversal_trigger = current_price > prev_bar['high']
            
            if reversal_trigger:
                self.log_message("SECONDARY SETUP TRIGGERED: Capitulation Bounce")
                # Stop loss at the low of the flush
                flush_low = min(current_bar['low'], prev_bar['low'])
                
                self._execute_entry(
                    side="buy", 
                    price=current_price, 
                    sl_price=flush_low, 
                    setup_name="Secondary_Capitulation"
                )

    def _execute_entry(self, side: str, price: float, sl_price: float, setup_name: str):
        risk_per_share = abs(price - sl_price)
        if risk_per_share <= 0:
            return
            
        # Position sizing based on risk
        portfolio_value = self.get_portfolio_value()
        risk_budget = portfolio_value * self.parameters["RiskPct"]
        quantity = int(risk_budget // risk_per_share)
        
        if quantity <= 0:
            self.log_message(f"Quantity 0 for {setup_name} (Risk ${risk_per_share:.2f})")
            return
            
        order = self.create_order(
            asset=self.parameters["Ticker"],
            quantity=quantity,
            side=side
        )
        
        self.submit_order(order)
        self.active_trades[order.identifier] = {
            "setup": setup_name,
            "side": side,
            "entry_price": price,
            "sl_price": sl_price,
            "status": "pending",
            "quantity": quantity
        }
        
    def _manage_active_trades(self, df: pd.DataFrame, current_price: float):
        """
        Systematic Trailing Stop: Prior Bar Rules
        Exit manually if current price breaks the prior bar low (in uptrend).
        """
        if len(df) < 2:
            return
            
        prev_bar_low = df.iloc[-2]['low']
        prev_bar_high = df.iloc[-2]['high']
        
        orders_to_close = []
        for order_id, trade in self.active_trades.items():
            if trade["status"] != "open":
                continue
                
            side = trade["side"]
            sl_price = trade["sl_price"]
            
            # Trail stop logic
            if side == "buy":
                # Trail stop behind prior bar low
                new_sl = max(sl_price, prev_bar_low)
                trade["sl_price"] = new_sl
                
                if current_price < new_sl:
                    orders_to_close.append((order_id, trade, "Trailing Stop / Prior Bar Low"))
            elif side == "sell":
                new_sl = min(sl_price, prev_bar_high)
                trade["sl_price"] = new_sl
                
                if current_price > new_sl:
                    orders_to_close.append((order_id, trade, "Trailing Stop / Prior Bar High"))
                    
        for order_id, trade, reason in orders_to_close:
            self.log_message(f"Exiting Trade ({trade['setup']}) - Reason: {reason}")
            # Create closing order
            close_side = "sell" if trade["side"] == "buy" else "buy"
            close_order = self.create_order(
                asset=self.parameters["Ticker"],
                quantity=trade["quantity"],
                side=close_side
            )
            self.submit_order(close_order)
            
            # Remove from active trades tracking (or mark as closing)
            del self.active_trades[order_id]

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """
        Update active trades status when orders fill.
        """
        # If it's an entry order
        if order.identifier in self.active_trades:
            trade = self.active_trades[order.identifier]
            trade["status"] = "open"
            trade["entry_price"] = price # Update actual fill
            self.log_message(f"[{self.get_datetime()}] Entered {trade['setup']} - {trade['side']} {quantity} @ {price:.2f}. Initial SL: {trade['sl_price']:.2f}")

    def on_strategy_end(self):
        if getattr(self, "will_plot", False) and getattr(self, "is_backtesting", False):
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'logs', 'charts',
                f"{self.parameters.get('Ticker', 'chart')}_lance_chart.html"
            )
            try:
                self.save_plot_html(output_path)
            except Exception as e:
                pass
        return super().on_strategy_end()


if __name__ == "__main__":
    backtesting_start = datetime(2023, 1, 1)
    backtesting_end = datetime(2023, 6, 1)
    budget = 10000
    
    print("Running Backtest for Multi-Timeframe Trend Continuation Strategy...")
    MultiTimeframeTrendContinuationStrategy.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        budget=budget,
        parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
    )
