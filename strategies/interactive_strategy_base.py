"""
Interactive Base Key Levels Strategy

Inherits from BaseKeyLevelsStrategy.
Adds functionality to:
1. Visualize the chart and key levels at each recalculation step (daily/weekly).
2. Pause execution and wait for user input (Enter) before continuing.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_key_levels_strategy import BaseKeyLevelsStrategy
from key_levels import KeyLevels
try:
    from plots.key_levels_plot import plot_key_levels_plotly
except ImportError:
    plot_key_levels_plotly = None

class InteractiveKeyLevelsStrategy(BaseKeyLevelsStrategy):
    """
    Interactive version of BaseKeyLevelsStrategy.
    
    Overrides _load_key_levels to:
    1. Calculate levels (via super).
    2. Fetch data for visualization.
    3. specific Plot the chart with levels.
    4. Print levels to console.
    5. Pause for user input.
    """
    
    def initialize(self):
        super().initialize()
        self._last_plotted_date = None
        
    def _load_key_levels(self, as_of_datetime: datetime):
        """
        Load key levels and then pause for interaction.
        """
        # 1. Standard level calculation
        super()._load_key_levels(as_of_datetime)
        
        # Check if we successfully loaded levels and it's a new date
        current_date_str = str(as_of_datetime.date())
        
        # Only interact if we have levels and haven't plotted this date yet
        if self.merged_levels is not None and not self.merged_levels.empty:
            if self._last_plotted_date != current_date_str:
                self._visualize_and_pause(as_of_datetime)
                self._last_plotted_date = current_date_str
    
    def _visualize_and_pause(self, as_of_datetime: datetime):
        """
        Visualize the current state and wait for user input.
        """
        ticker = self.parameters["Ticker"].symbol
        print(f"\n{'='*60}")
        print(f"INTERACTIVE MODE - {ticker} - {as_of_datetime.date()}")
        print(f"{'='*60}")
        
        # --- 1. CONSOLE OUTPUT ---
        
        # Print Support Levels
        print("\n🟢 SUPPORT LEVELS:")
        if self.support_levels is not None and not self.support_levels.empty:
            print(self.support_levels[['level_price', 'importance', 'touch_count']].to_string(index=False))
        else:
            print("None")
            
        # Print Resistance Levels
        print("\n🔴 RESISTANCE LEVELS:")
        if self.resistance_levels is not None and not self.resistance_levels.empty:
            print(self.resistance_levels[['level_price', 'importance', 'touch_count']].to_string(index=False))
        else:
            print("None")
            
        # --- 2. CHART VISUALIZATION ---
        
        print("\nGenerating chart...")
        try:
            # We need to fetch data for plotting since the base class doesn't expose the raw data
            # We use the KeyLevels class to fetch it exactly as it was used for calculation
            kl = KeyLevels(
                ticker=ticker,
                use_alpaca=False, # Use Yahoo to match backtest data source usually
                as_of_date=as_of_datetime
            )
            
            # Fetch data for all timeframes used in strategy
            # default: ['1d', '4h', '1h', '15m', '5m']
            timeframes = self.timeframes
            
            # We need to populate kl.timeframe_data
            for tf in timeframes:
                kl.fetch_data(interval=tf, lookback_days=150 if tf == '1d' else 30)
                # Note: We don't need to re-calculate pivots/clusters, just need the raw DF for plotting
                # But plot_key_levels_plotly expects 'timeframe_data' populated with DFs
                # KeyLevels.fetch_data returns the DF but doesn't store it in timeframe_data automatically unless called via find_key_levels...
                # So we manually store it
                # actually KeyLevels.fetch_data returns the DF.
                
            # Let's just use find_all_key_levels to do the work and populate everything correctly
            # It's a bit redundant calculation-wise but ensures 100% consistency for the plot
            kl.find_all_key_levels(timeframes=timeframes)
            
            # Now we can plot
            if plot_key_levels_plotly:
                # Use the plotly plotter from the project
                fig = plot_key_levels_plotly(
                    ticker=ticker,
                    timeframe_data=kl.timeframe_data,
                    all_levels_df=kl.all_levels_df,
                    merged_df=self.merged_levels, # Use the ACTUAL strategy levels
                    fib_df=None, # Optional: Add fib support if strategy has it
                    show=True,
                    save_html=False
                )
            else:
                print("Warning: Plotly plotter not available.")
                
        except Exception as e:
            print(f"Error generating interactive chart: {e}")
            import traceback
            traceback.print_exc()

        # --- 3. PAUSE ---
        print(f"\n[PAUSED] Strategy execution paused at {as_of_datetime}.")
        input("Press Enter to continue to next period...")
        print("Resuming...")

