"""
BEST MACD Trading Strategy [86% Win Rate]
Based on TradingLab's video breakdown.

Key Rules:
1. 200 EMA for trend direction.
2. Price must be near a Key Support/Resistance Level.
3. MACD (12, 26, 9) crossover.
    - Long: Bullish cross below zero line.
    - Short: Bearish cross above zero line.
4. Stop Loss: Placed strictly below/above the 200 EMA.
5. Take Profit: Fixed 1.5R.
"""

from typing import Optional, Dict, Any
import pandas as pd
import pandas_ta as ta
from lumibot.entities import Asset

from strategies.base_key_levels_strategy import BaseKeyLevelsStrategy

class MACDTradingStrategy(BaseKeyLevelsStrategy):
    parameters = {
        **BaseKeyLevelsStrategy.parameters,
        "Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK),
        "TradingStyle": "day_trading",
        "RiskPct": 0.02,
        "ENTRY_THRESHOLD": 0.005,  # 0.5% tolerance for Support/Resistance proximity
        "RiskRewardRatio": 1.5,
        "Plot": True,
        "Timeframe": "15M",  # Usually a 15-minute chart
    }

    def get_strategy_name(self) -> str:
        return "MACD_TradingLab_Strategy"

    def on_strategy_start(self):
        self.sleeptime = self.parameters.get("Timeframe", "15M")
        self.current_df = None

    def on_trading_iteration(self):
        # Fetch data to calculate 200 EMA and MACD
        ticker = self.parameters["Ticker"]
        
        # Determine timespan string based on sleeptime
        sleeptime_str = self.sleeptime.upper()
        if "M" in sleeptime_str:
            mins = int(sleeptime_str.replace("M", ""))
            bars = self.get_historical_prices(ticker, 250, "minute", timespan_multiplier=mins)
        else:
            bars = self.get_historical_prices(ticker, 250, "day")

        if bars is not None and len(bars.df) >= 200:
            df = bars.df.copy()
            df.ta.ema(length=200, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            self.current_df = df
        else:
            self.current_df = None

        # Call parent to handle the actual entry and exit logic using hooks
        super().on_trading_iteration()

    def get_entry_signal(
        self, 
        current_price: float,
        support_levels: pd.DataFrame,
        resistance_levels: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        if self.current_df is None or len(self.current_df) < 2:
            return None

        df = self.current_df
        
        # Get column names (pandas_ta dynamically generates them)
        ema_col = "EMA_200"
        macd_col = "MACD_12_26_9"
        macds_col = "MACDs_12_26_9"
        
        if ema_col not in df.columns or macd_col not in df.columns or macds_col not in df.columns:
            return None

        ema_200 = df[ema_col].iloc[-1]
        
        macd_curr = df[macd_col].iloc[-1]
        macds_curr = df[macds_col].iloc[-1]
        macd_prev = df[macd_col].iloc[-2]
        macds_prev = df[macds_col].iloc[-2]

        # 1. Support & Resistance Check
        near_support = False
        closest_supp = current_price
        for _, row in support_levels.iterrows():
            if abs(current_price - row['price']) / row['price'] <= self.entry_threshold:
                near_support = True
                closest_supp = row['price']
                break
                
        near_resistance = False
        closest_res = current_price
        for _, row in resistance_levels.iterrows():
            if abs(current_price - row['price']) / row['price'] <= self.entry_threshold:
                near_resistance = True
                closest_res = row['price']
                break

        rr_ratio = self.parameters.get("RiskRewardRatio", 1.5)

        # 2. Long Setup
        if current_price > ema_200 and near_support:
            # Bullish MACD cross strictly below zero line
            if macd_prev < macds_prev and macd_curr > macds_curr and macd_curr < 0:
                # Stop Loss below 200 EMA
                stop_loss = ema_200 * 0.999
                
                # If EMA is too far, risk might be huge. Ensure SL makes sense.
                if stop_loss >= current_price:
                    stop_loss = current_price * 0.99  # Fallback 1% stop loss
                    
                risk = current_price - stop_loss
                take_profit = current_price + (risk * rr_ratio)
                
                return {
                    'trade_type': 'BUY',
                    'entry_price': current_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'support_level': closest_supp,
                    'resistance_level': take_profit
                }

        # 3. Short Setup
        if current_price < ema_200 and near_resistance:
            # Bearish MACD cross strictly above zero line
            if macd_prev > macds_prev and macd_curr < macds_curr and macd_curr > 0:
                # Stop Loss above 200 EMA
                stop_loss = ema_200 * 1.001
                
                if stop_loss <= current_price:
                    stop_loss = current_price * 1.01  # Fallback 1% stop loss
                    
                risk = stop_loss - current_price
                take_profit = current_price - (risk * rr_ratio)
                
                return {
                    'trade_type': 'SELL',
                    'entry_price': current_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'support_level': take_profit,
                    'resistance_level': closest_res
                }

        return None

    def get_exit_signal(
        self,
        current_price: float,
        position,
        entry_support: float,
        target_resistance: float
    ) -> Optional[str]:
        # Rely on TP / SL bracket orders
        return None
