"""
System 1: Long Trend High Momentum
To be in trending STOCKs that have big momentum.
This gets you into the high-fliers, the popular STOCKs, when the market is an uptrend.
We only want to trade when the market sentiment is in our favor and we are in very liquid STOCKs.
I like a lot of volume for long-term positions, because the volume may diminish over time
and you want to have a cushion so you can always get out with good liquidity.
In this system I want the trend of the STOCK to be up in a very simple way. 
 
From the book : Automated STOCK Trading Systems by Lawrence Bensdorp
"""

import pandas_ta as ta
import pandas as pd
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset
from lumibot.traders import Trader
import os
from dotenv import load_dotenv

import plotly.graph_objects as go



load_dotenv()

apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

ALPACA_CONFIG = {
    "API_KEY":apikey,
    "API_SECRET": apisecret,
    "PAPER": True,  # Set to True for paper trading, False for live trading
}


# TODO: Migrate rank on screener
def rank():
    """
    In case we have more setups than our position sizing allows,
    we rank by the highest rate of change over the last 200 trading days.
    This means the highest percentage price increase over the last 200 trading days.
    """
    pass

class LongTrendHighMomentum(Strategy):
    
    parameters = {
        "AvgDailyVolume": 50000000,
        "SmaLength":50,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "TrailStopLoss" : False, # True if you want to use a trail stop with no tp,
                              #False if you want to use a 2:1 tp:sl ratio
        "RiskRewardRatio" : 2 # Risk Reward Ratio for the trade
    }
        
    def initialize(self):
        self.sleeptime = "1D" # Execute strategy every day once
        
    def before_market_opens(self):
        # Get the data once before market opens to not waste time
        self.ticker_bars = self.get_historical_prices(self.parameters["Ticker"],self.parameters["SmaLength"],"day").df
        self.spy_bars = self.get_historical_prices("SPY",100,"day").df

        return super().before_market_opens()

    def check_if_tradeable(self):
        """
        Average daily dollar volume greater than $50 million over the last twenty days.
        Minimum price $5.00.
        """
        bars = self.ticker_bars.iloc[-20:]
        if bars is None:
            print("No data found! Please consider changing the ticker symbol.")
            return False
        if bars["volume"].mean() < self.parameters["AvgDailyVolume"]: # Average daily dollar volume greater than $50 million
            return False
        if bars["close"].min() < 5: # Minimum price $5.00.
            return False
      #  print(f"{self.parameters["Ticker"]} meets the minimum requirements to be traded using the Long Trend High Momentum strategy.")
        return True

    def check_ta(self):
        """
        Close of the SPY is above the 100-day simple moving average (SMA).
        This indicates a trend in the overall index.
        The close of the 25-day simple moving average is above the close of the 50-day simple moving average.
        """
        if self.spy_bars is None:
           # print("No data found! Please consider changing the ticker symbol.")
            return False
        sma_100 = ta.sma(self.spy_bars["close"], length=100)
        if self.spy_bars["close"].iloc[-1] < sma_100.iloc[-1]: # Close of the SPY is above the 100-day simple moving average (SMA).
            return False
        sma_25 = ta.sma(self.ticker_bars["close"], length=25)
        sma_50 = ta.sma(self.ticker_bars["close"], length=50)
        if sma_25 is None or sma_50 is None:
            #print("No data found! Please consider changing the ticker symbol.")
            return False
        if sma_25.iloc[-1] < sma_50.iloc[-1]: # The close of the 25-day simple moving average is above the close of the 50-day simple moving average.
            return False
      #  print(f"{self.parameters["Ticker"]} meets the technical analysis requirements to be traded using the Long Trend High Momentum strategy.")
        return True
    
    def on_trading_iteration(self):

        if not (self.check_if_tradeable() and self.check_ta()):
            return
                
       # order_size = self.position_sizing()
        order_size = int((self.cash / self.ticker_bars["close"].iloc[-1]) * 0.02)
        if order_size == 0:
           # print("Position size is 0. No order will be placed.")
            return
        
        bars = self.ticker_bars.iloc[-20:]
        atr = (ta.atr(bars["high"],bars["low"],bars["close"]).iloc[-1] * 5)
        sl = bars["close"].iloc[-1] - atr
        tp = bars["close"].iloc[-1] + atr * self.parameters["RiskRewardRatio"]

              
#       print(f"{self.parameters["Ticker"]} meets all the requirements to be traded using the Long Trend High Momentum strategy.")
        # Place an oco order
        if self.parameters["TrailStopLoss"] :
            order = self.create_order(
                asset=self.parameters["Ticker"],
                quantity=order_size,
                side="buy",
                stop_loss_price=sl,
                trail_percent=0.25,
                position_filled=True,
                type="bracket",
                time_in_force="gtc")
        else:
            order = self.create_order(
                asset=self.parameters["Ticker"],
                quantity=order_size,
                side="buy",
                take_profit_price=tp,
                stop_loss_price=sl,
                position_filled=True,
                type="bracket",
                time_in_force="gtc")
            
        self.submit_order(order)
              
       

def run_live():
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = LongTrendHighMomentum(broker=broker,
                                         parameters={"Ticker": Asset(symbol="NIO",
                                                                    asset_type=Asset.AssetType.STOCK)
                                                     }
                                        )

        # Run the strategy live
        trader.add_strategy(strategy)
        trader.run_all()

def run_backtest():
        # Define parameters
        backtesting_start = datetime(2023, 10, 23)
        backtesting_end = datetime(2024, 10, 23)
        budget = 10000
        # Run the backtest    
        LongTrendHighMomentum.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()