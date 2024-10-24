"""
System 1: Long Trend High Momentum
To be in trending stocks that have big momentum.
This gets you into the high-fliers, the popular stocks, when the market is an uptrend.
We only want to trade when the market sentiment is in our favor and we are in very liquid stocks.
I like a lot of volume for long-term positions, because the volume may diminish over time
and you want to have a cushion so you can always get out with good liquidity.
In this system I want the trend of the stock to be up in a very simple way. 
 
From the book : Automated Stock Trading Systems by Lawrence Bensdorp
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
    
    #Define variables TODO: MAKE THEM DYNAMIC
    MIN_VOLUME = 50000000
    LAST_DAYS = 20
    stock  = Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK)
    total_positions =0 
    
    def initialize(self):
        self.sleeptime = "1D" # Execute strategy every day once

    def check_if_tradeable(self):
        """
        Average daily dollar volume greater than $50 million over the last twenty days.
        Minimum price $5.00.
        """
        ticker_bars = self.get_historical_prices(self.stock.symbol,self.LAST_DAYS,"day").df
        if ticker_bars is None:
            print("No data found! Please consider changing the ticker symbol.")
            return False
        if ticker_bars["volume"].mean() < self.MIN_VOLUME: # Average daily dollar volume greater than $50 million
            return False
        if ticker_bars["close"].min() < 5: # Minimum price $5.00.
            return False
      #  print(f"{self.stock} meets the minimum requirements to be traded using the Long Trend High Momentum strategy.")
        return True

    def check_ta(self):
        """
        Close of the SPY is above the 100-day simple moving average (SMA).
        This indicates a trend in the overall index.
        The close of the 25-day simple moving average is above the close of the 50-day simple moving average.
        """
        ticker_bars = self.get_historical_prices(self.stock.symbol,50,"day").df
        spy = self.get_historical_prices("SPY",100,"day").df
        if spy is None:
           # print("No data found! Please consider changing the ticker symbol.")
            return False
        sma_100 = ta.sma(spy["close"], length=100)
        if spy["close"].iloc[-1] < sma_100.iloc[-1]: # Close of the SPY is above the 100-day simple moving average (SMA).
            return False
        sma_25 = ta.sma(ticker_bars["close"], length=25)
        sma_50 = ta.sma(ticker_bars["close"], length=50)
        if sma_25 is None or sma_50 is None:
            #print("No data found! Please consider changing the ticker symbol.")
            return False
        if sma_25.iloc[-1] < sma_50.iloc[-1]: # The close of the 25-day simple moving average is above the close of the 50-day simple moving average.
            return False
      #  print(f"{self.stock} meets the technical analysis requirements to be traded using the Long Trend High Momentum strategy.")
        return True
        
    def stop_loss(self):
        """
        The day after entering the trade, place a stop-loss below the execution price of
        five times the average true range (ATR) of the last twenty days.
        """
        ticker_bars = self.get_historical_prices(self.stock.symbol,self.LAST_DAYS,"day").df
       # return 5*(ta.atr(ticker_bars["high"],ticker_bars["low"],ticker_bars["close"],length=20).iloc[-1])
        return 100

    def profit_taking(self):
        """
        No profit target; the goal is to ride this as high as it will go.
        """

    def position_sizing(self):
        """
        2 percent risk and 10 percent maximum percentage size, with a maximum of ten positions.
        """
        #TODO FIX THIS
        # if len(self.get_position(self.stock.symbol)) > 10:
        #     return 0
        size = 0.02 * self.cash # 2 percent risk 
        return size 
    
    def on_trading_iteration(self):
        # If all conditions are met, place an order 
        # TO BE PERFORMED ONCE EVERY DAY ON MARKET OPEN
        if not (self.check_if_tradeable() and self.check_ta()):
            return
                
       # order_size = self.position_sizing()
        order_size = 1
        if order_size == 0:
           # print("Position size is 0. No order will be placed.")
            return
        stop_loss = self.stop_loss()
        my_take_profit_price = self.get_historical_prices(self.stock.symbol,1,"day").df["close"].iloc[-1] * 1.25
        bars = self.get_historical_prices(self.stock.symbol,20,"day").df
        my_stop_loss_price = ta.atr(bars["high"],bars["low"],bars["close"],length=20).iloc[-1] * 5
        # Bypass the negative cash issue in backtesting
        if self.is_backtesting:
            # Place order only if i have enough cash
            if self.cash < order_size*self.get_historical_prices(self.stock.symbol,1,"day").df["close"].iloc[-1]:
                return
        
        print(f"{self.stock} meets all the requirements to be traded using the Long Trend High Momentum strategy.")
        # Place an oco order
        order = self.create_order(
            asset=self.stock,
            quantity=order_size,
            side="buy",
            take_profit_price=my_take_profit_price,
            stop_loss_price=my_stop_loss_price,
            position_filled=True,
            type="bracket",
            )
        self.submit_order(order)
       # print(f"Order placed for {self.stock} with a stop loss of {stop_loss}.")

def run_live():
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = LongTrendHighMomentum(broker=broker)

        # Run the strategy live
        trader.add_strategy(strategy)
        trader.run_all()

def run_backtest():
        # Define parameters
        backtesting_start = datetime(2023, 10, 23)
        backtesting_end = datetime(2024, 10, 23)
        budget = 1000
        # Run the backtest
        LongTrendHighMomentum.backtest(
            YahooDataBacktesting, backtesting_start, backtesting_end, budget=budget
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()