'''
The Opening Gap: Why Is This the First and Highest-Probability Play of the Day?

https://www.simplertrading.com/products/free-courses/mastering-the-trade?view=ch_7__part_2_opening_gap



One of the lowest-risk trades available.

A gap occurs when the opening price of the next day’s regular cash session
is greater or lower than the closing price of the previous day’s regular cash session,
creating a “gap” in price levels on the charts.

For this play, I’m specifically interested in gaps that have a 
high probability of filling on the same day they are created.


For example, if AAPL gaps up 1.00 percent, the overall S&P 500 is also gapping up 1.00 percent, 
and there isn’t any specific news on AAPL, then it can be played as a gap play. 
It’s just moving with the overall market.
 
It’s not the actual news, but how the markets respond to that news that is important.
 



From the book : Carter, John F - Mastering the trade proven techniques for profiting from intraday and swing trading setups-McGraw-Hill Education (2019)
'''

import webbrowser
import pandas_ta as ta
import pandas as pd
from datetime import datetime, timedelta
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




class OpeningGap(Strategy):
    
    parameters = {
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "Plot": True, # True if you want to plot the trades, False if you don't want to plot the trades

    }
    
    
    
    ##### CORE FUNCTIONS #####
        
    def initialize(self):
        self.sleeptime = "5M" # Execute strategy every 5 minutes.
        self.will_plot = self.parameters["Plot"]
        self.risk_percent = 0.02 # 2% risk per trade
    
    def before_market_opens(self):
        
        self.ticker_bars = self.get_historical_prices(self.parameters["Ticker"], 200, "day").df
        
        self.tradeable = self.filter()
        
        self.techincal = self.setup()
        
    def on_trading_iteration(self): 
        
        if self.tradeable and self.techincal:     
            entry_price = self.get_entry_price()
            position_size = self.get_position_sizing()
            if position_size != 0:           
                order = self.create_order(asset = self.parameters["Ticker"],
                                        quantity=position_size,
                                        side="buy",
                                        )
                
                self.submit_order(order)
#               print(f"Order submitted: {order}. Date: {self.get_datetime()}")
    
    def after_market_closes(self):
        # Cancel all open orders apart from stop loss/ take profit orders
        pass
                
                                        
    def on_canceled_order(self, order):
        
#       print(f"Order canceled: {order}. Status:{order.status} Date: {self.get_datetime()}")
        pass
    
        
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        '''
        Time-based: After six trading days, if not stopped out and the profit target is not hit,
                    then exit next day market on open.
        '''
        # If the order is filled, we can print the order details
    #    print(f"Order filled: {order}.Status: {order.status} Date: {self.get_datetime()} . Remaining cash: {self.cash}")
        if  order.side == "buy":
            return
        
        take_profit = self.get_take_profit(price)
        stop_loss = self.get_stop_loss(price)
        
        # Update order's take profit and stop loss prices
        order2 = self.create_order(asset = self.parameters["Ticker"],
                                    quantity=order.quantity,
                                    take_profit_price=take_profit,
                                    stop_loss_price= stop_loss,
                                    side="sell",
                                    time_in_force="gtd",
                                    good_till_date=self.get_datetime() + timedelta(days=6)
                                    )
        
        order.add_child_order(order2)
    #    self.submit_order(order2)

        if self.is_backtesting and self.will_plot: 
            # Plot the trade
            pass
            
                     
    
    ##### TRADING FUNCTIONS #####
    def filter(self):
        '''
        What I’m looking for is the premarket volume in these stocks as of 9:20 a.m.
        eastern, 10 minutes before the regular cash session opens. 
        
        Take a look at where the $VIX is trading and you’ll be able to get the approximate 
        volume numbers you’ll need. 
        If it’s trading at 60.00, you’ll need to triple the numbers given here; 
        if it’s trading at 10.00, you’ll need to cut them in half; and so forth.
        
        If these stocks are trading less than 30,000 shares each at this time,
        the gap (up or down) has an approximately 85 percent chance of filling that same day.
        However, if the volume jumps up to 50,000 shares each,
        the gap has only about a 60 percent chance of filling that same day.
        
        If the premarket volume jumps to more than 70,000 shares each,
        the chances of the gap filling that same day drop to 30 percent.
        
        
        
        '''
        pass
        
    def setup(self):
        pass
        
       
    def get_entry_price(self):
        pass
        
        
    def get_stop_loss(self,entry):
        pass
    
    def get_take_profit(self,entry):
        pass
    
    def get_position_sizing(self): 
        pass



def run_live():
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = OpeningGap(broker=broker,
                                         parameters={"Ticker": Asset(symbol="NVDA",
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
        budget = 2000
        # Run the backtest    
        OpeningGap.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()