#region Imports
import asyncio
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
import lightweight_charts as chart

import sys
sys.path.append('./database')
import db_functions as sql
import requests
import signal


load_dotenv()

apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

ALPACA_CONFIG = {
    "API_KEY":apikey,
    "API_SECRET": apisecret,
    "PAPER": True,  # Set to True for paper trading, False for live trading
}

#endregion Imports

class ChillGuy(Strategy):
    
    parameters = {
        "Tickers": ['NVDA', 'AAPL','AMZN','TSLA','MARA'],
        "Plot": False, # True if you want to plot the trades, False if you don't want to plot the trades
    }
    
    class TradeableAsset:
        def __init__(self, ticker):
            self.ticker = ticker
            self.ticker_bars = None
            self.tradeable = None
            self.techincal = None
            
        def __str__(self):
            return f"Ticker: {self.ticker} Tradeable: {self.tradeable} Technical: {self.techincal}"  
#region Core   
    ##### CORE FUNCTIONS #####
        
    def initialize(self):        
        if self.parameters["Tickers"][0].asset_type == Asset.AssetType.CRYPTO:
            self.set_market('24/7')
        else:
            self.set_market('NASDAQ')
            
            
        self.sleeptime = "1D" # Execute strategy every day.
        self.will_plot = self.parameters["Plot"]
        if self.will_plot:
            self.chart = chart.Chart(title="ChillGuy",toolbox=True)
        self.risk_percent = 0.2 # 2% risk per trade
        
        # Create a dictionary to store the tradeable assets
        self.tradeable_assets = {}
        for ticker in self.parameters["Tickers"]:
            self.add_asset(ticker)
    
    def before_market_opens(self):
        for ticker in self.tradeable_assets:
            self.tradeable_assets[ticker].ticker_bars = self.get_historical_prices(ticker, 200, "day").df
            self.tradeable_assets[ticker].tradeable = self.filter(self.tradeable_assets[ticker])
            self.tradeable_assets[ticker].techincal = self.setup(self.tradeable_assets[ticker])
    #        print(f"Tradeable: {self.tradeable_assets[ticker].tradeable} Technical: {self.tradeable_assets[ticker].techincal}")
        
    def on_trading_iteration(self):
        for ticker in self.tradeable_assets:      
            if self.tradeable_assets[ticker].tradeable and self.tradeable_assets[ticker].techincal:     
                position_size = self.get_position_sizing(self.tradeable_assets[ticker])
                if position_size != 0:
                    lp=self.get_last_price(self.tradeable_assets[ticker].ticker)
                    stop_loss = self.get_stop_loss(lp)
                    take_profit = self.get_take_profit(lp)
                    order = self.create_order(asset = self.tradeable_assets[ticker].ticker,
                                            quantity=position_size,
                                            side="buy",
                                            type="bracket",
                                            stop_loss_price=stop_loss,
                                            take_profit_price=take_profit,
                                            )
                    self.submit_order(order)
                    
                    # Register order on database
                    if not self.is_backtesting:
                        self.register_order(order)
                        print(f"Order submitted: {order}. Date: {self.get_datetime()}")
     
    
    def add_asset(self,ticker):
        self.ticker_bars = self.get_historical_prices(ticker, 200, "day").df
        self.tradeable = False
        self.techincal = False
        self.tradeable_assets[ticker] = self.TradeableAsset(ticker)
                                           
    def on_filled_order(self, position, order, price, quantity, multiplier):
        # If the order is filled, we can print the order details
    #    print(f"Order filled: {order}.Status: {order.status} Date: {self.get_datetime()} . Remaining cash: {self.cash}")
        if not self.is_backtesting:
            self.register_position(order, position,price,quantity)
            
            print(f"Position opened: {position}. Date: {self.get_datetime()}")
        if self.is_backtesting and self.will_plot:
            # Plot the trade
            if order.side == "buy":
                self.chart.marker(time=self.get_datetime(), position='below', color="green", shape="arrowUp")
            if order.side == "sell":
                self.chart.marker(time=self.get_datetime(), position='above', color="red", shape="arrowDown")    
                
    def register_order(self, order):
            url = "http://127.0.0.1:5000/orders"
            order_data = {
                "strategy": self.__class__.__name__,
                "symbol": order.asset.symbol,
                "quantity": order.quantity,
                "price": self.get_last_price(order.asset),
                "side": order.side,
                "order_id": str(order.identifier),
                "order_state": order.status,
                "stop_loss_price": order.stop_loss_price,
                "take_profit_price": order.take_profit_price
            }
            response = requests.post(url, json=order_data)
            if response.status_code == 201:
                print("Order posted successfully")
            else:
                print("Failed to post position")
        
    def register_position(self, order, position,price,quantity):
        url = "http://127.0.0.1:5000/positions"
        position_data = {
            "strategy": self.__class__.__name__,
            "symbol": order.asset.symbol,
            "quantity": quantity,
            "price": price,
            "side": order.side,
            "stop_loss_price": order.stop_loss_price,
            "take_profit_price": order.take_profit_price,
            "based_on_order_id": str(order.identifier),
            "position_state": position.status
        }
        response = requests.post(url, json=position_data)
        if response.status_code == 201:
            print("Position posted successfully")
        else:
            print("Failed to post position")
#endregion Core           
                                 

#region Description
    ##### TRADING FUNCTIONS #####
    def filter(self,tradeable_asset):
        return True
        
    def setup(self,tradeable_asset):
        '''
        Simple SMA cross
        '''
        sma_50 = ta.sma(tradeable_asset.ticker_bars["close"], length=50)
        sma_200 = ta.sma(tradeable_asset.ticker_bars["close"], length=200)
    
        if sma_50.iloc[-1] > sma_200.iloc[-1]:
            return True
        else:
            return False
                            
    def get_stop_loss(self,entry):
        '''
        5% stop loss
        '''
        return round(entry * 0.95,2)
    
    def get_take_profit(self,entry):
        '''
        10% take profit
        '''
        return round(entry * 1.1,2)
    
    def get_position_sizing(self,tradeable_asset): 
        '''
        2% risk per trade
        '''
        cash = self.get_cash()
        size = cash * self.risk_percent / tradeable_asset.ticker_bars["close"].iloc[-1]
        size = int(size)
        return size if size > 1 else 0
#endregion Description

#region Plot
    def on_strategy_end(self):
        if self.will_plot:
            self.plot()    
    
    def plot(self):
        self.chart.set(self.get_historical_prices(self.parameters["Ticker"],365, "day").df) # Set the data #TODO: make it dynamic        
        self.chart.show(block=True) # Show the chart
        

#endregion Plot


#region Execution
def run_live(tickers = ['NVDA', 'AAPL','AMZN','TSLA','MARA']):
        parameters = {"Tickers": [Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK) for ticker in tickers]}
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = ChillGuy(
            broker=broker,
            parameters=parameters
        )
        # Run the strategy live
        trader.add_strategy(strategy)
        trader.run_all()

def run_backtest(tickers = ['NVDA', 'AAPL','AMZN','TSLA','MARA'],backtesting_start = datetime(2023, 10, 23), backtesting_end = datetime(2024, 10, 23), budget = 10000):
        # Define parameters
        parameters = {"Tickers": [Asset(symbol=ticker, asset_type=Asset.AssetType.STOCK) for ticker in tickers]}
        # Run the backtest    
        ChillGuy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters=parameters,
            show_plot=True,
            show_tearsheet=True,
            save_logfile=False,
            save_tearsheet=False,
        )

#endregion Execution

if __name__ == '__main__':
   # run_backtest( )
    run_live()
