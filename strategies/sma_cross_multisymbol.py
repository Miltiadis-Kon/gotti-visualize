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
import streamlit as st
import sys
sys.path.append('./database')
import requests


load_dotenv()


apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")
PORT = 10000
HOST = "https://gotti-backend.onrender.com"

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
 
        self.set_market('24/7') #ONLY FOR DEMO PURPOSES
        self.sleeptime = "1M"  #ONLY FOR DEMO PURPOSES
         
        self.will_plot = self.parameters["Plot"]
        if self.will_plot:
            self.chart = chart.Chart(title="ChillGuy",toolbox=True)
        self.risk_percent = 0.01 # 2% risk per trade
        
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
                        asyncio.run(self.register_order(order))
                        print(f"Order submitted: {order}. Date: {self.get_datetime()}")
                        st.write(f"Order submitted: {order}. Date: {self.get_datetime()}")
     
    
    def add_asset(self,ticker):
        self.ticker_bars = self.get_historical_prices(ticker, 200, "day").df
        self.tradeable = False
        self.techincal = False
        self.tradeable_assets[ticker] = self.TradeableAsset(ticker)
                                           
    def on_filled_order(self, position, order, price, quantity, multiplier):
        # If the order is filled, we can print the order details
    #    print(f"Order filled: {order}.Status: {order.status} Date: {self.get_datetime()} . Remaining cash: {self.cash}")     
        if self.is_backtesting and self.will_plot:
            # Plot the trade
            if order.side == "buy":
                self.chart.marker(time=self.get_datetime(), position='below', color="green", shape="arrowUp")
            if order.side == "sell":
                self.chart.marker(time=self.get_datetime(), position='above', color="red", shape="arrowDown")    
                
    async def register_order(self, order):
        url = f"{HOST}/update_order_strategy"
        order_data = {
                "strategy": self.__class__.__name__,
                "order_id": str(order.identifier),
        }
        try:
            response = requests.post(url, json=order_data)
            print(response.json())
        except Exception as e:
            print(f"Error registering order: {e}")
            return False   
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

def run_backtest(tickers = ['NVDA', 'AAPL','AMZN','TSLA','MARA','NKE'],backtesting_start = datetime(2023, 10, 23), backtesting_end = datetime(2024, 10, 23), budget = 10000):
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
        
#region Execution
def run_live_crypto(tickers = ['SOL', 'BTC','DOT','SHIB','SUSHI']):
        parameters = {"Tickers": [Asset(symbol=ticker, asset_type=Asset.AssetType.CRYPTO) for ticker in tickers]}
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = ChillGuy(
            broker=broker,
            parameters=parameters
        )
        # Run the strategy live
        trader.add_strategy(strategy)
        trader.run_all()

#endregion Execution
