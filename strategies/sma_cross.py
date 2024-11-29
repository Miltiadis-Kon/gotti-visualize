#region Imports
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
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "Plot": True, # True if you want to plot the trades, False if you don't want to plot the trades
    }
    
    
#region Core   
    ##### CORE FUNCTIONS #####
        
    def initialize(self):
        self.sleeptime = "1D" # Execute strategy every day.
        self.will_plot = self.parameters["Plot"]
        if self.will_plot:
            self.chart = chart.Chart(title="ChillGuy",toolbox=True)
        self.risk_percent = 0.2 # 2% risk per trade
    
    def before_market_opens(self):
        
        self.ticker_bars = self.get_historical_prices(self.parameters["Ticker"], 200, "day").df
        
        self.tradeable = self.filter()
        
        self.techincal = self.setup()
        
    def on_trading_iteration(self): 
        
        if self.tradeable and self.techincal:     
            position_size = self.get_position_sizing()
            if position_size != 0:           
                order = self.create_order(asset = self.parameters["Ticker"],
                                        quantity=position_size,
                                        side="buy"
                                        )
                
                self.submit_order(order)
#               print(f"Order submitted: {order}. Date: {self.get_datetime()}")
                                           
    def on_canceled_order(self, order):
        
        print(f"Order canceled: {order}. Status:{order.status} Date: {self.get_datetime()}")
        pass
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        # If the order is filled, we can print the order details
    #    print(f"Order filled: {order}.Status: {order.status} Date: {self.get_datetime()} . Remaining cash: {self.cash}")

        if self.is_backtesting and self.will_plot:
            # Plot the trade
            if order.side == "buy":
                self.chart.marker(time=self.get_datetime(), position='below', color="green", shape="arrowUp")
            if order.side == "sell":
                self.chart.marker(time=self.get_datetime(), position='above', color="red", shape="arrowDown")
    
        if  order.side == "sell":
            return
        take_profit = self.get_take_profit(price)
        stop_loss = self.get_stop_loss(price)
        
 #       print(f"Take profit: {take_profit}, Stop loss: {stop_loss}")
        # Update order's take profit and stop loss prices
        order2 = self.create_order(asset = self.parameters["Ticker"],
                                    quantity=quantity,
                                    side="sell",
                                    stop_loss_price=stop_loss,
                                    take_profit_price=take_profit,
                                    )
        order.add_child_order(order2)
        self.submit_order(order2)
    
#endregion Core           
                                 

#region Description
    ##### TRADING FUNCTIONS #####
    def filter(self):
        return True
        
    def setup(self):
        '''
        Simple SMA cross
        '''
        sma_50 = ta.sma(self.ticker_bars["close"], length=50)
        sma_200 = ta.sma(self.ticker_bars["close"], length=200)
        if sma_50.iloc[-1] > sma_200.iloc[-1]:
            return True
        else:
            return False
        
              
        
    def get_stop_loss(self,entry):
        '''
        50% stop loss
        '''
        return entry * 0.5
    
    def get_take_profit(self,entry):
        '''
        100% take profit
        '''
        return entry * 2
    
    def get_position_sizing(self): 
        '''
        2% risk per trade
        '''
        size = self.cash * self.risk_percent / self.ticker_bars["close"].iloc[-1]
        return size if size > 1 else 0
#endregion Description

#region Plot
    def on_strategy_end(self):
        if self.will_plot:
            self.plot()
        return super().on_strategy_end()
    
    
    def plot(self):
        self.chart.set(self.get_historical_prices(self.parameters["Ticker"],365, "day").df) # Set the data #TODO: make it dynamic        
        self.chart.show(block=True) # Show the chart
        

#endregion Plot


#region Execution
def run_live():
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = ChillGuy(broker=broker,
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
        ChillGuy.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)},
            show_plot=False,
            show_tearsheet=False,
            save_logfile=False,
            save_tearsheet=False,
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()
    
#endregion Execution
