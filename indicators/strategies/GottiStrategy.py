
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



#TODO: Implement general strategy class to reduce boilerplate code!

class GottiStrategy(Strategy):

    def setup_keys(self):
        load_dotenv()
        apikey = os.getenv("APCA_API_KEY_PAPER")
        apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

        self.ALPACA_CONFIG = {
            "API_KEY":apikey,
            "API_SECRET": apisecret,
            "PAPER": True,  # Set to True for paper trading, False for live trading
        }
            
    def check_if_tradeable(self):
        '''
        Check if asset meets trade setup criteria
        '''
        return bool
    
    def check_technical_indicators(self):
        '''
        Check if asset meets technical indicators criteria
        '''
        return bool
    
    
    
    ##### CORE FUNCTIONS #####
        
    def initialize(self):
        self.sleeptime = "1D" # Execute strategy every day once
        self.will_plot = self.parameters["Plot"]


        if self.is_backtesting and self.will_plot: # Initialize the plot
            self.initialize_plot()
            self.plots = []
    
    def before_market_opens(self):
         return super().before_market_opens()
    
    def on_trading_iteration(self):  
        
        if not self.check_if_tradeable and not self.check_technical_indicators:
            # Execute trade
            return
        
        
        
        
        return super().on_trading_iteration()
     
    def on_strategy_end(self):
        if self.will_plot:
            self.plot()
            self.fig.write_html(f".\logs\charts\Chart.html")
            webbrowser.open(f".\logs\charts\Chart.html")
                
        return super().on_strategy_end()
    
    ########################
    
        ##### PLOT FUNCTIONS #####
    
    def schedule_plot(self,order,price,date):
        """Schedule the plot to be executed after the order is filled"""
        self.plots.append({"order": order, "price": price, "date": date})
        current_date = self.get_datetime()
        current_date_timestamp = pd.Timestamp(current_date - timedelta(days=60))
        for plot in self.plots:
        #    print (current_date_timestamp - plot["date"])
            if (current_date_timestamp - plot["date"]).days > 0:
                self.plots.remove(plot)      
                # Return a scatter list  
                expiration = plot["date"] + timedelta(days=90)
                
                buy = go.Scatter(x=[plot["date"],expiration],
                                 y=[plot["price"],plot["price"]],
                                    mode="lines",
                                    marker=dict(size=[10],color="blue"),
                                    name="Entry Price")
                
                stop_loss = go.Scatter(x=[plot["date"],expiration],
                                 y=[plot["order"].stop_loss_price,plot["price"]],
                                    mode="lines",
                                    marker=dict(size=[10],color="red"),
                                    name="Stop Loss")
                
                take_profit = go.Scatter(x=[plot["date"],expiration],
                                 y=[plot["order"].take_profit_price,plot["order"].take_profit_price],
                                    mode="lines",
                                    marker=dict(size=[10],color="green"),
                                    name="Take Profit")
                
                scatter = [buy,stop_loss,take_profit]
                self.scatters.append(scatter)                
                break        
       
    def initialize_plot(self):
        """Initialize the plot"""
        self.date = self.get_datetime()
        self.fig = go.Figure()
        
        self.scatters = []
        
        self.fig.update_layout(title=f"{self.parameters['Ticker']} - Long Trend High Momentum Strategy",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            template="plotly_dark")
      
    def plot(self):
        """Plot the strategy"""
        diff = abs(self.get_datetime() - self.date).days
        data = self.get_historical_prices(self.parameters["Ticker"],diff,"day").df
        self.fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                   open=data['open'],
                                                   high=data['high'],
                                                   low=data['low'],
                                                   close=data['close'])])
        for scatter in self.scatters:
            self.fig.add_traces(scatter) 
                # Create buttons to toggle scatter traces
                
        buttons = []
        for i, scatter in enumerate(self.scatters):
            buttons.append(dict(
                label=f'Scatter {i+1}',
                method='update',
                args=[{'visible': [True] * (len(self.fig.data) - len(self.scatters)) + [False] * len(self.scatters)},
                      {'title': f'Scatter {i+1}'}]
            ))
            buttons[-1]['args'][0]['visible'][len(self.fig.data) - len(self.scatters) + i] = True

        # Add buttons to layout
        self.fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="down",
                buttons=buttons,
                showactive=True,
            )]
        )      
    ############################    
    
    
    def run_live(self):
        trader = Trader()
        broker = Alpaca(self.ALPACA_CONFIG)
        strategy = LongMeanReversionSelloff(broker=broker,
                                         parameters={"Ticker": Asset(symbol="NIO",
                                                                    asset_type=Asset.AssetType.STOCK)
                                                     }
                                        )

        # Run the strategy live
        trader.add_strategy(strategy)
        trader.run_all()

    def run_backtest(self):
        # Define parameters
        backtesting_start = datetime(2023, 10, 23)
        backtesting_end = datetime(2024, 10, 23)
        budget = 10000
        # Run the backtest    
        LongMeanReversionSelloff.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )

    