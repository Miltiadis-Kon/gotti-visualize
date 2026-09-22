'''
Strategy 6: Short Mean Reversion High Six-Day Surge

A second mean reversion short system that does not overlap with Short RSI Thrust,
seeking profit from overbought stocks. 
This system might lose money in bull markets but can do very well in sideways and down markets.

An extended price increase of the stock 
means there is a large chance of it correcting 
and reverting back to its mean.

'''
import sys
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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from strategies.plot_mixin import PlottableStrategyMixin


load_dotenv()

apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

ALPACA_CONFIG = {
    "API_KEY":apikey,
    "API_SECRET": apisecret,
    "PAPER": True,  # Set to True for paper trading, False for live trading
}


class ShortMeanReversionHigh6DSurge(Strategy, PlottableStrategyMixin):
    
    parameters = {
        "AvgDailyShares": 1000000,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "TrailStopLoss" : False, # True if you want to use a trail stop with no tp,
                              #False if you want to use a 2:1 tp:sl ratio
        "Plot": True # True if you want to plot the trades, False if you don't want to plot the trades
    }
    
    ##### CORE FUNCTIONS #####
        
    def initialize(self):
        self.sleeptime = "1D" # Execute strategy every day once
        self.will_plot = self.parameters.get('Plot', True)
        self._plot_trades = []  # Collects trade records for get_plot_spec()
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
                                        limit_price=entry_price,
                                        side="sell",
                                        good_till_date=self.get_datetime() + timedelta(days=1)
                                        )
                self.submit_order(order)
#               print(f"Order submitted: {order}. Date: {self.get_datetime()}")
    
    def after_market_closes(self):
        # Cancel all open orders apart from stop loss/ take profit orders
        orders = self.get_orders()
        for order in orders:
            if order.status == "new" and order.side == "sell":
                self.cancel_order(order)
                        
    def on_canceled_order(self, order):
        
#       print(f"Order canceled: {order}. Status:{order.status} Date: {self.get_datetime()}")
        pass
    
        
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        
        # If the order is filled, we can print the order details
    #    print(f"Order filled: {order}.Status: {order.status} Date: {self.get_datetime()} . Remaining cash: {self.cash}")
        if  order.side == "sell":
            return
        
        take_profit = self.get_take_profit(price)
        stop_loss = self.get_stop_loss(price)
            
        # Update order's take profit and stop loss prices
        order2 = self.create_order(asset = self.parameters["Ticker"],
                                    quantity=order.quantity,
                                    take_profit_price=take_profit,
                                    stop_loss_price= stop_loss,
                                    side="buy",
                                    time_in_force="gtc"
                                    )
        
        
        order.add_child_order(order2)
    #    self.submit_order(order2)

        if self.is_backtesting and self.will_plot: 
            pass  # Trade visualization is handled via get_plot_spec() / render_plot()
            
    
     
    def on_strategy_end(self):
        if self.will_plot and self.is_backtesting:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            output_path = os.path.join(
                repo_root,
                'logs', 'charts',
                f"{self.parameters.get('Ticker', 'chart')}_chart.html"
            )
            self.save_plot_html(output_path)
                    
    ########################
    
    
    ##### TRADING FUNCTIONS #####
    def filter(self):
        """
        Minimum price $5
        Average dollar volume $10 million over the last fifty trading days.
        """
        if self.ticker_bars["close"].iloc[-1] < 5:
            return False
        
        df_vol = self.calculate_dollar_volume(self.ticker_bars,window=50)
               
        if df_vol["avg_dollar_volume"].iloc[-1] < 10000000:
            return False
                
        return True
        
    def setup(self):
        """
        The price of the stock has increased at least 20 percent over the last six trading days.
        Last two days had positive closes. 
        These two indicators mean the stock is very popular;
        there has been a lot of buying pressure.
        """
        if self.ticker_bars["close"].iloc[-1] < self.ticker_bars["close"].iloc[-2]:
            return False
        
        if self.ticker_bars["close"].iloc[-2] < self.ticker_bars["close"].iloc[-3]:
            return False
        
        if (self.ticker_bars["high"].iloc[-1] - self.ticker_bars["low"].iloc[-7]) < 0.2 * self.ticker_bars["low"].iloc[-7]:
            return False
        
        return True
    
    def calculate_dollar_volume(self,df, price_column='close', volume_column='volume', window=20):
        """
        Calculate dollar volume and related metrics.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and volume data
        price_column : str, default 'close'
            Name of the column containing price data
        volume_column : str, default 'volume'
            Name of the column containing volume data
        window : int, default 20
            Rolling window for average calculations
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the original data plus volume metrics
        """
        # Create a copy of the dataframe
        df_vol = df.copy()
        
        # Calculate dollar volume
        df_vol['dollar_volume'] = df_vol[price_column] * df_vol[volume_column]
        
        # Calculate rolling average dollar volume
        df_vol['avg_dollar_volume'] = df_vol['dollar_volume'].rolling(window=window).mean()
        
        # Calculate relative volume (compared to moving average)
        df_vol['relative_volume'] = df_vol['dollar_volume'] / df_vol['avg_dollar_volume']
        
        # Calculate volume momentum (rate of change)
        df_vol['volume_momentum'] = df_vol['dollar_volume'].pct_change(periods=window)
        
        # Add some additional volume metrics
        df_vol['volume_ma'] = df_vol[volume_column].rolling(window=window).mean()
        df_vol['volume_std'] = df_vol[volume_column].rolling(window=window).std()
        df_vol['volume_zscore'] = (df_vol[volume_column] - df_vol['volume_ma']) / df_vol['volume_std']
        
        return df_vol    
        
        
        
    def get_entry_price(self):
        """Sell limit 5 percent above the previous close"""
        return self.ticker_bars["close"].iloc[-1] * 0.95
        
        
    def get_stop_loss(self,entry):
        """
        3 times the ATR of the last ten days above the execution price.
        """
        atr = ta.atr(self.ticker_bars["high"], self.ticker_bars["low"], self.ticker_bars["close"], length=10)
        return entry + atr.iloc[-1] * 3
    
    def get_take_profit(self,entry):
        """
        If profit is 5 percent or more based on the closing price
        """
        return entry * 0.95
    
    def get_position_sizing(self): 
        """
        2 percent risk and 10 percent maximum percent size
        """
        return round((self.cash / self.ticker_bars["close"].iloc[-1]) * self.risk_percent)

    
    
    
    
    ############################
    
    def get_plot_spec(self):
        """Describe this strategy's visualization using the new plotting pipeline."""
        import yfinance as yf
        import pandas as pd
        import warnings
        from datetime import datetime, timedelta
        from plots.plot_spec import PlotSpec, CandlestickLayer, TradeMarkersLayer, IndicatorLayer

        ticker = self.parameters.get('Ticker')
        if hasattr(ticker, 'symbol'):
            ticker = ticker.symbol

        layers = []

        # Fetch daily OHLCV covering the backtest period
        try:
            end = datetime.now()
            start = end - timedelta(days=400)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                                 end=end.strftime('%Y-%m-%d'), interval='1d',
                                 progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                    'Close': 'close', 'Volume': 'volume'}, inplace=True)
                df.reset_index(inplace=True)
                layers.append(CandlestickLayer(data=df, ticker=ticker, timeframe='1D'))
        except Exception:
            pass

        # Add trade markers from scheduled plots
        if hasattr(self, '_plot_trades') and self._plot_trades:
            from plots.plot_spec import TradeMarkersLayer
            layers.append(TradeMarkersLayer(trades=self._plot_trades, show_tp_sl=True, show_zones=True))

        return PlotSpec(ticker=ticker, timeframe='1D', layers=layers)

    ############################



def run_live():
        trader = Trader()
        broker = Alpaca(ALPACA_CONFIG)
        strategy = ShortMeanReversionHigh6DSurge(broker=broker,
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
        ShortMeanReversionHigh6DSurge.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()