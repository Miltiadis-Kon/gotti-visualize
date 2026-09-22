'''
System 4: Long Trend Low Volatility


'''

import sys
import numpy as np
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


class LongTrendLowVolatility(Strategy, PlottableStrategyMixin):
    
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
        
        self.snp_bars = self.get_historical_prices(Asset(symbol="SPY", asset_type=Asset.AssetType.STOCK), 200, "day").df
        
        self.tradeable = self.filter()
        
        self.techincal = self.setup()
        
    def on_trading_iteration(self): 
        '''
        Entry on market open.
        No TP.
        Trailing stop of 20%.
        SL of 1.5 * ATR40
        '''
        if self.tradeable and self.techincal:
            # Entry market on open
            position_size = self.get_position_sizing()
            if position_size != 0:           
                order = self.create_order(asset = self.parameters["Ticker"],
                                        quantity=position_size,
                                        side="buy",
                                        trail_percent=0.2,
                                        )
                self.submit_order(order)
#               print(f"Order submitted: {order}. Date: {self.get_datetime()}")
 
        
           
    def on_filled_order(self, position, order, price, quantity, multiplier):
        
        # If the order is filled, we can print the order details
    #    print(f"Order filled: {order}.Status: {order.status} Date: {self.get_datetime()} . Remaining cash: {self.cash}")
        if  order.side == "sell":
            return
        
        stop_loss = self.get_stop_loss(price)
        # Update order's take profit and stop loss prices
        order2 = self.create_order(asset = self.parameters["Ticker"],
                                    quantity=order.quantity,
                                    stop_loss_price= stop_loss,
                                    side="sell_to_open",
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
        Average daily dollar volume greater than $100 million over the last fifty days.
        Historic volatility rating between 10 and 40 percent, which puts us in the lower range on that metric.
        """
        if self.ticker_bars["volume"].iloc[-50].mean() < 100000000:
            return False
        
        historical_volatility = self.calculate_historical_volatility(self.ticker_bars, price_column='close', window=21, trading_days=252)["annualized_volatility"].iloc[-1]
        
#       print(historical_volatility)
        
        if historical_volatility < 0.1 or historical_volatility > 0.4:
            return False
        
        return True
        
    def setup(self):
        """
        Close of the S&P 500 is above 200-day simple moving average. 
        Close of the stock is above the 200-day simple moving average.
        """
        sma200 = ta.sma(self.ticker_bars["close"], length=200)
        if self.ticker_bars["close"].iloc[-1] < sma200.iloc[-1]:
            return False
        
        sma200_snp = ta.sma(self.snp_bars["close"], length=200)
        
        if self.snp_bars["close"].iloc[-1] < sma200_snp.iloc[-1]:
            return False
        
        return True
    
    
    def calculate_historical_volatility(self,df, price_column='close', window=21, trading_days=252):
        """
        Calculate historical volatility for a given price series.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the price data
        price_column : str, default 'close'
            Name of the column containing price data
        window : int, default 21
            Rolling window for volatility calculation (typically 21 for monthly)
        trading_days : int, default 252
            Number of trading days in a year
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the original data plus volatility metrics
        """
        # Create a copy of the dataframe
        df_vol = df.copy()
        
        # Calculate daily returns
        df_vol['daily_return'] = df_vol[price_column].pct_change()
        
        # Calculate daily log returns
        df_vol['log_return'] = np.log(df_vol[price_column] / df_vol[price_column].shift(1))
        
        # Calculate rolling standard deviation of log returns
        df_vol['volatility'] = df_vol['log_return'].rolling(window=window).std()
        
        # Annualize the volatility
        df_vol['annualized_volatility'] = df_vol['volatility'] * np.sqrt(trading_days)
        
        # Convert volatility to percentage
        df_vol['volatility_pct'] = df_vol['volatility'] * 100
        df_vol['annualized_volatility_pct'] = df_vol['annualized_volatility'] * 100
        
        return df_vol    
        
        
    def get_stop_loss(self,entry):
        """
        The day after execution, we place a stop-loss of 
        one-and-a-half times the average true range (ATR) 
        of the last forty days below the execution price.        
        """
        atr = ta.atr(self.ticker_bars["high"], self.ticker_bars["low"], self.ticker_bars["close"], length=40)
        return entry - atr.iloc[-1] * 1.5
    
    
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
        strategy = LongTrendLowVolatility(broker=broker,
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
        budget = 10000
        # Run the backtest    
        LongTrendLowVolatility.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()