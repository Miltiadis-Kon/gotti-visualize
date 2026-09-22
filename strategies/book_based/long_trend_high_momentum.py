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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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


# TODO: Migrate rank on screener
def rank():
    """
    In case we have more setups than our position sizing allows,
    we rank by the highest rate of change over the last 200 trading days.
    This means the highest percentage price increase over the last 200 trading days.
    """
    pass

class LongTrendHighMomentum(Strategy, PlottableStrategyMixin):
    
    parameters = {
        "AvgDailyVolume": 50000000,
        "SmaLength":50,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "TrailStopLoss" : False, # True if you want to use a trail stop with no tp,
                              #False if you want to use a 2:1 tp:sl ratio
        "RiskRewardRatio" : 2, # Risk Reward Ratio for the trade
        "Plot": True # True if you want to plot the trades, False if you don't want to plot the trades
    }
        
    def initialize(self):
        self.sleeptime = "1D" # Execute strategy every day once
        self.will_plot = self.parameters.get('Plot', True)
        self._plot_trades = []  # Collects trade records for get_plot_spec()
        self.positions_count = 0 
        
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
    
    ##### PLOT FUNCTIONS #####

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

        ##### PLOT FUNCTIONS #####

    ############################

    
    
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
        
        if self.is_backtesting and self.will_plot:
            pass  # Trade visualization is handled via get_plot_spec() / render_plot()  
    
    def on_strategy_end(self):
        if self.will_plot and self.is_backtesting:
            ticker = self.parameters.get('Ticker', 'chart')
            ticker_symbol = ticker.symbol if hasattr(ticker, 'symbol') else str(ticker)
            output_path = os.path.join(
                repo_root,
                'logs', 'charts',
                f"{ticker_symbol}_chart.html"
            )
            self.save_plot_html(output_path)
                
        return super().on_strategy_end()  

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