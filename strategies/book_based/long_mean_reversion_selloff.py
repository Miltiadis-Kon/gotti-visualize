'''
Strategy 3: Long Mean Reversion Selloff

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


class LongMeanReversionSelloff(Strategy, PlottableStrategyMixin):
    
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
                                        side="buy",
                                        good_till_date=self.get_datetime() + timedelta(days=1)
                                        )
                self.submit_order(order)
#               print(f"Order submitted: {order}. Date: {self.get_datetime()}")
    
    def after_market_closes(self):
        # Cancel all open orders apart from stop loss/ take profit orders
        orders = self.get_orders()
        for order in orders:
            if order.status == "new" and order.side == "buy":
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
                                    side="sell",
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
        Minimum price of $1.00
        Average volume over the last fifty days of 1 million shares
        Average true range over the last ten days is 5 percent or higher. 
        This gets us into volatile stocks, which we need for this system to work.
        """
        if self.ticker_bars["close"].iloc[-1] < 1:
            return False
          
        shares = self.ticker_bars["volume"].iloc[-50].mean() / self.ticker_bars["close"].iloc[-50].mean()
        if shares < self.parameters["AvgDailyShares"]:
            return False
        
        atr = ta.atr(self.ticker_bars["high"], self.ticker_bars["low"], self.ticker_bars["close"], length=10)
        if atr is None or atr.iloc[-1] < 0.05:
            return False
        
        return True
        
    def setup(self):
        """
        Close is above the 150-day Simple Moving Average
        The stock has dropped 12.5 percent or more in the last three days.
        This setup measures a significant downward move in an uptrending stock.
        """
        sma150 = ta.sma(self.ticker_bars["close"], length=150)
        if self.ticker_bars["close"].iloc[-1] < sma150.iloc[-1]:
            return False
        
        if self.ticker_bars["close"].iloc[-1] < self.ticker_bars["close"].iloc[-4] * 0.875:
            return False
        
        return True
        
        
        
        
    def get_entry_price(self):
        """Limit order of 7 percent below the previous closing price."""
        return self.ticker_bars["close"].iloc[-1] * 0.93
        
        
    def get_stop_loss(self,entry):
        """
        2.5 times the ATR of the last ten days below the execution price.
        """
        atr = ta.atr(self.ticker_bars["high"], self.ticker_bars["low"], self.ticker_bars["close"], length=10)
        return entry - atr.iloc[-1] * 2.5
    
    def get_take_profit(self,entry):
        """
        If profit is 4 percent or more based on the closing price
        """
        return entry * 1.04
    
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
        strategy = LongMeanReversionSelloff(broker=broker,
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
        LongMeanReversionSelloff.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()