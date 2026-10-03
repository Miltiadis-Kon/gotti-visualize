"""
Objective: Short stocks in order to hedge when the markets move down.
When long positions start to lose money, this system should offset those losses.
This is the perfect add-on to an LTTF system, to capture those downward moves.
 
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
from lumibot.entities import Asset, Order
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

from strategies.strat_baseplate import StrategyBaseplate


class ShortRSIThrust(StrategyBaseplate):
    
    parameters = {
        **StrategyBaseplate.parameters,
        "AvgDailyVolume": 25000000,
        "SmaLength":50,
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "TrailStopLoss" : False, # True if you want to use a trail stop with no tp,
                              #False if you want to use a 2:1 tp:sl ratio
        "RiskRewardRatio" : 2, # Risk Reward Ratio for the trade
        "RiskPercent": 0.02, # 2 percent risk
        "MaxSizePercent" : 0.1, # 10 percent size
        "MaxPositions" : 10,
        "ATRPeriod" : 10,
        "TradingStyle": "swing_trading",
    }
    
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
        
        # Average true range percentage over the last ten days is 3 percent or more of the closing price of the stock.
        atr_percentage = (ta.atr(self.ticker_bars["high"],self.ticker_bars["low"],self.ticker_bars["close"],length=10)/self.ticker_bars["close"]).mean()
        if atr_percentage < 0.03:
            return False
        
        return True
    
    def setup(self):
        '''
        Three-day RSI is above ninety. 
        The last two days the close was higher than the previous day. 
        '''
        rsi = ta.rsi(self.ticker_bars["close"],length=3)
        if rsi.iloc[-1] < 90:
            return False
        if self.ticker_bars["close"].iloc[-1] < self.ticker_bars["close"].iloc[-2]:
            return False
        if self.ticker_bars["close"].iloc[-2] < self.ticker_bars["close"].iloc[-3]:
            return False
        return True
    
    def before_market_opens(self):
        self.ticker_bars = self.get_historical_prices(self.parameters["Ticker"], 50, "day").df
        self.is_tradeable = self.check_if_tradeable()
        self.is_setup_met = self.setup()
        return super().before_market_opens()
   
    def initialize(self):
        super().initialize()
        # Strategy parameters
        self.sleeptime = "1D" # Execute strategy every day once
        self.risk_percent = self.parameters.get("RiskPercent", 0.02) # 2 percent risk
        self.max_size_percent = self.parameters.get("MaxSizePercent", 0.1) # 10 percent size
        self.max_positions = self.parameters.get("MaxPositions", 10)
        
        # Trading parameters
        self.entry_percent = 0.04  # 4% above previous close
        self.profit_target = 0.04  # 4% profit target
        self.atr_period = self.parameters.get("ATRPeriod", 10)  # Period for ATR calculation
        self.atr_multiplier = 3  # Multiplier for stop loss
        
        # Track open positions
        self.positions_data = {}  # Dictionary to track position entry dates and prices
        
    def position_sizing(self): 
        """Calculate position size based on risk parameters"""
        if len(self.positions_data) >= self.max_positions:
            return 0     
        return int((self.get_portfolio_value() / self.ticker_bars["close"].iloc[-1]) * self.risk_percent)

    def get_atr(self, length=None, df=None):
        """Calculate ATR for the strategy"""
        resolved_length = length or getattr(self, "atr_period", 10)
        bars = df if df is not None else getattr(self, "ticker_bars", None)
        if bars is not None and len(bars) >= resolved_length:
            return ta.atr(bars['high'], bars['low'], bars['close'], length=resolved_length).iloc[-1]
        return super().get_atr(length=resolved_length, df=df)


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

        # Add trade markers if any
        if hasattr(self, '_plot_trades') and self._plot_trades:
            from plots.plot_spec import TradeMarkersLayer
            layers.append(TradeMarkersLayer(trades=self._plot_trades, show_tp_sl=True, show_zones=True))

        return PlotSpec(ticker=ticker, timeframe='1D', layers=layers)


        
        

    def on_trading_iteration(self):
        '''
        Entry : Next day, sell short 4 percent above the previous closing price.
        
        Stop loss : The day after we place the order,
                    place a buy stop of three times ATR of the last ten days above the execution price.
                    
        Take profit : If at the closing price the profit in the position is 4 percent or higher,
                      get out the next day’s market on close.
                         
        Expiration : If after two days the trade has not reached its profit target,
                     we place a market order on close for the next day.
                     
        Position size : 2 percent risk and 10 percent size, a maximum of ten positions 
        '''
        
        if  self.is_tradeable and  self.is_setup_met:
                # Place new orders
                
                # calculate entry price
                entry_price = float(self.ticker_bars["close"].iloc[-1] * (1 + self.entry_percent))
                # calculate custom stop loss and take profit
                custom_tp = entry_price * (1 - abs(self.profit_target))  # 4% profit target below entry for short
                custom_sl = entry_price + (self.get_atr() * self.atr_multiplier)  # 3x ATR above entry for short
                
                # Pass through ATR optimization function
                risk_levels = self.optimize_sl_tp(
                    entry_price=entry_price,
                    stop_loss=custom_sl,
                    take_profit=custom_tp,
                    side="sell",
                    atr=self.get_atr(),
                    trading_style=self.parameters.get("TradingStyle", "swing_trading"),
                    sl_multiplier=self.atr_multiplier,
                    risk_reward_ratio=self.parameters.get("RiskRewardRatio", 2),
                    trailing_stop=self.parameters.get("TrailStopLoss", False),
                )
                tp = risk_levels.take_profit
                sl = risk_levels.stop_loss
                
                # calculate position size
                position_size = self.position_sizing() 
                
                if position_size == 0:
                    return
                # place order
                order = self.create_order(
                    asset=self.parameters["Ticker"],
                    quantity=position_size,
                    side="sell_short",
                    stop_loss_price=sl,
                    take_profit_price=tp,
                    position_filled=True,
                    type="bracket",
                    time_in_force="gtd",
                    good_till_date=self.get_datetime() + timedelta(days=2),
                )
                
                print(f"\nPlacing order: {order.side} {position_size} {order.asset} at {entry_price} on {self.get_datetime()}")            
                order = self.submit_order(order)
                pass  # Trade visualization is handled via get_plot_spec() / render_plot()
    
    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Update position data after order is filled without double-stacking child bracket orders"""
        ticker = self.parameters.get("Ticker")
        if order.asset == ticker:
            print(f"Order filled: {order.side} {quantity} {order.asset} at {price} on {self.get_datetime()}")  
            if order.side in ("sell", "sell_short") and position and position.quantity < 0:
                self.positions_data[order.identifier] = {"entry_date": self.get_datetime()}
            elif order.side in ("buy", "buy_to_cover"):
                # Exit fill — clear tracking and reset trailing stop
                self.positions_data.clear()
                self.current_trailing_stop = None

    def on_strategy_end(self):
        if getattr(self, "is_backtesting", False) and getattr(self, "will_plot", False):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
        strategy = ShortRSIThrust(broker=broker,
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
        ShortRSIThrust.backtest(
            YahooDataBacktesting,
            backtesting_start,
            backtesting_end,
            budget=budget,
            parameters={"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK)}
        )


if __name__ == "__main__":
    run_backtest()
    #run_live()