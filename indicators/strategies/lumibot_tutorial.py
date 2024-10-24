from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

ALPACA_CONFIG = {
    "API_KEY":apikey,
    "API_SECRET": apisecret,
    "PAPER": True,  # Set to True for paper trading, False for live trading
}

# A simple strategy that buys AAPL on the first day and holds it
class MyStrategy(Strategy):
    def initialize(self):
        self.set_market("24/7")  
     
    def on_trading_iteration(self):
        if self.first_iteration:
            quantity = 1
            order = self.create_order("AAPL", quantity, "buy")
            self.submit_order(order)


# Pick the dates that you want to start and end your backtest
backtesting_start = datetime(2023, 10, 23)
backtesting_end = datetime(2024, 10, 23)

# # Run the backtest
# MyStrategy.backtest(
#     YahooDataBacktesting,
#     backtesting_start,
#     backtesting_end,
# )


trader = Trader()
broker = Alpaca(ALPACA_CONFIG)
strategy = MyStrategy(broker=broker)

# Run the strategy live
trader.add_strategy(strategy)
trader.run_all()