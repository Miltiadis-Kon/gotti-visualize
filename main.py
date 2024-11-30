'''
Run a strategy for multiple symbols
Set the start and end date to backtest
Set the budget for the backtest
Run the backtest
Run the live trading
'''

from datetime import datetime
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

from strategies.sma_cross import ChillGuy, run_backtest, run_live
from multiprocessing import Process

# Get 20 highest volume stocks today
tickers = ['NVDA', 'AAPL','AMZN','TSLA','MARA']
# Set global budget
global_budget = 10000
# Run strategy for each stock
if __name__ == '__main__':
    for ticker in tickers:
        print(f"Running strategy for {ticker}")
    #   p = Process(target=run_backtest, args=(ticker, datetime(2023, 10, 23),datetime(2024, 10, 23), global_budget))
        p = Process(target=run_live, args=(ticker,))
        p.start()   
        p.join()
        print(f"Finished strategy for {ticker}")
    print("All strategies finished")
