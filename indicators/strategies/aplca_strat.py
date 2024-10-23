from dotenv import load_dotenv
import os
load_dotenv()
from alpaca.trading.client import TradingClient



import asyncio


key = os.getenv('APCA_API_KEY_PAPER')
secret = os.getenv('APCA_API_SECRET_KEY_PAPER')
base = 'https://paper-api.alpaca.markets'


'''WRAPPER TO GET DATA FROM YFINANCE'''





'''WRAPPER TO GET DATA FROM ALPACA MARKET API'''

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestBarRequest
from alpaca.data.live import CryptoDataStream


class AlpacaTradingBot_LIVE : 

    def __init__(self) :
        self.trading_client = TradingClient(api_key=key, secret_key=secret,paper=True)
        print(self.trading_client.get_account().buying_power)
        
        self.stream = CryptoDataStream(api_key=key, secret_key=secret)
        self.streamed_data = []
    
      
    def run(self):
        
        async def quote_data_handler(data):
        # quote data will arrive here
            print('LIVE DATA: ', data)
            self.streamed_data.append(data) 
            if (self.streamed_data[0].close < self.streamed_data[-1].close):
                if (self.streamed_data[-1].close < self.streamed_data[-2].close):
                    print('BUY CREATE, %.2f' % self.streamed_data[0].close)
                    self.trading_client.submit_order('SOL/USD', 1, 'buy', 'market', 'gtc')
        
        self.stream.unsubscribe_bars()
        self.stream.subscribe_bars(quote_data_handler,'SOL/USD')
        self.stream.run()

#bot2 = AlpacaTradingBot_LIVE()
#bot2.run()

class AlpacaTradingBot_HISTORICAL():
    
    def __init__(self):
        self.trading_client = TradingClient(api_key=key, secret_key=secret,paper=True)
        print(self.trading_client.get_account().buying_power)
        self.cryptoClient = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
        self.streamed_data = []
    
    def process(self):
        
        def quote_data_handler(data):
            print('HISTO DATA: ', data)
            self.streamed_data.append(data)
            sol_data = [d['SOL/USD'] for d in self.streamed_data if 'SOL/USD' in d]
            if (sol_data[0].close < sol_data[-1].close):
                if (sol_data[-1].close < sol_data[-2].close):
                    print('BUY CREATE, %.2f' % sol_data[0].close)
                    self.trading_client.submit_order('SOL/USD', 1, 'buy', 'market', 'gtc')
        params = CryptoLatestBarRequest(symbol_or_symbols='SOL/USD')
        
        bars = self.cryptoClient.get_crypto_latest_bar(params)
        quote_data_handler(bars)
    
    
    def run(self):
        async def periodic_process():
            while True:
                self.process()
                await asyncio.sleep(60)  # Sleep for 5 minutes

        asyncio.run(periodic_process())
        

#bot1 = AlpacaTradingBot_HISTORICAL()
#bot1.run()




