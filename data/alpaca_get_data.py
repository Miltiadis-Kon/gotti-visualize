# Plotly imports 
import plotly.graph_objects as go
import plotly.express as px

# Importing the api and instantiating the rest client according to our keys
import alpaca_trade_api as api
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')




alpaca = api.REST(API_KEY, API_SECRET, api_version='v2')



# Setting parameters before calling method
symbol = "AAPL"
timeframe = "15Min"
start = "2024-10-01"
end = "2024-10-06"
# Retrieve daily bars for SPY in a dataframe and printing the first 5 rows
spy_bars = alpaca.get_bars(symbol, timeframe, start, end).df
print(spy_bars.head())


#                            open    high  ...  trade_count        vwap
# timestamp                                  ...                         
# 2021-01-04 05:00:00+00:00  375.30  375.45  ...       623066  # 369.335676
# 2021-01-05 05:00:00+00:00  368.05  372.50  ...       338927  370.390186
# 2021-01-06 05:00:00+00:00  369.50  376.98  ...       575347  373.807251
# 2021-01-07 05:00:00+00:00  376.11  379.90  ...       366626  378.249233
# 2021-01-08 05:00:00+00:00  380.77  381.49  ...       391944  380.111637

# SPY bar data candlestick plot
candlestick_fig = go.Figure(data=[go.Candlestick(x=spy_bars.index,
               open=spy_bars['open'],
               high=spy_bars['high'],
               low=spy_bars['low'],
               close=spy_bars['close'])])
candlestick_fig.update_layout(
    title="Candlestick chart for $SPY",
    xaxis_title="Date",
    yaxis_title="Price ($USD)")
candlestick_fig.show()


