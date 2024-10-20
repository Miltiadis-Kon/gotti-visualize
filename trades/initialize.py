import MetaTrader5 as mt5
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables from a .env file
load_dotenv()

# Retrieve environment variables
login = int(os.getenv('MT5_LOGIN'))
server = os.getenv('MT5_SERVER')
password = os.getenv('MT5_PASSWORD')
 
# establish MetaTrader 5 connection to a specified trading account
if not mt5.initialize(login=login, server=server,password=password):
    print("initialize() failed, error code =",mt5.last_error())
    quit()
    
# display data on connection status, server name and trading account
print("Successful connection to the MetaTrader 5 terminal!")
print(20*'-')
print("Account number: ",mt5.account_info().login)
print("Current Balace: ",mt5.account_info().balance)

print(20*'-')
print("Positions")

# get the list of positions on symbols whose names contain "*USD*"
positions=mt5.positions_get()
if positions==None:
    print("No positions")
elif len(positions)>0:
    # display these positions as a table using pandas.DataFrame
    df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.drop(['comment','type','time_update', 'time_msc', 'time_update_msc', 'external_id','ticket','magic','reason','swap','identifier'], axis=1, inplace=True)
    print(df)

print(20*'-')
print("Orders  ")

# get the list of orders on symbols whose names contain "*GBP*"
orders=mt5.orders_get()
if orders is None or len(orders) == 0:
    print("No orders")
else:
    # display these orders as a table using pandas.DataFrame
    df=pd.DataFrame(list(orders),columns=orders[0]._asdict().keys())
    df.drop(['time_done', 'time_done_msc', 'position_id', 'position_by_id', 'reason', 'volume_initial', 'price_stoplimit'], axis=1, inplace=True)
    df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
    print(df)
 
print(20*'-')


# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
