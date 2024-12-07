import requests
from datetime import datetime, timedelta


def get_into_msg(weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str):
    return f"Information:\n\n{weekly_pnl_str}\n\n{winrate_str}\n{trades_str}\n\n{positions_str}\n{orders_str}\n\n Please select what to do next."

async def get_bot_info():
    """ Call the server to get the bot info"""
    weekly_pnl_str = "💰  Weekly PnL:   +200.0 $"
    winrate_str = "🎯  Winrate:   56.0 %"
    trades_str = "📈  Trades:   100"
    positions_str = "📊  Active Positions:  10"
    orders_str = "✏️  Pending Orders:   100"
    return weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str


async def fetch_positions():
    past_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f'http://localhost:5000/positions?position_state=OPEN&from_date={past_week}'
    response = requests.get(url)
    return response.json()


async def get_active_positions():
    """ Call the server to get the active positions"""
    positions_data = await fetch_positions()
    positions = ""
    position_ctr=1
    for position in positions_data:  
        positions += f"{position_ctr}) {position[5]}    {position[2]} @   {position[4]}\nTP: {position[8]}    SL: {position[7]} \nStrategy: {position[1]}\n\n"
        position_ctr+=1
    return positions



async def fetch_orders():
    past_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f'http://localhost:5000/orders?order_state=new&from_date={past_week}'
    response = requests.get(url)
    return response.json()


async def get_active_orders():
    """ Call the server to get the active orders"""
    orders_data = await fetch_orders()
    orders = ""
    order_ctr=1
    for order in orders_data:  
        orders += f"{order_ctr}) {order[5]}  $ {order[2]} @ {order[4]}\nTP: {order[8]}    SL: {order[7]} \nStrategy: {order[1]}\n\n"
        order_ctr+=1
    return orders


    
last_order = None # Cache the last order

def set_last_order(order): # Set the last order in the cache
    global last_order
    last_order = order

async def fetch_last_order():
    url = 'http://localhost:5000/last_order' # Get the last order from the server
    response = requests.get(url)
    if last_order == response.json(): # If the last order is the same as the one in the cache, return None
        return None
    set_last_order(response.json()) # Set the last order in the cache
    return response.json()  # Return the last order
