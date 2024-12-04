



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

async def get_active_positions():
    """ Call the server to get the active positions"""
    positions = "TICKER |  SIZE  |  ENTRY  |  PNL | TAKE PROFIT | STOP LOSS \n\n\
                 ΤSLA |  100  |  345 $  | + 200 $ |  400 $ |  300 $ \n\n\
                 AMZN |  50  |  50 $  | - 100 $ |  600 $ |  400 $ \n\n\
                 NVDA |  20  |  245 $  | + 50 $ |  150 $ |  50 $ \n\n\
                "
    return positions


async def get_active_orders():
    """ Call the server to get the active orders"""
    orders = "TICKER |  SIZE  |  ENTRY  |  PNL | TAKE PROFIT | STOP LOSS \n\n\
                 ΤSLA |  100  |  345 $  | + 200 $ |  400 $ |  300 $ \n\n\
                 AMZN |  50  |  50 $  | - 100 $ |  600 $ |  400 $ \n\n\
                 NVDA |  20  |  245 $  | + 50 $ |  150 $ |  50 $ \n\n\
                "
    return orders
