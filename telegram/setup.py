#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""Simple inline keyboard bot with multiple CallbackQueryHandlers.

This Bot uses the Application class to handle the bot.
First, a few callback functions are defined as callback query handler. Then, those functions are
passed to the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Example of a bot that uses inline keyboard that has multiple CallbackQueryHandlers arranged in a
ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line to stop the bot.
"""
import asyncio
import logging
import threading

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
)
import websockets

import callbacks as cb


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Stages
START_ROUTES, END_ROUTES = range(2)
# Callback data
POSITIONS, ORDERS, HISTORY, STRATEGIES, SETTINGS,HOME = range(6)

subscribe_to_new_orders = True

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send message on `/start`."""
    user = update.message.from_user
    logger.info("User %s started the conversation.", user.first_name)
    # Build InlineKeyboard where each button has a displayed text
    # and a string as callback_data
    # The keyboard is a list of button rows, where each row is in turn
    # a list (hence `[[...]]`).
    keyboard = [
        [
            InlineKeyboardButton("  Active Positions", callback_data=str(POSITIONS)),
            InlineKeyboardButton("  Pending Orders", callback_data=str(ORDERS)),
        ],
        [InlineKeyboardButton("  🔍  History", callback_data=str(HISTORY))],
        [InlineKeyboardButton("  🧠  Strategies", callback_data=str(STRATEGIES))],
        [InlineKeyboardButton("  ⚙️  Settings", callback_data=str(SETTINGS))],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str = await cb.get_bot_info()
    
    msg = cb.get_into_msg(weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str)
    
    # Send message with text and appended InlineKeyboard
    await update.message.reply_text(msg, reply_markup=reply_markup)
#    await update.message.reply_text(msg, reply_markup=reply_markup)
    # Tell ConversationHandler that we're in state `FIRST` now
    return START_ROUTES


async def start_over(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Prompt same text & keyboard as `start` does but not as new message"""
    # Get CallbackQuery from Update
    query = update.callback_query
    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    keyboard = [
        [
            InlineKeyboardButton("  Active Positions", callback_data=str(POSITIONS)),
            InlineKeyboardButton("  Pending Orders", callback_data=str(ORDERS)),
        ],
        [InlineKeyboardButton("  🔍  History", callback_data=str(HISTORY))],
        [InlineKeyboardButton("  🧠  Strategies", callback_data=str(STRATEGIES))],
        [InlineKeyboardButton("  ⚙️  Settings", callback_data=str(SETTINGS))],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str = await cb.get_bot_info()
    
    msg = cb.get_into_msg(weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str)
    # Send message with text and appended InlineKeyboard
    await query.message(msg, reply_markup=reply_markup)
    return START_ROUTES


async def positions_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [
            InlineKeyboardButton(" 👈   Back", callback_data=str(HOME)),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str = await cb.get_bot_info()
    postions = await cb.get_active_positions()
    msg = f"{weekly_pnl_str}\n{winrate_str}\n\nActive Positions:\n\n{postions}\n\n"
    await query.edit_message_text(
        text=msg, reply_markup=reply_markup
    )
    return START_ROUTES


async def orders_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [
            InlineKeyboardButton(" 👈   Back", callback_data=str(HOME)),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    weekly_pnl_str,winrate_str,trades_str,positions_str,orders_str = await cb.get_bot_info()
    orders = await cb.get_active_orders()
    msg = f"{weekly_pnl_str}\n{winrate_str}\n\nActive Orders:\n\n{orders}\n\n"
    await query.edit_message_text(
        text=msg, reply_markup=reply_markup
    )
    return START_ROUTES


async def history_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show new choice of buttons. This is the end point of the conversation."""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton(" 🔗   Show Complete History", callback_data=str(STRATEGIES))],
        [InlineKeyboardButton(" 👈   Back", callback_data=str(HOME))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        text="Third CallbackQueryHandler. Do want to start over?", reply_markup=reply_markup
    )
    # Transfer to conversation state `SECOND`
    return START_ROUTES


async def strategies_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton(" 👈   Back", callback_data=str(HOME))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        text="Fourth CallbackQueryHandler, Choose a route", reply_markup=reply_markup
    )
    return START_ROUTES

async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    await query.answer()
    keyboard = [
        [
            InlineKeyboardButton("Orders", callback_data=str(ORDERS)),
            InlineKeyboardButton("History", callback_data=str(HISTORY)),
        ],
        [InlineKeyboardButton(" 👈   Back", callback_data=str(HOME))]
        
        
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        text="Fourth CallbackQueryHandler, Choose a route", reply_markup=reply_markup
    )
    return START_ROUTES


async def back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Returns `ConversationHandler.END`, which tells the
    ConversationHandler that the conversation is over.
    """
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text="See you next time!")
    return ConversationHandler.END

async def send_new_order_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when it is recieved from the ws"""
    if not subscribe_to_new_orders:
        return
    query = update.callback_query
    await query.answer()
    order = cb.last_order
    if order:
        msg = f"New Order: {order['symbol']} {order['quantity']} @ {order['price']}\nSL: {order['stop_loss_price']} TP: {order['take_profit_price']}"
        await query.edit_message_text(text=msg)
    else:
        await query.edit_message_text(text="No new orders")
    return START_ROUTES

async def listen():
    uri = "ws://localhost:5055"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server")
        try:
            async for message in websocket:
                cb.set_last_order(message)
        except websockets.ConnectionClosed:
            print("Connection closed")


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("7924089058:AAHfnR2vcgBq3LRyKVKu4XdqfRu0ofQMI40").build()
    
    # Setup conversation handler with the states FIRST and SECOND
    # Use the pattern parameter to pass CallbackQueries with specific
    # data pattern to the corresponding handlers.
    # ^ means "start of line/string"
    # $ means "end of line/string"
    # So ^ABC$ will only allow 'ABC'
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            START_ROUTES: [
                CallbackQueryHandler(positions_callback, pattern="^" + str(POSITIONS) + "$"),
                CallbackQueryHandler(orders_callback, pattern="^" + str(ORDERS) + "$"),
                CallbackQueryHandler(history_callback, pattern="^" + str(HISTORY) + "$"),
                CallbackQueryHandler(strategies_callback, pattern="^" + str(STRATEGIES) + "$"),
                CallbackQueryHandler(settings_callback, pattern="^" + str(SETTINGS) + "$"),
                CallbackQueryHandler(start_over, pattern="^" + str(HOME) + "$"),

            ],
            END_ROUTES: [
                CallbackQueryHandler(start_over, pattern="^" + str(POSITIONS) + "$"),
                CallbackQueryHandler(back, pattern="^" + str(ORDERS) + "$"),
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    # Add ConversationHandler to application that will be used for handling updates
    application.add_handler(conv_handler)

    # Add a handler to send new order message
    application.add_handler(CommandHandler("new_order", send_new_order_msg))
            
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
