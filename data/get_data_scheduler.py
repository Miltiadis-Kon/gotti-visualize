import schedule
import time
from datetime import datetime
import pytz


def end_of_market_function():
    # This function will be called at the end of each market day
    print(f"Market closed. Running end-of-day function at {datetime.now(pytz.timezone('US/Eastern'))}")
    # Add your daily tasks here, such as:
    # - Fetch & Proccess stock data
    get_daily_data(tickers)
    # - Save or upload the data
    

def schedule_market_close():
    # Set the timezone to US/Eastern (NYSE timezone)
    eastern = pytz.timezone('US/Eastern')
    
    # Schedule the function to run at 4:00 PM Eastern Time every weekday
    schedule.every().monday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().tuesday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().wednesday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().thursday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().friday.at("16:00").timezone(eastern).do(end_of_market_function)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

# Fetch and process stock data
def get_daily_data(tickers):
    for ticker in tickers:
        data = get_ticker(ticker, '1mo', '15m')
        data = detect_breakout(data)
        plot_breakout_signal(data)



if __name__ == "__main__":
    schedule_market_close()