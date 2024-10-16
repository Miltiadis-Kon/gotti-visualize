import io
import time
from datetime import datetime
import pytz
import asyncio
import os
import sys
import pandas as pd
import schedule
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.ticker_utils import get_ticker, screen_all, get_screened_tickers
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
from dotenv import load_dotenv
import logging


def auth():
    load_dotenv()
    # Define the Google Drive API scopes and service account file path
    SCOPES = ['https://www.googleapis.com/auth/drive']
    # Decode the JSON string from the environment variable
    google_json = os.getenv('GOOGLE_JSON')
    if not google_json:
        raise ValueError("The GOOGLE_JSON environment variable is not set.")
    
    credentials_info = json.loads(google_json)
    
    # Create credentials using the decoded JSON
    credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
    
    # Build the Google Drive service
    drive_service = build('drive', 'v3', credentials=credentials)
    return drive_service

def update_db(param):
    drive_service = auth()
    ticker = param['symbol']
    data = param['data']
    print(f"Updating database for {ticker} at {datetime.now(pytz.timezone('US/Eastern'))}...")
    # Convert data to CSV format
    csv_data = data.to_csv(index=False)
    # Save CSV to a temporary file
    temp_file = f"{ticker}_data.csv"
    with open(temp_file, 'w') as file:
        file.write(csv_data)
    # Upload the file to Google Drive
    file_metadata = {
        'name': temp_file,
    }
    media = drive_service.files().create(body=file_metadata, media_body=temp_file).execute()
    print(f"File ID: {media.get('id')}")
    # Delete the temporary file
    os.remove(temp_file)
    print(f"Database updated for {ticker} at {datetime.now(pytz.timezone('US/Eastern'))}.")
    return True
    

def end_of_market_function():
    # This function will be called at the end of each market day
    print(f"Market closed. Running end-of-day function at {datetime.now(pytz.timezone('US/Eastern'))}")
    # Add your daily tasks here:
    screener = screen_all('data/nasdaq_screener_all.csv')
    tickers = get_screened_tickers(screener)
    background_tasks = set()  # Create a set to store the background tasks

    for ticker in tickers:
        data = get_ticker(ticker, '5d', '5m')  # Get the data from Yahoo Finance
        # data = post_process(data)  # Add your post-processing logic here (breakout pivot points etc..)
        param = {'symbol': ticker, 'data': data}
        update_db(param=param)
    

def schedule_market_close():
    # Set the timezone to US/Eastern (NYSE timezone)
    eastern = pytz.timezone('US/Eastern') 
    # Schedule the function to run at 4:00 PM Eastern Time every weekday
    '''
    schedule.every().monday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().tuesday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().wednesday.at("16:00").timezone(eastern).do(end_of_market_function)
    schedule.every().thursday.at("16:00").timezone(eastern).do(end_of_market_function)
    '''
    schedule.every().friday.at("22:39").do(end_of_market_function)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)


def test():
    ticker = "AAPL"  # Apple Inc.
    data = get_ticker(ticker, '5d', '5m')  # Get the data from Yahoo Finance
    param = {'symbol': ticker, 'data': data}
    update_db(param=param)


def download_file(file_id, destination_path):
    """Download a file from Google Drive by its ID."""
    drive_service = auth()
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")


def delete_files(file_or_folder_id):
    """Delete a file or folder in Google Drive by ID."""
    drive_service = auth()
    try:
        drive_service.files().delete(fileId=file_or_folder_id).execute()
        print(f"Successfully deleted file/folder with ID: {file_or_folder_id}")
    except Exception as e:
        print(f"Error deleting file/folder with ID: {file_or_folder_id}")
        print(f"Error details: {str(e)}")


def list_folder(parent_folder_id=None, delete=False):
    drive_service = auth()
    """List folders and files in Google Drive."""
    results = drive_service.files().list(
        q=f"'{parent_folder_id}' in parents and trashed=false" if parent_folder_id else None,
        pageSize=1000,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print("No folders or files found in Google Drive.")
    else:
        print("Folders and files in Google Drive:")
        for item in items:
            print(f"Name: {item['name']}, ID: {item['id']}, Type: {item['mimeType']}")
            if delete:
                delete_files(item['id'])


def create_folder(folder_name, parent_folder_id=None):
    """Create a folder in Google Drive and return its ID."""
    drive_service = auth()
    folder_metadata = {
        'name': folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        'parents': [parent_folder_id] if parent_folder_id else []
    }

    created_folder = drive_service.files().create(
        body=folder_metadata,
        fields='id'
    ).execute()

    print(f'Created Folder ID: {created_folder["id"]}')
    return created_folder["id"]

if __name__ == "__main__":
    try:
        print("Scheduler started. Waiting for market close...")
        schedule_market_close()
        #test()
    except (KeyboardInterrupt, SystemExit):
        pass