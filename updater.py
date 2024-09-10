import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
from datetime import date, timedelta
import pathlib
import time

# Configure retries for handling network issues and rate limiting
retry_strategy = Retry(
    total=3,  # Maximum number of retries
    backoff_factor=1,  # Exponential backoff between retries (1, 2, 4 seconds)
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

files = []  # List to store downloaded file paths

def download_url(url, download_path, name=None):
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url: The URL of the file to download.
        download_path: The directory where the file should be saved.
        name: (Optional) The desired filename for the downloaded file.
              If not provided, the filename will be extracted from the URL.
    """
    try:
        global files

        # Determine the filename
        file_name = os.path.join(download_path, name) if name else os.path.join(download_path, os.path.basename(url))

        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_name)
        os.makedirs(dir_path, exist_ok=True)

        # Skip download if the file already exists
        if os.path.exists(file_name):
            print(f"{file_name} already exists. Skipping download.")
            return

        # Make the request using the session with retry strategy
        response = http.get(url, timeout=10)  # Set a timeout to prevent hanging requests
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the file
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {url} to {file_name}")
        files.append(file_name)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def daterange(start_date, end_date):
    """
    Generates a range of dates between the start and end dates (inclusive).

    Args:
        start_date: The start date (datetime.date object).
        end_date: The end date (datetime.date object).

    Yields:
        datetime.date objects representing each date in the range.
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def download_binance_daily_data(pair, training_days, region, download_path):
    """
    Downloads daily kline data from Binance for the specified pair and time range.

    Args:
        pair: The trading pair (e.g., "BTCUSDT").
        training_days: The number of days of historical data to download.
        region: The Binance region (e.g., "us").
        download_path: The directory where the downloaded files should be saved.

    Returns:
        A list of downloaded file paths.
    """
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today()
    start_date = end_date - timedelta(days=int(training_days))

    global files
    files = []  # Reset the files list for each download

    with ThreadPoolExecutor() as executor:
        print(f"Downloading Binance daily data for {pair} from {start_date} to {end_date}")
        for single_date in daterange(start_date, end_date):
            url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date.strftime('%Y-%m-%d')}.zip"
            executor.submit(download_url, url, download_path)
        # Wait for all downloads to complete
        executor.shutdown(wait=True)

    return files

def download_binance_current_day_data(pair, region):
    """
    Downloads kline data for the current day from the Binance API.

    Args:
        pair: The trading pair (e.g., "BTCUSDT").
        region: The Binance region (e.g., "us").

    Returns:
        A pandas DataFrame containing the downloaded kline data.
    """
    limit = 1000
    base_url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}'

    response = http.get(base_url)
    response.raise_for_status()

    data = response.json()
    columns = ['start_time', 'open', 'high', 'low', 'close', 'volume', 'end_time',
               'volume_usd', 'n_trades', 'taker_volume', 'taker_volume_usd', 'ignore']

    df = pd.DataFrame(data, columns=columns)
    df['date'] = pd.to_datetime(df['end_time'] + 1, unit='ms') 
    df.drop(columns=["end_time", "ignore"], inplace=True)
    df.set_index("date", inplace=True)
    # Convert relevant columns to numeric
    df = df.astype({
        'volume': float, 
        'taker_volume': float, 
        'open': float, 
        'high': float, 
        'low': float, 
        'close': float
    })

    return df.sort_index()

def get_coingecko_coin_id(token):
    """
    Maps a token symbol to its corresponding Coingecko coin ID.

    Args:
        token: The cryptocurrency token symbol (e.g., "BTC").

    Returns:
        The Coingecko coin ID.

    Raises:
        ValueError: If the token is not supported.
    """
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum',
        # Add more tokens here
    }

    token = token.upper()
    if token in token_map:
        return token_map[token]
    else:
        raise ValueError("Unsupported token")

def download_coingecko_data(token, training_days, download_path, CG_API_KEY):
    """
    Downloads historical OHLC data from Coingecko for the specified token and time range.

    Args:
        token: The cryptocurrency token symbol (e.g., "BTC").
        training_days: The number of days of historical data to download.
        download_path: The directory where the downloaded file should be saved.
        CG_API_KEY: The Coingecko API key.

    Returns:
        A list containing the downloaded file path.
    """

    # Map training_days to Coingecko's supported durations
    days_map = {
        7: 7,
        14: 14,
        30: 30,
        90: 90,
        180: 180,
        365: 365,
    }
    days = days_map.get(training_days, "max") 

    coin_id = get_coingecko_coin_id(token)

    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}&api_key={CG_API_KEY}'

    global files
    files = []  # Reset the files list for each download
