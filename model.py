import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from updater import download_binance_daily_data, download_binance_current_day_data, \
    download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY

# Data paths
binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data_binance(token, training_days, region):
    """Downloads historical Binance data for the specified token."""
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new Binance files")
    return files

def download_data_coingecko(token, training_days):
    """Downloads historical Coingecko data for the specified token."""
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new Coingecko files")
    return files

def download_data(token, training_days, region, data_provider):
    """
    Downloads historical data based on the specified data provider.

    Args:
        token: The cryptocurrency token symbol (e.g., "BTC").
        training_days: The number of days of historical data to download.
        region: The Binance region (e.g., "us").
        data_provider: The data provider to use ("binance" or "coingecko").

    Returns:
        A list of downloaded file paths.

    Raises:
        ValueError: If an unsupported data provider is specified.
    """
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files, data_provider):
    """
    Formats downloaded data into a CSV file for training.

    Args:
        files: A list of downloaded file paths.
        data_provider: The data provider used to download the data.
    """
    if not files:
        print("Already up to date")
        return

    price_df = pd.DataFrame()

    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)
            if not zip_file_path.endswith(".zip"):
                continue

            with ZipFile(zip_file_path) as myzip:
                with myzip.open(myzip.filelist[0]) as f:
                    header = None if f.readline().decode("utf-8").startswith("open_time") else 0
                df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
                df.columns = [
                    "start_time", "open", "high", "low", "close", "volume",
                    "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"
                ]
                df.index = pd.to_datetime(df["end_time"] + 1, unit="ms")
                df.index.name = "date"
                price_df = pd.concat([price_df, df])

    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = ["timestamp", "open", "high", "low", "close"]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

    price_df.sort_index().to_csv(training_price_data_path)
    print(f"Formatted data saved to {training_price_data_path}")

def load_frame(frame, timeframe):
    """Loads and preprocesses price data for training or inference."""
    print("Loading data...")
    df = frame.loc[:, ['open', 'high', 'low', 'close']].dropna()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def train_model(timeframe):
    """Trains the specified model on the formatted price data."""

    # Load and preprocess data
    price_data = pd.read_csv(training_price_data_path)
    df = load_frame(price_data, timeframe)

    print(df.tail())

    # Prepare training data
    y_train = df['close'].shift(-1).dropna().values
    X_train = df[:-1]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Define the model based on configuration
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    else:
        raise ValueError("Unsupported model")

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    """
    Generates an inference using the trained model.

    Args:
        token: The cryptocurrency token symbol.
        timeframe: The timeframe for the prediction (e.g., "1d").
        region: The Binance region (if using Binance as the data provider).
        data_provider: The data provider to use for current price data.

    Returns:
        The predicted price.
    """
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current price data and preprocess
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)

    print(X_new.tail())
    print(X_new.shape)

    # Make prediction
    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0]
