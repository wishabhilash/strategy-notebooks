import pandas as pd
from typing import Dict, List
from .fyers_util import FyersSession, Historical
import json
import os
from tqdm import tqdm
import datetime as dt


def prepare_data(tickers: List[str], interval: str, start_date: dt.datetime, 
                 end_date: dt.datetime, path: str, overwrite: bool = False, update=False, show_progress = True) -> Dict[str, pd.DataFrame]:
    file_paths = {}
    config = json.loads(os.environ['FYERS_CONFIG'])
    
    
    fs: FyersSession = FyersSession(
        config['CLIENT_ID'],
        config['SECRET_KEY'],
        config['USERNAME'],
        config['TOTP_KEY'],
        config['PIN'],
    )
    h: Historical = None

    if show_progress:
        pb = tqdm(total=len(tickers), desc='Downloading data', unit='ticker')
    
    for ticker in tickers:
        file_path = f'{path}/{ticker}-{interval}m-EQ.parquet'
        if os.path.exists(file_path):
            if not overwrite:
                file_paths[ticker] = file_path
                if not update:
                    pb.update(1)
                    continue
                file_paths[ticker] = file_path
                existing_data = pd.read_parquet(file_path)
                new_date = existing_data.tail(1).iloc[0].name.date() + dt.timedelta(days=1)
                start_date = dt.datetime.combine(new_date, dt.datetime.min.time())
            else:
                os.remove(file_path)

        if not h:
            token = None
            now = dt.datetime.now()
            token_file_name = f"fyers-token-{now:%Y-%m-%d}.txt"
            if os.path.exists(token_file_name):
                with open(token_file_name, 'r') as f:
                    token = f.read()
                    fs.token = token
            h = Historical(fs)
            with open(token_file_name, 'w') as f:
                f.write(h._client.token)
        
        try:
            data = h.historical(f'{ticker}', interval, start_date, end_date)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            pb.update(1)
            continue
        data = data.rename(columns={
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'open': 'Open',
            'volume': 'Volume',
        }).drop(columns=['datetime'])
        data.index.name = 'Date'

        if not overwrite and os.path.exists(file_path):
            existing_data = pd.read_parquet(file_path)
            combined_data = pd.concat([existing_data, data]).drop_duplicates().sort_index()
            combined_data.to_parquet(file_path)
        else:
            data.to_parquet(file_path)
        file_paths[ticker] = file_path
        if show_progress:
            pb.update(1)
    
    if show_progress:
        pb.close()
    return file_paths

def load_stock_data(stock_list, path: str, interval: str):
    stocks_data = {}
    pb = tqdm(total=len(stock_list), desc=f"Loading {interval}min data")
    for stock in stock_list:
        try:
            file_path = f'{path}/{stock}-{interval}m-EQ.parquet'
            if not os.path.exists(file_path):
                file_path = f'{path}/{stock}-{interval}m-BE.parquet'
            stock_data = pd.read_parquet(file_path)
            stock_data.index = stock_data.index.tz_localize(None)
            stocks_data[stock] = stock_data
        except FileNotFoundError:
            print(f"File for {stock} not found.")
            continue
        pb.update(1)
    pb.close()
    return stocks_data

def resample(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    resampled_data = {}
    pb = tqdm(total=len(data), desc=f"Resampling to {freq} from 1m")
    for stock, data in data.items():
        resampled_data[stock] = data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        resampled_data[stock] = resampled_data[stock].dropna()
        pb.update(1)
    pb.close()
    return resampled_data