from copy import copy
import os
import pandas as pd
from fyers_helper import prepare_data, load_stock_data
import datetime as dt
from lib import PositionManager, generate_tearsheet, grid_search, walkforward_optimisation
import talib as ta
from tqdm.notebook import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def backtest(_df, pm: PositionManager, params, show_pb=False):
    max_positions = params['max_positions']
    slippage = params['slippage']
    bb_period = int(params['bb_period'])
    bb_sd = round(params['bb_sd'], 2)
    start_date = params['start_date']
    end_date = params['end_date']
    initial_capital = params['initial_capital']

    print(f"Params: max_positions={max_positions}, bb_period={bb_period}, bb_sd={bb_sd}")

    def calc_bbands(group):
        group = group.copy()
        group['upper_band'], group['middle_band'], group['lower_band'] = ta.BBANDS(
            group['Close'],
            timeperiod=bb_period,     # window size for moving average
            nbdevup=bb_sd,         # number of stdevs for upper band
            nbdevdn=bb_sd,         # number of stdevs for lower band
            matype=0           # MA_Type.SMA (0), EMA (1), etc.
        )
        return group
    
    def calc_aroon(group):
        group = group.copy()
        group = group.set_index('Date')
        resampled = group.resample('4h').agg({
            'High': 'max',
            'Low': 'min'
        })
        aroon_up, aroon_down = ta.AROON(resampled.High, resampled.Low, timeperiod=14)
        group['aroon_up'] = aroon_up.reindex(group.index, method='ffill')
        group = group.reset_index()
        return group
    

    _df = _df[(_df.Date >= start_date) & (_df.Date < end_date)].reset_index()

    _df = _df.groupby('Stock', group_keys=False).apply(calc_bbands)
    _df = _df.groupby('Stock', group_keys=False).apply(calc_aroon)
    _df['close_below_lower_bb'] = np.where(_df.Close < _df.lower_band, 1, 0)
    _df['exit_price'] = _df.groupby('Stock')['Open'].shift(-1)
    _df['close_below_lower_bb_2'] = _df.groupby('Stock')['close_below_lower_bb'].shift(2)
    _df['close_below_lower_bb_1'] = _df.groupby('Stock')['close_below_lower_bb'].shift(1)


    signals = (
        _df[
            (_df.close_below_lower_bb_1 == 1) & 
            (_df.close_below_lower_bb_2 == 1) & 
            (_df.close_below_lower_bb == 1) #&
            # (_df.aroon_up > 90)
        ].reset_index(drop=True)
    )


    if show_pb:
        pb = tqdm(total=_df.Date.nunique(), desc="Backtesting...")

    for idx, group in _df.groupby('Date'):
        for position in pm.get_active_positions():
            stock = group[group.Stock == position.stock]
            if len(stock) < 1:
                continue
            try:
                if (stock.Close.iloc[0] >= stock.middle_band.iloc[0]):# or (stock.Close.iloc[0] - position.avg_entry_price)/position.avg_entry_price * 100 <= -5:
                    pm.close_position(position.stock, idx, stock.exit_price.iloc[0] * (1 - slippage))

            except Exception as e:
                print(f"stock.Close - {stock.Close}")
                print(f"stock - {stock}")
                print(f"position - {position}")
                print(f"idx - {idx}")
                print(f"group - {group}")
                raise e
        
        day_signals = signals[signals.Date == idx]
        for signal in day_signals.itertuples():
            if pm.get_position(signal.Stock) is None and len(pm.get_active_positions()) < max_positions:
                capital = round(pm.bank.capital_available / (max_positions - len(pm.get_active_positions())), 2)
                position = pm.new_position(signal.Stock, idx, signal.exit_price * (1 + slippage), capital)

        if show_pb:
            pb.update(1)

    if show_pb:
        pb.close()

    return generate_tearsheet(initial_capital, pm)


if __name__ == "__main__":
    interval = "5"

    nifty_df = pd.read_csv('/Users/abhilashnanda/code/strategy-notebooks/notebooks/nifty200.csv')
    tickers = [ f'NSE:{n}-EQ' for n in nifty_df.Symbol.tolist()]

    data_path = "/Users/abhilashnanda/code/strategy-notebooks/data5m"

    end_date = dt.datetime.now()
    start_date = dt.datetime(2015, 1, 1)

    file_paths = prepare_data(tickers, interval, start_date=start_date, end_date=end_date, path=data_path, update=False, overwrite=False)
    loaded_data = load_stock_data(file_paths, data_path, interval)

    df = pd.concat(loaded_data, names=["Stock", "Date"]).reset_index()


    CONSTANT_PARAMS = {
        'initial_capital': 500000,
        'max_positions': 10,
        'slippage': 0.001,
        'brokerage': 0,
        'min_trade_num': 50,
        'show_pb': False,
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        # 'start_date': '2025-02-15',
        # 'end_date': '2025-10-14',
    }

    space = {
        "bb_period": {"type": "int", "low": 10, "high": 100, 'step': 10},
        "bb_sd": {"type": "float", "low": 2, "high": 4.25, 'step': 0.25},
    }

    walkforward_optimisation(
        backtest,
        df,
        CONSTANT_PARAMS,
        space,
        num_of_splits=4,
        out_path='/Users/abhilashnanda/code/strategy-notebooks/notebooks'
    )
    
    grid_search(backtest, df, space, CONSTANT_PARAMS, n_jobs=5)