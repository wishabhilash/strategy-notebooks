from dataclasses import dataclass, field, asdict
from typing import List
import pandas as pd
import copy
from tqdm.notebook import tqdm
import multiprocessing as mp
import optuna
import numpy as np
import datetime as dt
import os


class Bank:
    capital_deployed: float = 0.0
    capital_available: float = 0.0

    def __init__(self, initial_capital) -> None:
        self.capital_available = initial_capital
        self.snapshot = []

    def borrow(self, requested_capital: float = 0.0) -> tuple:
        requested_capital = round(requested_capital, 2)
        if requested_capital <= self.capital_available:
            self.capital_available = round(self.capital_available - requested_capital, 2)
            self.capital_deployed = round(self.capital_deployed + requested_capital, 2)
            self.snapshot.append({
                'type': 'borrow',
                'capital_deployed': self.capital_deployed,
                'capital_available': self.capital_available,
                'transaction': requested_capital
            })
            return requested_capital
        return 0.0
    
    def total_capital(self):
        return self.capital_available + self.capital_deployed
    
    def settle(self, deployed: float, pnl: float, tax: float):
        deployed = round(deployed, 2)
        self.capital_deployed = round(self.capital_deployed - deployed, 2)
        self.capital_available = round(self.capital_available + deployed + pnl, 2)
        self.snapshot.append({
            'type': 'settle',
            'capital_deployed': self.capital_deployed,
            'capital_available': self.capital_available,
            'transaction': deployed
        })

    def add_capital(self, amount):
        self.capital_available + amount
    
    def has_capital(self):
        return self.capital_available > 0


@dataclass
class Trade:
    entry_time: str
    entry_price: float
    quantity: int

@dataclass
class Position:
    stock: str
    entry_time: pd.Timestamp
    avg_entry_price: float = 0.0
    last_entry_price: float = 0.0
    quantity: int = 0
    exit_time: pd.Timestamp | None = None
    exit_price: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    pnl: float = 0.0
    tax: float = 0.0
    mtf_charge: float = 0.0
    leverage: int = 1
    mtf_rate_daily: float = 0.0192 / 100
    max_mtf_days: int = 100
    brokerage: float = 0.0

    def capital_deployed(self):
        return (self.avg_entry_price * self.quantity)/self.leverage

    def close(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        if self.quantity < 0:
            raise Exception("Quantity can't be negative")
        self.calculate_taxes()
        self.pnl = (exit_price - self.avg_entry_price) * self.quantity - self.tax

    def calculate_taxes(self):
        extry_taxes = 0
        for trade in self.trades:
            stt = abs(trade.quantity) * trade.entry_price * 0.025/100
            transaction_charges = (abs(trade.quantity) * trade.entry_price * 0.00322/100)
            gst = (stt + transaction_charges) * 18/100
            extry_taxes += stt + transaction_charges + gst

        transaction_charges = abs(self.quantity) * self.exit_price * 0.00322/100
        gst = transaction_charges * 18/100
        stamp_duty = abs(self.quantity) * self.exit_price * 0.003/100
        exit_taxes = transaction_charges + gst + stamp_duty

        # --- Add MTF charge ---
        holding_days = (self.exit_time - self.entry_time).days
        holding_days = min(holding_days, self.max_mtf_days)
        mtf_funded_amount = self.avg_entry_price * self.quantity * (self.leverage - 1) / self.leverage
        self.mtf_charge = round(mtf_funded_amount * self.mtf_rate_daily * holding_days, 2)

        # Brokerage
        if self.entry_time.date() == self.exit_time.date():
            buy_side_brokerage = min((self.quantity * self.avg_entry_price) * 0.03, self.brokerage)
            sell_side_brokerage = min((self.quantity * self.exit_price) * 0.03, self.brokerage)
            self.brokerage = round(buy_side_brokerage + sell_side_brokerage, 2)

        self.tax = round(extry_taxes + exit_taxes + self.mtf_charge + self.brokerage, 2)

    def rebalance_position(self):
        total_cost = 0
        total_qty = 0
        for t in self.trades:
            total_cost += t.entry_price * t.quantity
            total_qty += t.quantity
        if total_qty == 0:
            print(self)
        self.avg_entry_price = total_cost / total_qty
        self.quantity = total_qty
        self.last_entry_price = self.trades[-1].entry_price if len(self.trades) > 0 else 0

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        self.rebalance_position()

@dataclass
class PositionManager:
    bank: Bank
    active_positions: dict = field(default_factory=dict)
    closed_positions: list = field(default_factory=list)
    leverage: int = 1
    mtf_rate_daily: float = 0.0192 / 100
    brokerage: float = 0.0

    def new_position(self, stock, entry_time, entry_price, capital: float) -> Position | None:
        qty = 0

        capital = self.bank.borrow(capital)
        if capital <= 0:
            return

        try:
            qty = round(capital * self.leverage / entry_price)
        except Exception as e:
            return None
        
        if qty <= 0:
            return None

        position = Position(
            stock,
            entry_time,
            mtf_rate_daily=self.mtf_rate_daily,
            leverage=self.leverage,
            brokerage=self.brokerage
        )
        trade = Trade(entry_time, entry_price, qty)
        position.add_trade(trade)
        self.active_positions[position.stock] = position
        return position
    
    def close_position(self, stock, exit_time, exit_price):
        position = self.get_position(stock)
        if position is not None:
            position.close(exit_time, exit_price)
            self.closed_positions.append(position)
            self.active_positions[position.stock] = None
            self.bank.settle(position.capital_deployed(), position.pnl, position.tax)
        return position
    
    def add_trade_to_position(self, stock, entry_time, entry_price):
        position = self.get_position(stock)
        if position is None:
            return
        
        capital_required = entry_price * position.quantity
        capital = self.bank.borrow(capital_required)
        if capital <= 0:
            return
        
        trade = Trade(entry_time, entry_price, position.quantity)
        position.add_trade(trade)
   
    def get_position(self, stock) -> Position | None:
        return self.active_positions[stock] if stock in self.active_positions else None

    def get_active_positions(self):
        return [p for p in self.active_positions.values() if p is not None]

    def has_active_positions(self):
        return len(self.get_active_positions()) > 0
    
    def get_trades(self):
        if len(self.closed_positions) == 0:
            return pd.DataFrame()
        return pd.DataFrame([asdict(p) for p in self.closed_positions]).sort_values(['entry_time']).reset_index(drop=True)


def generate_tearsheet(initial_capital, pm: PositionManager, trades=None):
    if trades is None:
        trades = pm.get_trades()

    if len(trades) == 0:
        return {}, trades
    
    # Ensure entry_time and exit_time are datetime
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    trades['returns'] = (trades['pnl'] / (trades['avg_entry_price'] * trades['quantity']))

    # Total trades
    total_trades = len(trades)

    # Win rate
    win_trades = (trades['pnl'] > 0).sum()
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    # Profit
    total_profit = trades['pnl'].sum()
    avg_profit = trades[trades.pnl > 0].pnl.mean()
    avg_loss = trades[trades.pnl <= 0].pnl.mean()

    # Total tax
    total_tax = trades['tax'].sum()

    # Total brokerage
    total_brokerage = trades['brokerage'].sum()

    # MTF charge
    total_mtf_charge = trades['mtf_charge'].sum()

    # CAGR calculation
    start = trades['entry_time'].min()
    end = trades['exit_time'].max()
    years = (end - start).days / 365.25
    initial = initial_capital  # initial_capital from your code
    final = initial + total_profit
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else None

    # Active positions
    active_position_count = 0
    if pm is not None:
        active_position_count = sum([len(p.trades) for p in pm.get_active_positions() if p is not None])

    # Period
    period = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

    # Max holding period
    max_holding_period = (trades['exit_time'] - trades['entry_time']).max().days
    avg_holding_period = (trades['exit_time'] - trades['entry_time']).mean().days

    # Final capital
    final_capital = initial + total_profit


    # Calculate drawdown
    trades['cum_pnl'] = initial_capital + trades['pnl'].cumsum()

    trades['cum_max'] = trades['cum_pnl'].cummax()
    trades['drawdown'] = trades['cum_pnl'] - trades['cum_max']
    trades['drawdown_pct'] = trades['drawdown'] / trades['cum_max'] * 100
    avg_dd_perc = trades['drawdown_pct'].mean()
    max_drawdown = trades['drawdown'].min()
    max_drawdown_pct = abs(max_drawdown) / trades['cum_max'].max() * 100 if trades['cum_max'].max() != 0 else 0

    number_of_losses = len(trades[trades['pnl'] < 0])
    number_of_wins = len(trades[trades['pnl'] > 0])
    profit_factor = trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum()) if abs(trades[trades['pnl'] < 0]['pnl'].sum()) > 0 else None

    # Tearsheets summary
    tearsheet = {
        'Period': period,
        'Starting capital': initial_capital,
        'Final capital': final_capital,
        'Total Trades': total_trades,
        'Winners': number_of_wins,
        'Losers': number_of_losses,
        'Profit factor': profit_factor if profit_factor else "N/A",
        'Active Position Count': active_position_count,
        'Max holding period (days)': max_holding_period,
        'Avg holding period (days)': avg_holding_period,
        'Win Rate (%)': win_rate,
        'Total Profit': total_profit,
        'Avg Profit': avg_profit,
        'Avg Loss': avg_loss,
        'Total Brokerage': total_brokerage,
        'Total Tax': total_tax if total_tax else "N/A",
        'Total MTF': total_mtf_charge if total_mtf_charge else "N/A",
        'CAGR (%)': cagr if cagr else "N/A",
        'Max Drawdown': max_drawdown,
        'Max Drawdown (%)': max_drawdown_pct,
        'Avg Drawdown (%)': avg_dd_perc
    }
    
    return tearsheet, trades

def show_equity_curve(trades: pd.DataFrame):
    trades = trades.sort_values(['exit_time']).reset_index(drop=True)
    trades['cum_pnl'] = trades['pnl'].cumsum()
    trades.plot(x='exit_time', y='cum_pnl', title='Cumulative PnL vs Exit Time', figsize=(12, 6))


    # Yearly pnl plot
    trades['exit_year'] = trades['exit_time'].dt.year
    yearly_pnl = trades.groupby('exit_year')['pnl'].sum()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    yearly_pnl.plot(kind='bar', edgecolor='black')
    plt.title('Year-wise PnL')
    plt.xlabel('Year')
    plt.ylabel('Total PnL')
    plt.xticks(rotation=45)
    plt.show()

def get_perturb_params(params, CONSTANT_PARAMS, perturb_pct=0.10):
    if perturb_pct <= 0:
        return []
    
    perturb_params = []
    for key, value in params.items():
        if isinstance(value, (int, float)):
            # Perturb down
            params_copy = copy.deepcopy(params)
            params_copy[key] = value * (1 - perturb_pct)
            down_params = {**CONSTANT_PARAMS, **params_copy}
            perturb_params.append(down_params)
            
            # Perturb up
            params_copy = copy.deepcopy(params)
            params_copy[key] = value * (1 + perturb_pct)
            up_params = {**CONSTANT_PARAMS, **params_copy}
            perturb_params.append(up_params)
    return perturb_params

def backtest_worker(args):
    idx, backtest_func, df, params = args
    bank = Bank(params['initial_capital'])
    pm = PositionManager(bank, brokerage=params['brokerage'])
    tearsheet, trades = backtest_func(df.copy(), pm, params, show_pb=params['show_pb'])
    return (idx, tearsheet, trades)

def calc_perturb_loss3(base_tearsheet, base_trades, perturbed_results, min_trade_num):
    mo = calc_performance(base_tearsheet, base_trades, min_trade_num)
    if mo == 0:  # Avoid division issues
        return -float('inf')
    
    perturbed_metrics = []
    for result in perturbed_results:
        _, tearsheet, trades = result
        metric = calc_performance(tearsheet, trades, min_trade_num)
        perturbed_metrics.append(metric)

    mp = np.mean(perturbed_metrics) if len(perturbed_metrics) > 1 else 0
    vp = np.std(perturbed_metrics, ddof=1) if len(perturbed_metrics) > 1 else 0
    ratio = mp / mo
    penalty = mo * (ratio ** 1.5) * np.exp(-1.0 * vp)
    return penalty

def calc_perturb_loss2(base_tearsheet, base_trades, perturbed_results, min_trade_num):
    base_metric = calc_performance(base_tearsheet, base_trades, min_trade_num)
    perturbed_metrics = []
    for result in perturbed_results:
        _, tearsheet, trades = result
        metric = calc_performance(tearsheet, trades, min_trade_num)
        perturbed_metrics.append(metric)
    metrics = [base_metric] + perturbed_metrics

    stability_penalty = np.std(metrics, ddof=1) if len(metrics) > 1 else 0

    if np.isnan(stability_penalty):
        return -float('inf')
    return stability_penalty

def calc_perturb_loss(base_tearsheet, base_trades, perturbed_results, min_trade_num):
    base_metric = base_trades['returns'].mean() if len(base_trades) > 0 else 0
    perturbed_metrics = []
    for result in perturbed_results:
        trades = result[-1]
        metric = trades['returns'].mean() if len(trades) > 0 else 0
        perturbed_metrics.append(metric)
    metrics = [base_metric] + perturbed_metrics
    try:
        stability_penalty = np.std(metrics, ddof=1) if len(metrics) > 1 else 0
    except Exception as e:
        return -float('inf')
    return stability_penalty

def get_cagr(tearsheet):
    return tearsheet['CAGR (%)'] if tearsheet.get('CAGR (%)', "N/A") != "N/A" else 0

def get_avg_dd(tearsheet):
    return tearsheet['Avg Drawdown (%)'] if tearsheet.get('Avg Drawdown (%)', "N/A") != "N/A" else 0

def get_max_dd(tearsheet):
    return tearsheet['Max Drawdown (%)'] if tearsheet.get('Max Drawdown (%)', "N/A") != "N/A" else 0

def calc_performance(tearsheet, trades, min_trade_num=50):
    cagr = get_cagr(tearsheet)
    avg_dd = get_avg_dd(tearsheet)
    # avg_dd = get_max_dd(tearsheet)
    trade_count = len(trades)
    if trade_count < min_trade_num:
        print("Insufficient trades:", trade_count)
        return -float('inf')
    
    perf = (cagr / abs(avg_dd)) if avg_dd != 0 else -float('inf')
    # perf = cagr
    return perf

def objective(trial, backtest_func, df, CONSTANT_PARAMS, space, perturb_pct, min_trade_num):
    # Build param dict using Optuna's suggest API
    params = {}
    for k, v in space.items():
        if v['type'] == 'float':
            params[k] = trial.suggest_float(k, v['low'], v['high'], step=v['step'])
        elif v['type'] == 'int':
            params[k] = trial.suggest_int(k, v['low'], v['high'], v['step'], log=v.get('log', False))
        elif v['type'] == 'categorical':
            params[k] = trial.suggest_categorical(k, v['choices'])

    combined_params = {**CONSTANT_PARAMS, **params}
    _, main_tearsheet, main_trades = backtest_worker((0, backtest_func, df, combined_params))  # Warm-up run to avoid first-run overhead
    # loss = -calc_performance(main_tearsheet, main_trades, min_trade_num)
    # if np.isnan(loss) or np.isinf(loss):
    #     return float('inf')
    
    perturb_params = get_perturb_params(params, CONSTANT_PARAMS, perturb_pct)
    results = []

    with mp.Pool(processes=min(len(perturb_params), mp.cpu_count())) as pool:
        args = [(i, backtest_func, df, param_set) for i, param_set in enumerate(perturb_params)]
        for result in pool.map(backtest_worker, args):
            results.append(result)

    loss = -calc_perturb_loss3(main_tearsheet, main_trades, results, min_trade_num)
    # loss += perturbed_loss
    if np.isnan(loss) or np.isinf(loss):
        return float('inf')
    
    cagr = get_cagr(main_tearsheet)
    avg_dd = get_avg_dd(main_tearsheet)
    print(f"cagr: {cagr}, avg_dd: {avg_dd}, loss: {loss}")
    return loss

def optimize(backtest_func, df, space, CONSTANT_PARAMS, perturb_pct=0.1, max_evals=100, n_jobs=1, min_trade_num=50):
    pb = tqdm(total=max_evals, desc="Optimizing")
    def optuna_obj(trial):
        result = objective(trial, backtest_func, df.copy(), CONSTANT_PARAMS, space, perturb_pct, min_trade_num)
        pb.update(1)
        return result
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_obj, n_trials=max_evals, n_jobs=n_jobs)
    pb.close()
    return study

def grid_search(backtest_func, df, space, CONSTANT_PARAMS, n_jobs=1):
    from itertools import product

    space_ranges = {}
    for key in space.keys():
        space_ranges[key] = np.arange(space[key]['low'], space[key]['high'], space[key]['step'])

    param_combinations = [dict(zip(space_ranges.keys(), v)) for v in product(*space_ranges.values())]

    results = {}
    with mp.Pool(processes=min(n_jobs, mp.cpu_count())) as pool:
        args = [(idx, backtest_func, df, {**param_set, **CONSTANT_PARAMS}) for idx, param_set in enumerate(param_combinations)]
        for result in tqdm(pool.imap_unordered(backtest_worker, args), desc="Grid Search", total=len(args)):
            results[result[0]] = result


    rows = []
    for idx, param in enumerate(param_combinations):
        _, tearsheet, trades = results[idx]
        perf = calc_performance(tearsheet, trades, min_trade_num=CONSTANT_PARAMS.get('min_trade_num', 50))
        keys = sorted(param.keys())
        row = {keys[0]: param[keys[0]], keys[1]: param[keys[1]], "performance": perf}
        rows.append(row)

    df = pd.DataFrame(rows)

    # Get sorted unique parameter values
    x_unique = np.sort(df[keys[0]].unique())
    y_unique = np.sort(df[keys[1]].unique())

    # Pivot to create 2D grid of performance
    Z = df.pivot(index=keys[0], columns=keys[1], values='performance').values

    # Create meshgrid for X and Y
    X, Y = np.meshgrid(x_unique, y_unique, indexing='ij')

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel(keys[0])
    ax.set_ylabel(keys[1])
    ax.set_zlabel("Performance")
    ax.set_title("3D Surface Plot")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=7)
    plt.show()

def generate_wfo_splits(start_date: str, end_date: str, num_of_splits: int, 
                         insample_ratio_size: float, outsample_ratio_size: float) -> list:
    """
    Generate a walk-forward optimization schedule with overlapping IS and OOS periods.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        num_of_splits (int): Number of WFO splits (windows).
        insample_ratio_size (float): Proportion of each split for IS (e.g., 0.75 for 3 months in a 4-month split).
        outsample_ratio_size (float): Proportion of each split for OOS (e.g., 0.25 for 1 month in a 4-month split).
    
    Returns:
        list: List of dictionaries, each containing 'is_start', 'is_end', 'oos_start', 'oos_end' as strings.
    """
    # Parse input dates
    start = dt.datetime.strptime(start_date, '%Y-%m-%d')
    end = dt.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Validate inputs
    if start >= end:
        raise ValueError("start_date must be before end_date")
    if num_of_splits < 1:
        raise ValueError("num_of_splits must be at least 1")
    if insample_ratio_size <= 0 or outsample_ratio_size <= 0:
        raise ValueError("Ratios must be positive")
    if abs(insample_ratio_size + outsample_ratio_size - 1.0) > 0.01:
        raise ValueError("insample_ratio_size + outsample_ratio_size must sum to ~1.0")
    
    # Calculate r = IS/OOS ratio
    r = insample_ratio_size / outsample_ratio_size
    
    # Calculate OOS and IS days to fit the period
    total_days = (end - start).days
    oos_days = total_days / (r + num_of_splits)
    is_days = r * oos_days
    
    # Initialize result list
    schedule = []
    
    # Generate splits with rolling windows
    for i in range(num_of_splits):
        is_start = start + dt.timedelta(days=i * oos_days)
        is_end = is_start + dt.timedelta(days=is_days)
        oos_start = is_end
        oos_end = min(oos_start + dt.timedelta(days=oos_days), end)
        
        # Store as strings in 'YYYY-MM-DD' format
        schedule.append({
            'is_start': is_start.strftime('%Y-%m-%d'),
            'is_end': is_end.strftime('%Y-%m-%d'),
            'oos_start': oos_start.strftime('%Y-%m-%d'),
            'oos_end': oos_end.strftime('%Y-%m-%d')
        })
    
    return schedule

def walkforward_optimisation(backtest_func, df, params, space, num_of_splits, out_path, insample_ratio_size=0.75, outsample_ratio_size=0.25):
    all_trades = []
    all_tearsheets = []
    splits = generate_wfo_splits(
        start_date=params['start_date'], 
        end_date=params['end_date'], 
        num_of_splits=num_of_splits, 
        insample_ratio_size=insample_ratio_size, 
        outsample_ratio_size=outsample_ratio_size
    )
    for split in splits:
        filename = f"{out_path}/trades/{split['oos_start']}_{split['oos_end']}.csv"
        if os.path.exists(filename):
            all_trades.append(pd.read_csv(filename))
            print(f"File {filename} exists, skipping...")
            continue
        print(split)
        params = copy(params)
        params['start_date'] = split['is_start']
        params['end_date'] = split['is_end']
        for i in range(1, 4):
            study = optimize(backtest_func, df, space, params, perturb_pct=0.1, max_evals=i*30, n_jobs=2)
            if not np.isinf(study.best_value):
                break
            print("Retrying optimization with more evaluations...")

        if study.best_value > 0:
            print(f"All perturbations resulted in performance of loss, skipping this split - {split}.")
            continue
        params = copy(params)
        params['start_date'] = split['oos_start']
        params['end_date'] = split['oos_end']
        params.update(study.best_params)
        if len(all_tearsheets) > 0:
            params['initial_capital'] = all_tearsheets[-1]['Final capital']

        _, tearsheet, trades = backtest_worker((0, backtest_func, df, params))
        trades.to_csv(filename, index=False)
        all_trades.append(trades)
        all_tearsheets.append(tearsheet)


    all_trades_df = pd.concat(all_trades).reset_index(drop=True)
    final_tearsheet, final_trades = generate_tearsheet(params['initial_capital'], None, trades=all_trades_df)

    print(pd.DataFrame({
        "Metrics": final_tearsheet.keys(),
        "Values": final_tearsheet.values(),
    }))
    show_equity_curve(final_trades)

