from dataclasses import dataclass, field
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
import statistics
import copy
from tqdm.notebook import tqdm
import multiprocessing as mp
import optuna
import uuid


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
        return pd.DataFrame([asdict(p) for p in self.closed_positions]).sort_values(['entry_time']).reset_index(drop=True)


def generate_tearsheet(initial_capital, pm: PositionManager, trades=None):
    if trades is None:
        trades = pm.get_trades()
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
            down_params = {**params_copy, **CONSTANT_PARAMS}
            perturb_params.append(down_params)
            
            # Perturb up
            params_copy = copy.deepcopy(params)
            params_copy[key] = value * (1 + perturb_pct)
            up_params = {**params_copy, **CONSTANT_PARAMS}
            perturb_params.append(up_params)
    return perturb_params

def backtest_worker(idx, q, backtest_func, df, params):
    bank = Bank(params['initial_capital'])
    pm = PositionManager(bank, brokerage=params['brokerage'])
    tearsheet, trades = backtest_func(df.copy(), pm, params, show_pb=params['show_pb'])

    filename = f'results/{uuid.uuid4().hex}_{idx}.csv'
    trades.to_csv(filename, index=False)
    q.put((idx, tearsheet, filename))
    with open(f'backtest_log_{idx}.txt', 'w') as f:
        f.write(str(params) + '\n')
        f.write('done')

def calc_perturb_loss2(base_metric, perturbed_results):
    perturbed_metrics = []
    for result in perturbed_results:
        metric = calc_performance(*result)
        perturbed_metrics.append(metric)
    metrics = [base_metric] + perturbed_metrics
    stability_penalty = statistics.stdev(metrics) if len(metrics) > 1 else 0
    return stability_penalty

def calc_perturb_loss(base_trades_file, perturbed_results):
    base_trades = pd.read_csv(base_trades_file)
    base_metric = base_trades['returns'].mean() if len(base_trades) > 0 else 0
    perturbed_metrics = []
    for result in perturbed_results:
        trades = pd.read_csv(result[2])
        metric = trades['returns'].mean() if len(trades) > 0 else 0
        perturbed_metrics.append(metric)
    metrics = [base_metric] + perturbed_metrics
    stability_penalty = statistics.stdev(metrics) if len(metrics) > 1 else 0
    return stability_penalty

def clean_results():
    import os
    import glob
    files = glob.glob('results/*.csv') + glob.glob('backtest_log_*.txt')
    for f in files:
        os.remove(f)

def get_cagr(tearsheet):
    return tearsheet['CAGR (%)'] if tearsheet['CAGR (%)'] != "N/A" else 0

def get_avg_dd(tearsheet):
    return tearsheet['Avg Drawdown (%)'] if tearsheet['Avg Drawdown (%)'] != "N/A" else 0

def calc_performance(idx, tearsheet, trades_file):
    trades = pd.read_csv(trades_file)
    cagr = get_cagr(tearsheet)
    avg_dd = get_avg_dd(tearsheet)
    trade_count = len(trades)
    if trade_count < 50:
        return float('inf')
    
    perf = (cagr / abs(avg_dd)) if avg_dd != 0 else float('inf')
    return perf

def objective(trial, backtest_func, df, CONSTANT_PARAMS, space, perturb_pct):
    # Build param dict using Optuna's suggest API
    params = {}
    for k, v in space.items():
        if v['type'] == 'float':
            params[k] = trial.suggest_float(k, v['low'], v['high'], step=v['step'])
        elif v['type'] == 'int':
            params[k] = trial.suggest_int(k, v['low'], v['high'], v['step'], log=v.get('log', False))
        elif v['type'] == 'categorical':
            params[k] = trial.suggest_categorical(k, v['choices'])

    combined_params = {**params, **CONSTANT_PARAMS}
    perturb_params = get_perturb_params(params, CONSTANT_PARAMS, perturb_pct)

    processes = []
    q = mp.Queue()
    all_params = [combined_params] + perturb_params

    for i, param_set in enumerate(all_params):
        p = mp.Process(target=backtest_worker, args=(i, q, backtest_func, df, param_set))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    results = [q.get() for _ in processes]
    results.sort(key=lambda x: x[0])  # Sort by index to maintain order
    
    main_result = results[0]
    
    loss = -calc_performance(*main_result)

    perturbed_loss = calc_perturb_loss(main_result[-1], results[1:])
    loss += perturbed_loss
    
    clean_results()
    cagr = get_cagr(main_result[1])
    avg_dd = get_avg_dd(main_result[1])
    print(f"cagr: {cagr}, avg_dd: {avg_dd}, loss: {loss}, perturbed_loss: {perturbed_loss}")
    return loss

def optimize(backtest_func, df, space, CONSTANT_PARAMS, perturb_pct=0.1, max_evals=100, n_jobs=1):
    pb = tqdm(total=max_evals, desc="Optimizing")
    def optuna_obj(trial):
        result = objective(trial, backtest_func, df.copy(), CONSTANT_PARAMS, space, perturb_pct)
        pb.update(1)
        return result
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_obj, n_trials=max_evals, n_jobs=n_jobs)
    pb.close()
    return study