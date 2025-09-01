from dataclasses import dataclass, field
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict


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
        self.capital_available = round(self.capital_available + deployed + pnl - tax, 2)
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
    tp_perc: float = 3.0
    sl_perc: float = 1.0
    max_mtf_days: int = 100
    brokerage: float = 0.0

    def capital_deployed(self):
        return (self.avg_entry_price * self.quantity)/self.leverage

    def close(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        if self.quantity < 0:
            print(self)
            raise Exception("Quantity can't be negative")
        self.pnl = (exit_price - self.avg_entry_price) * self.quantity
        self.calculate_taxes()

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
            buy_side_brokerage = min((self.quantity * self.avg_entry_price) * 0.03, 20)
            sell_side_brokerage = min((self.quantity * self.exit_price) * 0.03, 20)
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
        self.tp = self.avg_entry_price * (1 + self.tp_perc / 100)
        self.sl = self.avg_entry_price * (1 - self.sl_perc / 100)
        self.last_entry_price = self.trades[-1].entry_price if len(self.trades) > 0 else 0

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        self.rebalance_position()

@dataclass
class PositionManager:
    bank: Bank
    tp_perc: float = 1
    sl_perc: float = 1
    active_positions: dict = field(default_factory=dict)
    closed_positions: list = field(default_factory=list)
    leverage: int = 1
    mtf_rate_daily: float = 0.0192 / 100

    def new_position(self, stock, entry_time, entry_price, capital: float) -> Position | None:
        qty = 0

        capital = self.bank.borrow(capital)
        if capital <= 0:
            print(f"capital not available - {capital}")
            return

        try:
            qty = round(capital * self.leverage / entry_price)
        except Exception as e:
            print(e)
            print(capital, entry_price, self.leverage)
            return None
        
        if qty <= 0:
            print(f"invalid quantity - {qty}")
            return None

        position = Position(
            stock,
            entry_time,
            mtf_rate_daily=self.mtf_rate_daily,
            tp_perc=self.tp_perc,
            sl_perc=self.sl_perc,
            leverage=self.leverage
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
            print(f"capital not available - {capital}")
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


def print_tearsheet(initial_capital, pm: PositionManager, trades: pd.DataFrame):
    # Ensure entry_time and exit_time are datetime
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])

    # Total trades
    total_trades = len(trades)

    # Win rate
    win_trades = (trades['pnl'] > 0).sum()
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    # Total profit
    total_profit = trades['pnl'].sum() - trades['tax'].sum()

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
    trades['cum_pnl'] = trades['pnl'].cumsum()

    trades['cum_max'] = trades['cum_pnl'].cummax()
    trades['drawdown'] = trades['cum_pnl'] - trades['cum_max']
    max_drawdown = trades['drawdown'].min()
    max_drawdown_pct = abs(max_drawdown) / trades['cum_max'].max() * 100 if trades['cum_max'].max() != 0 else 0

    number_of_losses = len(trades[trades['pnl'] < 0])
    number_of_wins = len(trades[trades['pnl'] > 0])
    profit_factor = trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum()) if abs(trades[trades['pnl'] < 0]['pnl'].sum()) > 0 else None

    # Tearsheets summary
    tearsheet = pd.DataFrame({
        'Metric': [
            'Period',
            'Starting capital',
            'Final capital',
            'Total Trades',
            'Winners',
            'Losers',
            'Profit factor',
            'Active Position Count',
            'Max holding period (days)',
            'Avg holding period (days)',
            'Win Rate (%)',
            'Total Profit',
            'Total Brokerage',
            'Total Tax',
            'Total MTF',
            'CAGR (%)',
            'Max Drawdown (%)'
        ],
        'Value': [
            period,
            f"{initial_capital:.2f}",
            f"{final_capital:.2f}",
            f"{total_trades:,}",
            f"{number_of_wins:,}",
            f"{number_of_losses:,}",
            f"{profit_factor:.2f}" if profit_factor else "N/A",
            f"{active_position_count:,}",
            f"{max_holding_period:,}",
            f"{avg_holding_period:,}",
            f"{win_rate:.2f}",
            f"{total_profit:,.2f}",
            f"{total_brokerage:,.2f}",
            f"{total_tax:.2f}" if total_tax else "N/A",
            f"{total_mtf_charge:.2f}" if total_mtf_charge else "N/A",
            f"{cagr:.2f}" if cagr else "N/A",
            f"{max_drawdown_pct:,.2f}"
        ]
    })

    print(tearsheet)

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
