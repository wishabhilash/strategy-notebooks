from dataclasses import dataclass, field
from typing import List
from uuid import uuid1
from tqdm.notebook import tqdm
from copy import copy
import pandas as pd


class Bank:
    initial_capital: float
    buckets: dict = {}
    ids: list = []
    residue: dict = {}
    snapshot: list = []
    bucket_usage_count: dict = {}

    def __init__(self, initial_capital, number_of_buckets = 10) -> None:
        self.residue.clear()
        self.snapshot.clear()
        self.bucket_usage_count.clear()
        self.initial_capital = initial_capital
        self.ids = [str(uuid1()).split('-')[0] for i in range(number_of_buckets)]
        self.buckets = dict(zip(self.ids, [initial_capital/number_of_buckets] * number_of_buckets))

    def borrow(self) -> tuple:
        if len(self.buckets) == 0:
            return None, None
        
        _df = pd.DataFrame(self.buckets.items(), columns=['key', 'amount']).sort_values('amount', ascending=True)
        key = _df.iloc[0].key
        bucket_amount = self.buckets.pop(key)

        self.bucket_usage_count[key] = self.bucket_usage_count.get(key, 0) + 1
        self._take_snapshot()
        return key, bucket_amount
    
    def save_residue(self, key, amount):
        self.residue[key] = amount

    def _take_snapshot(self):
        self.snapshot.append(copy(self.buckets))
    
    def settle(self, key, amount):
        if key not in self.ids:
            raise Exception("invalid key")
        residue = self.residue.pop(key)
        self.buckets[key] = amount + residue
        self._take_snapshot()


@dataclass
class Trade:
    capital_key: str
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
    leverage: int = 1
    mtf_rate_daily: float = 0.0192 / 100
    tp_perc: float = 3.0
    sl_perc: float = 1.0

    def exit_margin(self):
        return (self.exit_price * self.quantity)/self.leverage - self.tax

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
        holding_days = min(holding_days, 100)
        mtf_funded_amount = self.avg_entry_price * self.quantity * (self.leverage - 1) / self.leverage
        mtf_charge = mtf_funded_amount * self.mtf_rate_daily * holding_days

        self.tax = extry_taxes + exit_taxes + mtf_charge

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

    def new_position(self, stock, entry_time, entry_price):
        qty = 0
        while True:    
            key, capital = self.bank.borrow()
            if key is None:
                return None
            
            if capital <= 0:
                continue
            
            qty = int(capital * self.leverage / entry_price)
            
            if qty == 0:
                return None
            break

        position = Position(
            stock,
            entry_time,
            mtf_rate_daily=self.mtf_rate_daily,
            tp_perc=self.tp_perc,
            sl_perc=self.sl_perc,
            leverage=self.leverage
        )
        trade = Trade(key, entry_time, entry_price, qty)
        position.add_trade(trade)
        self.bank.save_residue(key, capital - (qty * trade.entry_price) / self.leverage)
        self.active_positions[position.stock] = position
        return position
    
    def close_position(self, stock, exit_time, exit_price):
        position = self.get_position(stock)
        if position is not None:
            position.close(exit_time, exit_price)
            self.closed_positions.append(position)
            self.active_positions[position.stock] = None

            exit_margin = position.exit_margin() / len(position.trades)
            for trade in position.trades:
                self.bank.settle(trade.capital_key, exit_margin)
        return position
    
    def add_trade_to_position(self, stock, entry_time, entry_price):
        position = self.get_position(stock)
        if position is None:
            return
        
        key, capital = self.bank.borrow()
        if key is None:
            return
        
        trade = Trade(key, entry_time, entry_price, position.quantity)
        position.add_trade(trade)
        
        self.bank.save_residue(key, capital - (position.quantity * trade.entry_price)/ self.leverage)
   
    def get_position(self, stock) -> Position | None:
        return self.active_positions[stock] if stock in self.active_positions else None

    def get_active_positions(self):
        return [p for p in self.active_positions.values() if p is not None]

    def has_active_positions(self):
        return len(self.get_active_positions()) > 0


