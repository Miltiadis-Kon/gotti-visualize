"""
Trade Tracker Module

Tracks all trades during backtesting/live trading with full details for reporting and visualization.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import os


@dataclass
class Trade:
    """Represents a single trade with all relevant details."""
    trade_id: int
    ticker: str
    trade_type: str  # "BUY" or "SELL"
    date_executed: datetime
    entry_price: float
    quantity: int
    take_profit: float
    stop_loss: float
    support_level: float  # The support level that triggered entry
    resistance_level: float  # Target resistance level
    
    # Filled when trade closes
    date_completed: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "TP", "SL", "MANUAL", "EOD"
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.date_completed is None
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl is not None and self.pnl > 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get trade duration in hours."""
        if self.date_completed and self.date_executed:
            delta = self.date_completed - self.date_executed
            return delta.total_seconds() / 3600
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        d = asdict(self)
        # Convert datetime to ISO format string
        if d['date_executed']:
            d['date_executed'] = d['date_executed'].isoformat()
        if d['date_completed']:
            d['date_completed'] = d['date_completed'].isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create Trade from dictionary."""
        # Convert ISO strings back to datetime
        if data.get('date_executed') and isinstance(data['date_executed'], str):
            data['date_executed'] = datetime.fromisoformat(data['date_executed'])
        if data.get('date_completed') and isinstance(data['date_completed'], str):
            data['date_completed'] = datetime.fromisoformat(data['date_completed'])
        return cls(**data)


class TradeTracker:
    """
    Tracks all trades during a strategy run.
    
    Usage:
        tracker = TradeTracker(ticker="NVDA")
        
        # On entry
        trade_id = tracker.open_trade(
            date=datetime.now(),
            entry_price=180.50,
            quantity=10,
            take_profit=190.00,
            stop_loss=175.00,
            support_level=179.00,
            resistance_level=192.00
        )
        
        # On exit
        tracker.close_trade(
            trade_id=trade_id,
            date=datetime.now(),
            exit_price=189.50,
            exit_reason="TP"
        )
        
        # Get results
        df = tracker.to_dataframe()
        tracker.save_to_json("trades.json")
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.trades: List[Trade] = []
        self._next_id = 1
        self._total_pnl = 0.0
    
    def open_trade(
        self,
        date: datetime,
        entry_price: float,
        quantity: int,
        take_profit: float,
        stop_loss: float,
        support_level: float,
        resistance_level: float,
        trade_type: str = "BUY"
    ) -> int:
        """
        Record a new trade entry.
        
        Returns:
            trade_id: Unique identifier for this trade
        """
        trade = Trade(
            trade_id=self._next_id,
            ticker=self.ticker,
            trade_type=trade_type,
            date_executed=date,
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            support_level=support_level,
            resistance_level=resistance_level
        )
        self.trades.append(trade)
        self._next_id += 1
        return trade.trade_id
    
    def close_trade(
        self,
        trade_id: int,
        date: datetime,
        exit_price: float,
        exit_reason: str = "MANUAL"
    ) -> Optional[Trade]:
        """
        Record a trade exit.
        
        Parameters:
            trade_id: ID of the trade to close
            date: Exit datetime
            exit_price: Price at which position was closed
            exit_reason: "TP", "SL", "MANUAL", "EOD"
            
        Returns:
            The closed Trade object, or None if trade_id not found
        """
        trade = self.get_trade(trade_id)
        if trade is None:
            return None
        
        trade.date_completed = date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate PnL
        if trade.trade_type == "BUY":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
            trade.pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:  # SELL (short)
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
            trade.pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        self._total_pnl += trade.pnl
        return trade
    
    def get_trade(self, trade_id: int) -> Optional[Trade]:
        """Get a trade by ID."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None
    
    def get_open_trade(self) -> Optional[Trade]:
        """Get the currently open trade (if any)."""
        for trade in self.trades:
            if trade.is_open:
                return trade
        return None
    
    @property
    def total_pnl(self) -> float:
        """Get total PnL across all closed trades."""
        return self._total_pnl
    
    @property
    def trade_count(self) -> int:
        """Get total number of trades."""
        return len(self.trades)
    
    @property
    def closed_trades(self) -> List[Trade]:
        """Get all closed trades."""
        return [t for t in self.trades if not t.is_open]
    
    @property
    def open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [t for t in self.trades if t.is_open]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate trading statistics.
        
        Returns:
            Dictionary with:
            - total_trades, winning_trades, losing_trades
            - win_rate, total_pnl
            - avg_win, avg_loss, largest_win, largest_loss
            - avg_trade_duration_hours
        """
        closed = self.closed_trades
        
        if not closed:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_trade_duration_hours': 0.0
            }
        
        winners = [t for t in closed if t.is_winner]
        losers = [t for t in closed if not t.is_winner]
        
        win_pnls = [t.pnl for t in winners if t.pnl]
        loss_pnls = [t.pnl for t in losers if t.pnl]
        durations = [t.duration for t in closed if t.duration is not None]
        
        return {
            'total_trades': len(closed),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(closed) * 100 if closed else 0.0,
            'total_pnl': self._total_pnl,
            'avg_win': sum(win_pnls) / len(win_pnls) if win_pnls else 0.0,
            'avg_loss': sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0,
            'largest_win': max(win_pnls) if win_pnls else 0.0,
            'largest_loss': min(loss_pnls) if loss_pnls else 0.0,
            'avg_trade_duration_hours': sum(durations) / len(durations) if durations else 0.0
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert trades to DataFrame with required columns.
        
        Columns:
        - date_executed, ticker, type, date_completed
        - entry_price, exit_price, quantity
        - take_profit, stop_loss
        - pnl, total_pnl
        """
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        running_pnl = 0.0
        
        for trade in self.trades:
            if trade.pnl:
                running_pnl += trade.pnl
            
            records.append({
                'trade_id': trade.trade_id,
                'date_executed': trade.date_executed,
                'ticker': trade.ticker,
                'type': trade.trade_type,
                'date_completed': trade.date_completed,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'take_profit': trade.take_profit,
                'stop_loss': trade.stop_loss,
                'support_level': trade.support_level,
                'resistance_level': trade.resistance_level,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'total_pnl': running_pnl if trade.pnl else None,
                'duration_hours': trade.duration
            })
        
        return pd.DataFrame(records)
    
    def save_to_json(self, filepath: str):
        """Save all trades to JSON file."""
        data = {
            'ticker': self.ticker,
            'total_pnl': self._total_pnl,
            'statistics': self.get_statistics(),
            'trades': [t.to_dict() for t in self.trades]
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_to_csv(self, filepath: str):
        """Save trades to CSV file."""
        df = self.to_dataframe()
        if not df.empty:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            df.to_csv(filepath, index=False)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'TradeTracker':
        """Load trades from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls(ticker=data['ticker'])
        tracker._total_pnl = data.get('total_pnl', 0.0)
        
        for trade_data in data.get('trades', []):
            trade = Trade.from_dict(trade_data)
            tracker.trades.append(trade)
            tracker._next_id = max(tracker._next_id, trade.trade_id + 1)
        
        return tracker
    
    def print_summary(self):
        """Print a formatted summary of trading results."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print(f"  TRADE SUMMARY - {self.ticker}")
        print("=" * 50)
        print(f"  Total Trades:     {stats['total_trades']}")
        print(f"  Winning Trades:   {stats['winning_trades']}")
        print(f"  Losing Trades:    {stats['losing_trades']}")
        print(f"  Win Rate:         {stats['win_rate']:.1f}%")
        print("-" * 50)
        print(f"  Total PnL:        ${stats['total_pnl']:,.2f}")
        print(f"  Avg Win:          ${stats['avg_win']:,.2f}")
        print(f"  Avg Loss:         ${stats['avg_loss']:,.2f}")
        print(f"  Largest Win:      ${stats['largest_win']:,.2f}")
        print(f"  Largest Loss:     ${stats['largest_loss']:,.2f}")
        print("-" * 50)
        print(f"  Avg Duration:     {stats['avg_trade_duration_hours']:.1f} hours")
        print("=" * 50 + "\n")
