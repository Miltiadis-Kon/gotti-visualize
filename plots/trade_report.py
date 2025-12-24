"""
Trade Report Generator

Functions for generating trade reports and statistics from backtest results.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import sys

# Import Trade class
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategies.trade_tracker import Trade, TradeTracker
except ImportError:
    Trade = None
    TradeTracker = None


def generate_trade_report(trades: List[Any]) -> pd.DataFrame:
    """
    Generate a trade report DataFrame with required columns.
    
    Parameters:
    -----------
    trades : List[Trade]
        List of Trade objects
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - date_executed
        - ticker
        - type (buy/sell)
        - date_completed
        - take_profit
        - stop_loss
        - pnl
        - total_pnl
    """
    if not trades:
        return pd.DataFrame(columns=[
            'date_executed', 'ticker', 'type', 'date_completed',
            'entry_price', 'exit_price', 'take_profit', 'stop_loss',
            'pnl', 'total_pnl'
        ])
    
    records = []
    running_pnl = 0.0
    
    for trade in trades:
        if trade.pnl:
            running_pnl += trade.pnl
        
        records.append({
            'date_executed': trade.date_executed,
            'ticker': trade.ticker,
            'type': trade.trade_type,
            'date_completed': trade.date_completed,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'take_profit': trade.take_profit,
            'stop_loss': trade.stop_loss,
            'exit_reason': trade.exit_reason,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'total_pnl': running_pnl if trade.pnl else None,
        })
    
    return pd.DataFrame(records)


def calculate_statistics(trades: List[Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive trading statistics.
    
    Parameters:
    -----------
    trades : List[Trade]
        List of Trade objects
        
    Returns:
    --------
    dict
        Dictionary with:
        - total_trades, winning_trades, losing_trades
        - win_rate, total_pnl
        - avg_win, avg_loss, largest_win, largest_loss
        - avg_trade_duration_hours
        - profit_factor, avg_pnl_per_trade
    """
    closed = [t for t in trades if t.date_completed is not None]
    
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
            'avg_trade_duration_hours': 0.0,
            'profit_factor': 0.0,
            'avg_pnl_per_trade': 0.0,
        }
    
    winners = [t for t in closed if t.pnl and t.pnl > 0]
    losers = [t for t in closed if t.pnl and t.pnl <= 0]
    
    win_pnls = [t.pnl for t in winners]
    loss_pnls = [abs(t.pnl) for t in losers]  # Absolute values for loss
    all_pnls = [t.pnl for t in closed if t.pnl]
    
    total_pnl = sum(all_pnls) if all_pnls else 0.0
    total_wins = sum(win_pnls) if win_pnls else 0.0
    total_losses = sum(loss_pnls) if loss_pnls else 0.0
    
    durations = [t.duration for t in closed if t.duration is not None]
    
    return {
        'total_trades': len(closed),
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'win_rate': (len(winners) / len(closed) * 100) if closed else 0.0,
        'total_pnl': total_pnl,
        'avg_win': (total_wins / len(winners)) if winners else 0.0,
        'avg_loss': -(total_losses / len(losers)) if losers else 0.0,
        'largest_win': max(win_pnls) if win_pnls else 0.0,
        'largest_loss': min([t.pnl for t in losers]) if losers else 0.0,
        'avg_trade_duration_hours': (sum(durations) / len(durations)) if durations else 0.0,
        'profit_factor': (total_wins / total_losses) if total_losses > 0 else float('inf'),
        'avg_pnl_per_trade': (total_pnl / len(closed)) if closed else 0.0,
    }


def export_trades_csv(trades: List[Any], filepath: str):
    """
    Export trades to CSV file.
    
    Parameters:
    -----------
    trades : List[Trade]
        List of Trade objects
    filepath : str
        Output file path
    """
    df = generate_trade_report(trades)
    
    # Ensure directory exists
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"Trades exported to: {filepath}")


def print_trade_summary(trades: List[Any], ticker: str = ""):
    """
    Print a formatted summary of trading results.
    
    Parameters:
    -----------
    trades : List[Trade]
        List of Trade objects
    ticker : str
        Stock symbol for display
    """
    stats = calculate_statistics(trades)
    
    if trades:
        ticker = trades[0].ticker if not ticker else ticker
    
    print("\n" + "=" * 60)
    print(f"  TRADE SUMMARY - {ticker}")
    print("=" * 60)
    print(f"  Total Trades:     {stats['total_trades']}")
    print(f"  Winning Trades:   {stats['winning_trades']}")
    print(f"  Losing Trades:    {stats['losing_trades']}")
    print(f"  Win Rate:         {stats['win_rate']:.1f}%")
    print("-" * 60)
    print(f"  Total PnL:        ${stats['total_pnl']:,.2f}")
    print(f"  Avg PnL/Trade:    ${stats['avg_pnl_per_trade']:,.2f}")
    print(f"  Profit Factor:    {stats['profit_factor']:.2f}")
    print("-" * 60)
    print(f"  Avg Win:          ${stats['avg_win']:,.2f}")
    print(f"  Avg Loss:         ${stats['avg_loss']:,.2f}")
    print(f"  Largest Win:      ${stats['largest_win']:,.2f}")
    print(f"  Largest Loss:     ${stats['largest_loss']:,.2f}")
    print("-" * 60)
    print(f"  Avg Duration:     {stats['avg_trade_duration_hours']:.1f} hours")
    print("=" * 60 + "\n")


def print_trade_table(trades: List[Any]):
    """
    Print a formatted table of all trades.
    
    Parameters:
    -----------
    trades : List[Trade]
        List of Trade objects
    """
    if not trades:
        print("No trades to display.")
        return
    
    df = generate_trade_report(trades)
    
    # Format for display
    display_df = df.copy()
    display_df['date_executed'] = pd.to_datetime(display_df['date_executed']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['date_completed'] = pd.to_datetime(display_df['date_completed']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
    display_df['take_profit'] = display_df['take_profit'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
    display_df['stop_loss'] = display_df['stop_loss'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "")
    display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+,.2f}" if pd.notnull(x) else "")
    display_df['total_pnl'] = display_df['total_pnl'].apply(lambda x: f"${x:+,.2f}" if pd.notnull(x) else "")
    
    # Select columns for display
    cols = ['date_executed', 'ticker', 'type', 'entry_price', 'exit_price', 
            'take_profit', 'stop_loss', 'exit_reason', 'pnl', 'total_pnl']
    
    print("\n" + "=" * 120)
    print("  TRADE LOG")
    print("=" * 120)
    print(display_df[cols].to_string(index=False))
    print("=" * 120 + "\n")


def format_trade_for_display(trade: Any) -> Dict[str, str]:
    """
    Format a single trade for display in UI.
    
    Parameters:
    -----------
    trade : Trade
        Trade object
        
    Returns:
    --------
    dict
        Formatted string values for display
    """
    return {
        'Trade ID': str(trade.trade_id),
        'Date Executed': trade.date_executed.strftime('%Y-%m-%d %H:%M') if trade.date_executed else '',
        'Ticker': trade.ticker,
        'Type': trade.trade_type,
        'Entry': f"${trade.entry_price:.2f}",
        'Exit': f"${trade.exit_price:.2f}" if trade.exit_price else 'Open',
        'Qty': str(trade.quantity),
        'TP': f"${trade.take_profit:.2f}",
        'SL': f"${trade.stop_loss:.2f}",
        'Date Completed': trade.date_completed.strftime('%Y-%m-%d %H:%M') if trade.date_completed else 'Open',
        'Exit Reason': trade.exit_reason or '',
        'PnL': f"${trade.pnl:+,.2f}" if trade.pnl else '',
        'PnL %': f"{trade.pnl_percent:+.2f}%" if trade.pnl_percent else '',
        'Duration': f"{trade.duration:.1f}h" if trade.duration else '',
    }
