import sys
import json
from datetime import datetime
sys.path.append(r"E:\repos\gotti-visualize")

from lumibot.backtesting import YahooDataBacktesting
from lumibot.entities import Asset

from strategies.book_based_2.hcr_breakout_long import HCRBreakoutLong
from strategies.book_based_2.hcr_breakout_short import HCRBreakoutShort
from strategies.book_based_2.fib_retrace_swing import FibRetraceSwing
from strategies.book_based_2.gap_setup_long import OpeningGapLong
from strategies.book_based_2.gap_setup_short import OpeningGapShort
from strategies.book_based_2.retrace_long import RetraceLong
from strategies.book_based_2.retrace_short import RetraceShort
from strategies.book_based_2.rsi_setup import RSISetup

start = datetime(2025, 9, 12)
end = datetime(2026, 9, 12)
params = {"Ticker": Asset(symbol="NVDA", asset_type=Asset.AssetType.STOCK), "Plot": False}

strategies = [
    ("HCR_Long", HCRBreakoutLong),
    ("HCR_Short", HCRBreakoutShort),
    ("Fib_Retrace", FibRetraceSwing),
    ("Gap_Long", OpeningGapLong),
    ("Gap_Short", OpeningGapShort),
    ("Retrace_Long", RetraceLong),
    ("Retrace_Short", RetraceShort),
    ("RSI_Setup", RSISetup)
]

results = {}
for name, cls in strategies:
    try:
        # backtest() returns stats, traces, df, something else depending on version
        res = cls.backtest(
            YahooDataBacktesting,
            start, end,
            budget=10_000,
            parameters=params,
            show_plot=False,
            show_tearsheet=False
        )
        
        # lumibot usually returns stats dictionary as the first element if it's a tuple, or just the dict
        stats = res[0] if isinstance(res, tuple) else res
        
        # extract just the key info to avoid massive json
        results[name] = {
            "cagr": stats.get("cagr", "N/A"),
            "total_return": stats.get("total_return", "N/A"),
            "max_drawdown": stats.get("max_drawdown", "N/A"),
            "win_rate": stats.get("win_rate", "N/A"),
            "sharpe_ratio": stats.get("sharpe_ratio", "N/A"),
            "total_trades": stats.get("total_trades", "N/A")
        }
    except Exception as e:
        results[name] = {"error": str(e)}

print("===JSON_START===")
print(json.dumps(results, indent=2))
print("===JSON_END===")
