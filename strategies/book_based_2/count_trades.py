import os
import glob
import pandas as pd

logs_dir = r"E:\repos\gotti-visualize\logs"
trades_files = glob.glob(os.path.join(logs_dir, "*", "trades.csv"))

results = {}

for f in trades_files:
    try:
        df = pd.read_csv(f)
        if df.empty or 'asset' not in df.columns:
            continue
            
        folder_name = os.path.basename(os.path.dirname(f))
        # Strategy name is everything before the date. Date is typically 'YYYY-MM-DD'
        strat_name = folder_name.rsplit('_', 2)[0]
        
        # Filter only entry orders (ignore closing brackets that might skew the "how many setups" metric)
        # Actually, let's just look at unique entry points, or just divide by 2 for round trips?
        # A setup creates 1 entry. A bracket close creates 1 exit.
        # Let's count unique 'datetime' where a trade happened to estimate setup count,
        # or just count total filled orders. Let's count total round-trip setups.
        # In Lumibot, each trade is a row.
        
        for asset in df['asset'].unique():
            if asset not in ['NVDA', 'PLTR', 'MARA']:
                continue
                
            asset_trades = df[df['asset'] == asset]
            # How many buys/sells?
            entries = asset_trades[asset_trades['status'] == 'filled']
            # Divide by 2 because each trade has an entry and an exit
            trade_count = len(entries) // 2
            
            if asset not in results:
                results[asset] = {}
            
            # Since we ran multiple backtests, we want the max count or just the most recent one
            # The most recent folder has the highest timestamp.
            # Let's just keep the latest file's count.
            file_mtime = os.path.getmtime(f)
            
            if strat_name not in results[asset]:
                results[asset][strat_name] = {'count': trade_count, 'mtime': file_mtime}
            else:
                if file_mtime > results[asset][strat_name]['mtime']:
                    results[asset][strat_name] = {'count': trade_count, 'mtime': file_mtime}
                    
    except Exception as e:
        pass

for asset, strats in results.items():
    print(f"\n--- {asset} ---")
    for strat, data in strats.items():
        print(f"{strat:20s}: {data['count']} completed trades")

