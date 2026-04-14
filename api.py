from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

try:
    import pymysql
    import ssl as _ssl
    HAS_DB = True
except ImportError:
    HAS_DB = False

app = FastAPI(title="Gotti Chart API")

# Allow the HTML page to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the plots directory (HTML chart) as static files at /plots
from fastapi.staticfiles import StaticFiles
_plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
app.mount("/plots", StaticFiles(directory=_plots_dir), name="plots")

@app.get("/chart")
def chart_redirect():
    """Convenience redirect to the chart HTML page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/plots/stock_chart.html")

@app.get("/")
def read_root():
    return {"status": "ok", "service": "gotti-api"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Example endpoint to list strategies
@app.get("/strategies")
def list_strategies():
    strategies_dir = "strategies"
    strategies = []
    if os.path.exists(strategies_dir):
        for f in os.listdir(strategies_dir):
            if f.endswith(".py") and f != "__init__.py":
                strategies.append(f)
    return {"strategies": strategies}

def get_db_connection():
    if not HAS_DB:
        return None
    try:
        # Provide path to the CA cert correctly
        ca_path = os.path.abspath(os.getenv("TIDB_CA_PATH", "isrgrootx1.pem"))
        ssl_ctx = _ssl.create_default_context(cafile=ca_path)
        
        connection = pymysql.connect(
            host=os.getenv("TIDB_HOST"),
            port=int(os.getenv("TIDB_PORT", 4000)),
            user=os.getenv("TIDB_USER"),
            password=os.getenv("TIDB_PASSWORD"),
            database=os.getenv("TIDB_DATABASE"),
            ssl=ssl_ctx,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

@app.get("/tidb-data")
def get_tidb_data():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cursor:
            # Fetch some data. Since we don't know the schema, we'll return list of tables first
            # and attempt to fetch 10 rows from the first table if there's any.
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            
            data = {"tables": tables}
            
            if tables:
                first_table = list(tables[0].values())[0]
                cursor.execute(f"SELECT * FROM `{first_table}` LIMIT 10;")
                data["sample_data"] = cursor.fetchall()
                data["sample_table"] = first_table
            
            return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Local Cloud Data Endpoints ---

def get_sqlite_connection():
    db_path = "local_cloud_data.db"
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # To return dict-like rows
    return conn

@app.get("/local-data/tables")
def get_local_tables():
    conn = get_sqlite_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Local database not found. Please run the load script first.")
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        return {"status": "success", "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/local-data/table/{table_name}")
def get_local_table_data(table_name: str, limit: int = 100, offset: int = 0):
    conn = get_sqlite_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Local database not found.")
    
    try:
        cursor = conn.cursor()
        
        # Verify table exists to prevent SQL injection
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
            
        # Fetch data
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ? OFFSET ?;", (limit, offset))
        rows = [dict(row) for row in cursor.fetchall()]
        
        # Fetch total count
        cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`;")
        total_count = cursor.fetchone()["count"]
        
        return {
            "status": "success", 
            "table": table_name,
            "total_rows": total_count,
            "data": rows,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ─────────────────── CHART DATA ENDPOINTS ───────────────────

INTERVAL_DAYS = {
    "5m":  7,
    "15m": 30,
    "1h":  60,
    "4h":  90,
    "1d":  365,
}

@app.get("/chart/ohlcv")
async def get_ohlcv(
    ticker: str = Query(..., description="Stock ticker e.g. NVDA"),
    interval: str = Query("5m", description="Candle interval: 5m, 15m, 1h, 4h, 1d"),
    days: Optional[int] = Query(None, description="Lookback days (auto if not set)"),
):
    """
    Return OHLCV candles for a ticker + interval from Yahoo Finance.
    Runs yfinance in a thread executor to avoid blocking the event loop.
    """
    import asyncio, warnings
    loop = asyncio.get_event_loop()

    def _fetch():
        import yfinance as yf
        import pandas as pd

        lookback = days or INTERVAL_DAYS.get(interval, 30)
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=lookback)

        fetch_interval = "1h" if interval == "4h" else interval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(
                ticker.upper(),
                start=start_dt,
                end=end_dt,
                interval=fetch_interval,
                progress=False,
                auto_adjust=True,
            )

        if df.empty:
            return None, f"No data returned for {ticker} at interval={interval}"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"}, inplace=True)

        if interval == "4h":
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna(subset=["open"])

        df = df.dropna(subset=["open", "close"])
        df.index = pd.to_datetime(df.index)

        candles = []
        for ts, row in df.iterrows():
            time_val = ts.strftime("%Y-%m-%d") if interval == "1d" else int(ts.timestamp())
            candles.append({
                "time":   time_val,
                "open":   round(float(row["open"]),  4),
                "high":   round(float(row["high"]),  4),
                "low":    round(float(row["low"]),   4),
                "close":  round(float(row["close"]), 4),
                "volume": int(row["volume"]),
            })
        return candles, None

    try:
        candles, err = await loop.run_in_executor(None, _fetch)
        if err:
            raise HTTPException(status_code=404, detail=err)
        return {
            "ticker":   ticker.upper(),
            "interval": interval,
            "candles":  candles,
            "count":    len(candles),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart/key-levels")
def get_key_levels(
    ticker: str = Query(..., description="Stock ticker e.g. NVDA"),
):
    """
    Run the multi-timeframe analyzer and return S/R + Fibonacci levels.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategies"))
        from key_levels.analyzer import analyze

        result = analyze(
            ticker.upper(),
            resolutions=["1D", "4H", "15m"],
        )

        # Support / Resistance
        levels = []
        if not result.merged_levels.empty:
            for _, row in result.merged_levels.iterrows():
                levels.append({
                    "price":       round(float(row["level_price"]), 4),
                    "type":        str(row["type"]),
                    "touchCount":  int(row["touch_count"]),
                    "importance":  int(row["importance"]),
                })

        # Fibonacci setups
        fibs = []
        if not result.trade_setups.empty:
            for _, row in result.trade_setups.iterrows():
                fib_data = {
                    "patternId":  str(row.get("pattern_id", "")),
                    "trend":      str(row["trend"]),
                    "resolution": str(row.get("resolution", "")),
                    "low":        round(float(row["low_price"]),    4),
                    "high":       round(float(row["high_price"]),   4),
                    "entry":      round(float(row["entry_price"]),  4),
                    "sl":         round(float(row["stop_loss"]),    4),
                    "tp":         round(float(row["take_profit"]),  4),
                    "rangePct":   round(float(row["range_pct"]),    2),
                    "rr":         round(float(row["risk_reward"]),  2),
                    "levels": {
                        "0.000": round(float(row["fib_0"]),   4),
                        "0.236": round(float(row["fib_236"]), 4),
                        "0.382": round(float(row["fib_382"]), 4),
                        "0.500": round(float(row["fib_500"]), 4),
                        "0.618": round(float(row["fib_618"]), 4),
                        "0.786": round(float(row["fib_786"]), 4),
                        "1.000": round(float(row["fib_1000"]), 4),
                    }
                }
                fibs.append(fib_data)

        return {
            "ticker": ticker.upper(),
            "levels": levels,
            "fibSetups": fibs,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
