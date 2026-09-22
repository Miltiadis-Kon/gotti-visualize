"""
MySQL Institutional Candle Feed for Strategy Backtesting & Visualization
========================================================================
Queries historical 5-minute, 1-hour, and 1-day candles directly from the local
MySQL `gotti.candles` table (holding 12.5M+ institutional LSEG candles).
Provides clean, structured Pandas DataFrames ready for Lumibot, Plotly, or Pandas-TA.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()
# Also search common env locations
for p in ["E:/repos/gotti-backend/.env", "E:/repos/stock-alchemist/.env", ".env"]:
    if os.path.exists(p):
        load_dotenv(p)

logger = logging.getLogger("MySQLCandleFeed")


def get_mysql_connection():
    """Returns a connection to the local MySQL database."""
    host = os.getenv("DB_HOST", os.getenv("MYSQL_HOST", "127.0.0.1"))
    port = int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", 3306)))
    user = os.getenv("DB_USER", os.getenv("DB_USERNAME", "root"))
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", os.getenv("DB_DATABASE", "gotti"))

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True
    )


class MySQLCandleFeed:
    """
    Data provider querying local MySQL candlestick warehouse.
    """

    TIMEFRAME_MAP = {
        "5m": "5 min",
        "5min": "5 min",
        "5 min": "5 min",
        "1h": "1 hour",
        "1hour": "1 hour",
        "1 hour": "1 hour",
        "1d": "1 day",
        "1day": "1 day",
        "1 day": "1 day",
    }

    def __init__(self):
        pass

    def get_candles_df(
        self,
        ticker: str,
        timeframe: str = "5 min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieves historical candles for a ticker as an OHLCV DataFrame.

        Args:
            ticker: Stock symbol (e.g. 'AAPL', 'NVDA', 'MSFT').
            timeframe: '5 min', '1 hour', or '1 day' (default: '5 min').
            start_date: ISO date or datetime string (e.g. '2025-01-01').
            end_date: ISO date or datetime string.
            limit: Maximum candles to return (None for all).

        Returns:
            pd.DataFrame with columns: ['open', 'high', 'low', 'close', 'volume', 'vwap']
            indexed by datetime (UTC).
        """
        clean_ticker = ticker.replace(".O", "").replace(".N", "").upper()
        normalized_tf = self.TIMEFRAME_MAP.get(timeframe.lower(), timeframe)

        query = """
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM candles
            WHERE ticker = %s AND timeframe = %s
        """
        params: List[Any] = [clean_ticker, normalized_tf]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {int(limit)}"

        conn = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)

            for col in ["open", "high", "low", "close", "volume", "vwap"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch candles for {ticker} from MySQL: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
        finally:
            if conn and conn.is_connected():
                conn.close()

    def get_ticker_coverage(self, ticker: str, timeframe: str = "5 min") -> Dict[str, Any]:
        """Returns date range and total candle count for a ticker."""
        clean_ticker = ticker.replace(".O", "").replace(".N", "").upper()
        normalized_tf = self.TIMEFRAME_MAP.get(timeframe.lower(), timeframe)

        conn = None
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT COUNT(*) as count, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                FROM candles
                WHERE ticker = %s AND timeframe = %s
            """, (clean_ticker, normalized_tf))
            row = cursor.fetchone()
            cursor.close()

            if row:
                return {
                    "ticker": clean_ticker,
                    "timeframe": normalized_tf,
                    "count": row["count"],
                    "start_date": row["min_ts"].isoformat() if row["min_ts"] else None,
                    "end_date": row["max_ts"].isoformat() if row["max_ts"] else None
                }
            return {"ticker": clean_ticker, "count": 0}
        finally:
            if conn and conn.is_connected():
                conn.close()

    def create_lumibot_pandas_data(
        self,
        ticker: str,
        timeframe: str = "5 min",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[Any, Any]:
        """
        Creates a Lumibot-compatible pandas_data dictionary for PandasDataBacktesting.
        """
        from lumibot.entities import Asset, Data

        df = self.get_candles_df(ticker, timeframe=timeframe, start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError(f"No candles found for {ticker} with timeframe {timeframe}")

        asset = Asset(symbol=ticker.upper(), asset_type="stock")
        quote_asset = Asset(symbol="USD", asset_type="forex")

        # Map timeframe to Lumibot timestep
        ts_map = {"5 min": "5minute", "1 hour": "hour", "1 day": "day"}
        timestep = ts_map.get(timeframe, "5minute")

        return {
            asset: Data(asset, df, timestep=timestep, quote=quote_asset)
        }


# Singleton instance
candle_feed = MySQLCandleFeed()
