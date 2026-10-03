"""
Strategy Parameter Database
============================
SQLite schema + seed data for all swing strategies.

Run once to create/reset the parameters database:
    python strategies/create_params_db.py

The DB file is written to:
    logs/strategy_params.db

Columns
-------
strategy_name  TEXT  — must match get_strategy_name() return value exactly
param_key      TEXT  — parameter key (matches parameters dict key)
param_value    TEXT  — value stored as string (cast on read via param_type)
param_type     TEXT  — "int" | "float" | "bool" | "str"
updated_at     TEXT  — ISO 8601 timestamp of last manual edit
description    TEXT  — human-readable note shown in any future UI
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "logs", "strategy_params.db"
)

# ── Seed data — one row per (strategy, parameter) ────────────────────────────
# Format: (strategy_name, param_key, param_value, param_type, description)

SEED_ROWS = [
    # ── HCRBreakoutLongSwing ──────────────────────────────────────────────────
    ("HCRBreakoutLongSwing", "RiskPct",           "0.02",  "float", "Risk fraction per trade leg"),
    ("HCRBreakoutLongSwing", "MAX_PYRAMIDS",       "3",     "int",   "Max concurrent open positions"),
    ("HCRBreakoutLongSwing", "TOTAL_RISK_PCT",     "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("HCRBreakoutLongSwing", "ATR_Multiplier",     "1.5",   "float", "ATR multiple for initial stop"),
    ("HCRBreakoutLongSwing", "ATR_Length",         "14",    "int",   "ATR period"),
    ("HCRBreakoutLongSwing", "Lookback",           "40",    "int",   "Daily bars to fetch"),
    ("HCRBreakoutLongSwing", "ConsolidationDays",  "15",    "int",   "HCR window in trading days"),
    ("HCRBreakoutLongSwing", "MaxRangePct",        "0.05",  "float", "Max HCR price range (5 %)"),
    ("HCRBreakoutLongSwing", "BreakoutBuffer",     "0.002", "float", "% above HCR high before entry"),
    ("HCRBreakoutLongSwing", "TP_ATR_Multiple",    "2.0",   "float", "Take-profit = entry + N x ATR"),

    # ── HCRBreakoutShortSwing ─────────────────────────────────────────────────
    ("HCRBreakoutShortSwing", "RiskPct",           "0.02",  "float", "Risk fraction per trade leg"),
    ("HCRBreakoutShortSwing", "MAX_PYRAMIDS",      "3",     "int",   "Max concurrent open positions"),
    ("HCRBreakoutShortSwing", "TOTAL_RISK_PCT",    "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("HCRBreakoutShortSwing", "ATR_Multiplier",    "1.5",   "float", "ATR multiple for stop"),
    ("HCRBreakoutShortSwing", "ATR_Length",        "14",    "int",   "ATR period"),
    ("HCRBreakoutShortSwing", "Lookback",          "40",    "int",   "Daily bars to fetch"),
    ("HCRBreakoutShortSwing", "ConsolidationDays", "15",    "int",   "HCR window in trading days"),
    ("HCRBreakoutShortSwing", "MaxRangePct",       "0.05",  "float", "Max HCR price range"),
    ("HCRBreakoutShortSwing", "BreakdownBuffer",   "0.002", "float", "% below HCR low before entry"),
    ("HCRBreakoutShortSwing", "TP_ATR_Multiple",   "2.0",   "float", "Take-profit = entry - N x ATR"),

    # ── GapSetupLongSwing ─────────────────────────────────────────────────────
    ("GapSetupLongSwing", "RiskPct",          "0.02",  "float", "Risk fraction per trade leg"),
    ("GapSetupLongSwing", "MAX_PYRAMIDS",     "3",     "int",   "Max concurrent open positions"),
    ("GapSetupLongSwing", "TOTAL_RISK_PCT",   "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("GapSetupLongSwing", "ATR_Multiplier",   "1.5",   "float", "ATR multiple for stop"),
    ("GapSetupLongSwing", "ATR_Length",       "14",    "int",   "ATR period"),
    ("GapSetupLongSwing", "Lookback",         "40",    "int",   "Daily bars to fetch"),
    ("GapSetupLongSwing", "GapPct",           "0.02",  "float", "Minimum gap-down magnitude (2 %)"),
    ("GapSetupLongSwing", "SL_ATR_Fraction",  "0.5",   "float", "SL = today_open - fraction x ATR"),

    # ── GapSetupShortSwing ────────────────────────────────────────────────────
    ("GapSetupShortSwing", "RiskPct",         "0.02",  "float", "Risk fraction per trade leg"),
    ("GapSetupShortSwing", "MAX_PYRAMIDS",    "3",     "int",   "Max concurrent open positions"),
    ("GapSetupShortSwing", "TOTAL_RISK_PCT",  "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("GapSetupShortSwing", "ATR_Multiplier",  "1.5",   "float", "ATR multiple for stop"),
    ("GapSetupShortSwing", "ATR_Length",      "14",    "int",   "ATR period"),
    ("GapSetupShortSwing", "Lookback",        "40",    "int",   "Daily bars to fetch"),
    ("GapSetupShortSwing", "GapPct",          "0.02",  "float", "Minimum gap-up magnitude (2 %)"),
    ("GapSetupShortSwing", "SL_ATR_Fraction", "0.5",   "float", "SL = today_open + fraction x ATR"),

    # ── FibRetraceSwing ───────────────────────────────────────────────────────
    ("FibRetraceSwing", "RiskPct",          "0.02",  "float", "Risk fraction per trade leg"),
    ("FibRetraceSwing", "MAX_PYRAMIDS",     "3",     "int",   "Max concurrent open positions"),
    ("FibRetraceSwing", "TOTAL_RISK_PCT",   "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("FibRetraceSwing", "ATR_Multiplier",   "2.0",   "float", "ATR multiple for stop"),
    ("FibRetraceSwing", "ATR_Length",       "14",    "int",   "ATR period"),
    ("FibRetraceSwing", "Lookback",         "40",    "int",   "Daily bars to fetch"),
    ("FibRetraceSwing", "TrendDays",        "10",    "int",   "Swing window for measuring the move"),
    ("FibRetraceSwing", "MinMovePct",       "0.03",  "float", "Min swing move size (3 %) for validity"),
    ("FibRetraceSwing", "FibLow",           "0.382", "float", "Lower Fibonacci zone boundary"),
    ("FibRetraceSwing", "FibHigh",          "0.618", "float", "Upper Fibonacci zone boundary"),

    # ── RetraceLongSwing ──────────────────────────────────────────────────────
    ("RetraceLongSwing", "RiskPct",          "0.02",  "float", "Risk fraction per trade leg"),
    ("RetraceLongSwing", "MAX_PYRAMIDS",     "3",     "int",   "Max concurrent open positions"),
    ("RetraceLongSwing", "TOTAL_RISK_PCT",   "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("RetraceLongSwing", "ATR_Multiplier",   "1.5",   "float", "ATR multiple for stop"),
    ("RetraceLongSwing", "ATR_Length",       "14",    "int",   "ATR period"),
    ("RetraceLongSwing", "Lookback",         "40",    "int",   "Daily bars to fetch"),
    ("RetraceLongSwing", "TP_ATR_Multiple",  "1.5",   "float", "Take-profit = entry + N x ATR"),

    # ── RetraceShortSwing ─────────────────────────────────────────────────────
    ("RetraceShortSwing", "RiskPct",         "0.02",  "float", "Risk fraction per trade leg"),
    ("RetraceShortSwing", "MAX_PYRAMIDS",    "3",     "int",   "Max concurrent open positions"),
    ("RetraceShortSwing", "TOTAL_RISK_PCT",  "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("RetraceShortSwing", "ATR_Multiplier",  "1.5",   "float", "ATR multiple for stop"),
    ("RetraceShortSwing", "ATR_Length",      "14",    "int",   "ATR period"),
    ("RetraceShortSwing", "Lookback",        "40",    "int",   "Daily bars to fetch"),
    ("RetraceShortSwing", "TP_ATR_Multiple", "1.5",   "float", "Take-profit = entry - N x ATR"),

    # ── RSISetupSwing ─────────────────────────────────────────────────────────
    ("RSISetupSwing", "RiskPct",          "0.02",  "float", "Risk fraction per trade leg"),
    ("RSISetupSwing", "MAX_PYRAMIDS",     "3",     "int",   "Max concurrent open positions"),
    ("RSISetupSwing", "TOTAL_RISK_PCT",   "0.06",  "float", "Hard cap on total open portfolio risk"),
    ("RSISetupSwing", "ATR_Multiplier",   "2.0",   "float", "ATR multiple for stop"),
    ("RSISetupSwing", "ATR_Length",       "14",    "int",   "ATR period"),
    ("RSISetupSwing", "Lookback",         "60",    "int",   "Daily bars to fetch (more needed for RSI warmup)"),
    ("RSISetupSwing", "RsiLength",        "14",    "int",   "RSI period"),
    ("RSISetupSwing", "RsiOversold",      "30",    "float", "RSI oversold threshold"),
    ("RSISetupSwing", "TP_ATR_Multiple",  "2.0",   "float", "Take-profit = entry + N x ATR"),

    # ── MultiTimeframeKeyLevelsStrategy ───────────────────────────────────────
    ("MultiTFKeyLevels", "RISK_PERCENT",        "0.10",  "float", "Risk fraction per trade leg"),
    ("MultiTFKeyLevels", "MIN_IMPORTANCE",      "3",     "int",   "Min S/R level importance score"),
    ("MultiTFKeyLevels", "ENTRY_THRESHOLD",     "0.005", "float", "Price proximity to level for entry (0.5 %)"),
    ("MultiTFKeyLevels", "EXIT_THRESHOLD",      "0.01",  "float", "TP/SL proximity tolerance (1 %)"),
    ("MultiTFKeyLevels", "TP_THRESHOLD",        "0.02",  "float", "TP set N % before resistance"),
    ("MultiTFKeyLevels", "SL_THRESHOLD",        "0.05",  "float", "SL set N % below support"),
    ("MultiTFKeyLevels", "MIN_RISK_REWARD",     "1.5",   "float", "Minimum acceptable R:R ratio"),
    ("MultiTFKeyLevels", "RECALC_ITERATIONS",   "3",     "int",   "Recalculate levels every N iterations (×5min)"),
    ("MultiTFKeyLevels", "FIB_SR_PROXIMITY",    "0.02",  "float", "Fib/S-R overlap suppression radius (2 %)"),
]


def create_db(db_path: str = DB_PATH, overwrite: bool = False):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if overwrite and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing DB: {db_path}")

    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_parameters (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT    NOT NULL,
            param_key     TEXT    NOT NULL,
            param_value   TEXT    NOT NULL,
            param_type    TEXT    NOT NULL DEFAULT 'float',
            description   TEXT    DEFAULT '',
            updated_at    TEXT    NOT NULL,
            UNIQUE (strategy_name, param_key)
        )
    """)

    now = datetime.utcnow().isoformat()
    inserted = 0
    for strategy_name, key, value, ptype, desc in SEED_ROWS:
        conn.execute(
            """
            INSERT OR IGNORE INTO strategy_parameters
                (strategy_name, param_key, param_value, param_type, description, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strategy_name, key, value, ptype, desc, now)
        )
        inserted += conn.execute(
            "SELECT changes()"
        ).fetchone()[0]

    conn.commit()
    conn.close()
    print(f"DB ready: {db_path}")
    print(f"Rows inserted: {inserted} / {len(SEED_ROWS)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create / reset the strategy parameters DB.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing DB and rebuild from scratch.")
    parser.add_argument("--db", default=DB_PATH, help="Path to the SQLite file.")
    args = parser.parse_args()
    create_db(db_path=args.db, overwrite=args.overwrite)
