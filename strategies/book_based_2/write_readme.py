import os

readme_path = r"E:\repos\gotti-visualize\strategies\book_based_2\README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("""# Bulkowski's Day Trading Strategies (5-Minute Timeframe)

This repository implements 8 of Thomas Bulkowski's day trading patterns from *Swing and Day Trading: Evolution of a Trader* and ThePatternSite.com.

## Design Philosophy

**EXACT IMPLEMENTATION:**
At user request, these algorithms execute **exactly** as described on ThePatternSite.com for day trading on 5-minute charts. There are no daily proxies or ATR modifications.

*   **Timeframe:** 5-minute bars (`sleeptime = "5M"`).
*   **Thresholds:** Breakouts and stop-losses are triggered precisely 1 penny ($0.01) above or below the signal candles, as explicitly defined by Bulkowski.
*   **Resampling:** The data is pulled at the minute level and mathematically resampled into 5-minute chunks locally inside the `on_trading_iteration` loop to guarantee precision across all broker data feeds.

## The 8 Strategies

1.  **HCR Breakout Long (`hcrl.html`)**
    *   **Setup:** Scans for a Horizontal Consolidation Region (HCR) over the last 15 bars (75 minutes) where price ranges no more than 0.3%.
    *   **Trigger:** Buys when price breaks $0.01 above the HCR high.
    *   **Exit:** Trails a stop $0.01 below the prior 5-minute candle's low.

2.  **HCR Breakout Short (`hcrs.html`)**
    *   **Setup:** Scans for an HCR.
    *   **Trigger:** Shorts when price breaks $0.01 below the HCR low.
    *   **Exit:** Trails a stop $0.01 above the prior 5-minute candle's high.

3.  **Fib Retrace Swing (`Daytrade.html`)**
    *   **Setup:** Identifies a straight-line uprun (3+ consecutive green candles).
    *   **Trigger:** Buys when price retraces 38% to 62% of the run and prints a bullish reversal candle.
    *   **Exit:** Stops out $0.01 below the swing low.

4.  **Gap Setup Long (`gapsetupl.html`)**
    *   **Setup:** Opening gap *down*.
    *   **Trigger:** Buys when the price crosses $0.01 above the high of the first 5-minute candle.
    *   **Exit:** Targets the gap fill (yesterday's close) or stops out $0.01 below the day's low.

5.  **Gap Setup Short (`gapsetups.html`)**
    *   **Setup:** Opening gap *up*.
    *   **Trigger:** Shorts when price crosses $0.01 below the low of the first 5-minute candle.
    *   **Exit:** Targets the gap fill.

6.  **Retrace Long (`retracel.html`)**
    *   **Setup:** Identifies a 3-candle straight downtrend on the 5-minute chart.
    *   **Trigger:** Buys when price crosses $0.01 above the prior candle's high.
    *   **Exit:** Stops out $0.01 below the swing low.

7.  **Retrace Short (`retraces.html`)**
    *   **Setup:** Identifies a 3-candle straight uptrend on the 5-minute chart.
    *   **Trigger:** Shorts when price crosses $0.01 below the prior candle's low.
    *   **Exit:** Stops out $0.01 above the swing high.

8.  **RSI Setup (`rsisetup.html`)**
    *   **Setup:** RSI(14) falls below 30.
    *   **Trigger:** Buys when RSI crosses back *above* 30.
    *   **Exit:** Stops out $0.01 below the local swing low.

## Usage Limitations
Because these are now strictly `5M` intraday strategies, standard Yahoo Finance backtesting will only work for the last 60 days of history due to Yahoo's strict intraday data limits. To run deep 1-year backtests, you must use Polygon or ThetaData as the `datasource_class`.
""")
print("README overwritten.")
