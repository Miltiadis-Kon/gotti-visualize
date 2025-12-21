"""
Test file for Live Key Levels Streaming

Tests the live streaming functionality with Alpaca's WebSocket API.
Monitors NVDA in real-time and alerts when price approaches key levels.

NOTE: This requires:
1. Alpaca API keys set as environment variables
2. Market hours (9:30 AM - 4:00 PM ET) for live data
"""

from key_levels import LiveKeyLevels, run_live_monitor
from datetime import datetime
import os


def test_live_streaming():
    """
    Start live monitoring for NVDA.
    Will load key levels first, then stream real-time data.
    """
    print("=" * 70)
    print("LIVE KEY LEVELS MONITOR - NVDA")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will:")
    print("  1. Load historical key levels (1D, 4H, 1H, 15m)")
    print("  2. Connect to Alpaca live stream")
    print("  3. Monitor price and alert when near key levels")
    print("\nPress Ctrl+C to stop\n")
    
    # Make sure API keys are set
    if not os.getenv("APCA_API_KEY_PAPER"):
        print("ERROR: Set APCA_API_KEY_PAPER environment variable first!")
        print("Example: $env:APCA_API_KEY_PAPER = 'your_key'")
        return
    
    # Start the live monitor
    run_live_monitor(
        ticker="NVDA",
        alert_distance_pct=0.5,  # Alert when within 0.5% of level
        use_bars=True  # Use 1-min bars (set False for quotes)
    )


def test_live_with_quotes():
    """
    Alternative test using quote data instead of bars.
    Quotes are more real-time but noisier.
    """
    run_live_monitor(
        ticker="NVDA",
        alert_distance_pct=0.3,
        use_bars=False  # Use quotes instead of bars
    )


if __name__ == "__main__":
    # Run live streaming test
    test_live_streaming()
    
    # Alternative: use quotes
    # test_live_with_quotes()
