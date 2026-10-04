import sys
import os
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_mixin import PlottableStrategyMixin
from risk_management import optimize_sl_tp, calculate_atr, calculate_trailing_stop, ATRRiskLevels

load_dotenv()

# Ensure singleton module registration across both 'strategies.strat_baseplate' and 'strat_baseplate'
if __name__ == "strategies.strat_baseplate" and "strat_baseplate" not in sys.modules:
    sys.modules["strat_baseplate"] = sys.modules[__name__]
elif __name__ == "strat_baseplate" and "strategies.strat_baseplate" not in sys.modules:
    sys.modules["strategies.strat_baseplate"] = sys.modules[__name__]

apikey = os.getenv("APCA_API_KEY_PAPER")
apisecret = os.getenv("APCA_API_SECRET_KEY_PAPER")

ALPACA_CONFIG = {
    "API_KEY": apikey,
    "API_SECRET": apisecret,
    "PAPER": True,  # Set to True for paper trading, False for live trading
}


class StrategyBaseplate(Strategy, PlottableStrategyMixin):
    """
    Foundational Strategy Baseplate for Lumibot.
    
    Provides:
      - Template lifecycle hooks (before_market_opens, on_trading_iteration, on_filled_order, etc.)
      - Unified ATR Volatility Risk Framework (optimize_sl_tp, get_atr, trailing stop)
      - PlottableStrategyMixin integration for visual reporting
      - Dynamic multi-trading style adaptation (scalping, day_trading, swing_trading, trend_following)
    """

    parameters = {
        "Ticker": Asset(symbol="AAPL", asset_type=Asset.AssetType.STOCK),
        "Plot": True,
        "TradingStyle": "swing_trading",  # "scalping", "day_trading", "swing_trading", "trend_following"
        "ATR_Length": 14,
        "ATR_SL_Multiplier": None,  # None uses TradingStyle default
        "RiskRewardRatio": None,    # None uses TradingStyle default
        "ATR_TP_Multiplier": None,  # Optional direct ATR multiplier for Take Profit
        "TrailingStop": False,      # Enable trailing stop mechanics
    }

    ##### CORE LIFECYCLE FUNCTIONS #####

    def initialize(self):
        self.sleeptime = "5M"  # Execute strategy every 5 minutes by default
        self.will_plot = self.parameters.get("Plot", True)
        self.risk_percent = self.parameters.get("RiskPct", self.parameters.get("RISK_PERCENT", 0.02))
        self.current_trailing_stop = None

    def before_market_opens(self):
        ticker = self.parameters.get("Ticker")
        if ticker:
            try:
                self.ticker_bars = self.get_historical_prices(ticker, 200, "day").df
            except Exception:
                self.ticker_bars = None

        self.tradeable = self.filter()
        self.technical = self.setup()
        self.techincal = self.technical  # Backwards compatibility alias

    def on_trading_iteration(self):
        is_technical = getattr(self, "technical", False) or getattr(self, "techincal", False)
        if getattr(self, "tradeable", False) and is_technical:
            entry_price = self.get_entry_price()
            position_size = self.get_position_sizing()
            if position_size and position_size != 0:
                ticker = self.parameters.get("Ticker")
                order = self.create_order(
                    asset=ticker,
                    quantity=position_size,
                    side="buy",
                )
                self.submit_order(order)

    def after_market_closes(self):
        # Cancel all open unfulfilled entry orders
        pass

    def on_canceled_order(self, order):
        pass

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """
        Triggered when an order is filled.
        Calculates strategy's custom Take Profit & Stop Loss, then passes them
        through the volatility-adjusted ATR risk optimization function.
        """
        # Determine if this was an entry fill or an exit fill
        # For long positions: entry is 'buy', exit is 'sell'
        # For short positions: entry is 'sell'/'sell_short', exit is 'buy'
        is_entry = False
        is_long = True

        if order.side in ("buy", "buy_to_cover"):
            # If position exists with quantity > 0, this was an entry buy
            if position and position.quantity > 0:
                is_entry = True
                is_long = True
            else:
                # Exit buy closing or covering a short position
                is_entry = False
        elif order.side in ("sell", "sell_short"):
            # If position exists with quantity < 0, this was an entry short
            if position and position.quantity < 0:
                is_entry = True
                is_long = False
            else:
                # Exit sell closing a long position
                is_entry = False

        if not is_entry:
            # Exit order filled — trade is complete
            self.current_trailing_stop = None
            return

        # 1. Custom strategy calculation
        custom_tp = self.get_take_profit(price)
        custom_sl = self.get_stop_loss(price)

        # 2. Pass through ATR optimization function
        risk_side = "buy" if is_long else "sell"
        risk_levels = self.optimize_sl_tp(
            entry_price=price,
            stop_loss=custom_sl,
            take_profit=custom_tp,
            side=risk_side,
        )

        final_tp = risk_levels.take_profit
        final_sl = risk_levels.stop_loss

        # 3. Create exit child order with GTD / GTC
        ticker = self.parameters.get("Ticker")
        exit_side = "sell" if is_long else "buy"
        order_kwargs = {
            "asset": ticker,
            "quantity": order.quantity,
            "side": exit_side,
            "time_in_force": "gtc",
        }
        if final_tp is not None:
            order_kwargs["take_profit_price"] = final_tp
        if final_sl is not None:
            order_kwargs["stop_loss_price"] = final_sl

        order2 = self.create_order(**order_kwargs)
        order.add_child_order(order2)

        if getattr(self, "is_backtesting", False) and getattr(self, "will_plot", False):
            pass

    ##### ATR & RISK MANAGEMENT HELPER METHODS #####

    def get_atr(self, *args, length: int = None, df: pd.DataFrame = None, **kwargs) -> float:
        """Fetch or calculate the current ATR value from historical prices."""
        resolved_length = length or self.parameters.get("ATR_Length", 14)
        target_df = df

        for arg in args:
            if isinstance(arg, int):
                resolved_length = arg
            elif isinstance(arg, pd.DataFrame):
                target_df = arg

        if target_df is None:
            target_df = getattr(self, "ticker_bars", None)

        if target_df is None or len(target_df) < 2:
            ticker = self.parameters.get("Ticker")
            if ticker:
                try:
                    bars = self.get_historical_prices(ticker, resolved_length + 20, "day")
                    if bars is not None and hasattr(bars, "df"):
                        target_df = bars.df
                except Exception:
                    pass

        if target_df is not None and not target_df.empty:
            return calculate_atr(target_df, length=resolved_length)
        return 0.0

    def optimize_sl_tp(
        self,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        side: str = "buy",
        trading_style: str = None,
        sl_multiplier: float = None,
        tp_multiplier: float = None,
        risk_reward_ratio: float = None,
        trailing_stop: bool = None,
        atr: float = None,
    ) -> ATRRiskLevels:
        """
        Passes custom Stop Loss and Take Profit through the ATR volatility framework.
        """
        resolved_style = trading_style or self.parameters.get("TradingStyle", "swing_trading")
        resolved_sl_mult = sl_multiplier if sl_multiplier is not None else self.parameters.get("ATR_SL_Multiplier")
        resolved_tp_mult = tp_multiplier if tp_multiplier is not None else self.parameters.get("ATR_TP_Multiplier")
        resolved_rr = risk_reward_ratio if risk_reward_ratio is not None else self.parameters.get("RiskRewardRatio")
        resolved_trailing = trailing_stop if trailing_stop is not None else self.parameters.get("TrailingStop")

        resolved_atr = atr
        if resolved_atr is None or resolved_atr <= 0:
            resolved_atr = self.get_atr()

        return optimize_sl_tp(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            side=side,
            atr=resolved_atr,
            trading_style=resolved_style,
            sl_multiplier=resolved_sl_mult,
            tp_multiplier=resolved_tp_mult,
            risk_reward_ratio=resolved_rr,
            trailing_stop=resolved_trailing,
        )

    def get_trailing_stop(
        self,
        current_price: float,
        side: str = "buy",
        risk_distance: float = None,
        sl_multiplier: float = None,
    ) -> float:
        """Calculates ratcheted trailing stop price."""
        mult = sl_multiplier or self.parameters.get("ATR_SL_Multiplier", 2.0)
        atr_val = self.get_atr()
        new_stop = calculate_trailing_stop(
            current_price=current_price,
            side=side,
            risk_distance=risk_distance,
            atr=atr_val,
            sl_multiplier=mult,
            current_stop=self.current_trailing_stop,
        )
        self.current_trailing_stop = new_stop
        return new_stop

    ##### TRADING HOOKS (To be overridden by child strategies) #####

    def filter(self) -> bool:
        """Pre-market or macro filters."""
        return True

    def setup(self) -> bool:
        """Technical analysis setup verification."""
        return True

    def get_entry_price(self) -> Optional[float]:
        """Custom strategy entry price calculation."""
        return None

    def get_stop_loss(self, entry: float) -> Optional[float]:
        """Custom strategy stop loss calculation (will be passed through optimize_sl_tp)."""
        return None

    def get_take_profit(self, entry: float) -> Optional[float]:
        """Custom strategy take profit calculation (will be passed through optimize_sl_tp)."""
        return None

    def get_position_sizing(self) -> int:
        """Custom strategy position sizing."""
        return 0

    ##### VISUALIZATION FUNCTIONS #####

    def get_plot_spec(self):
        """Describe this strategy's visualization using the plotting pipeline."""
        import yfinance as yf
        import warnings
        from plots.plot_spec import PlotSpec, CandlestickLayer

        ticker = self.parameters.get("Ticker")
        if hasattr(ticker, "symbol"):
            ticker = ticker.symbol

        layers = []
        try:
            end = datetime.now()
            start = end - timedelta(days=400)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    },
                    inplace=True,
                )
                df.reset_index(inplace=True)
                layers.append(CandlestickLayer(data=df, ticker=ticker, timeframe="1D"))
        except Exception:
            pass

        return PlotSpec(ticker=str(ticker), timeframe="1D", layers=layers)

    def on_strategy_end(self):
        if getattr(self, "will_plot", False) and getattr(self, "is_backtesting", False):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            ticker = self.parameters.get("Ticker", "chart")
            ticker_symbol = ticker.symbol if hasattr(ticker, "symbol") else str(ticker)
            output_path = os.path.join(
                repo_root,
                "logs",
                "charts",
                f"{ticker_symbol}_chart.html",
            )
            self.save_plot_html(output_path)
        return super().on_strategy_end()