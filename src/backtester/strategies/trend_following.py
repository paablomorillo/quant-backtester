"""Time-series trend following with a moving-average rule (Faber, 2007)."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Strategy


class TrendFollowing(Strategy):
    """Holds each asset only while its price is above its own `window`-day moving average.

    The signal is computed asset by asset (time-series), not by ranking assets against each
    other (cross-sectional): an asset in an uptrend gets a fixed 1/n of the portfolio, where n
    is the number of assets with a price that day, and an asset out of trend gets 0. The
    weights of the assets out of trend are left in cash, not shared among the rest, so the
    exposure goes from 0 (nothing trending) to 1 (everything trending).

    The idea is that trends persist, so stepping aside below the average avoids part of the
    long drawdowns. Faber (2007) uses this rule with a 10-month average (about 200 sessions);
    Moskowitz, Ooi and Pedersen (2012) document the underlying time-series momentum effect.
    """

    name = "trend_following"

    def __init__(self, window: int = 200):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """True where the price is strictly above its moving average (False during warm-up)."""
        return prices > prices.rolling(self.window).mean()

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        n_listed = prices.notna().sum(axis=1)
        return self.signal(prices).div(n_listed, axis=0).fillna(0.0)
