"""Cross-sectional momentum (Jegadeesh & Titman, 1993)."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Strategy


class CrossSectionalMomentum(Strategy):
    """Buys the `top_n` assets with the highest return over the window [t-lookback, t-skip].

    The `skip` (~1 month by default) avoids the short-term reversal effect.
    long_short=True also shorts the `top_n` worst (a market-neutral portfolio).
    """

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 3, long_short=False):
        if skip >= lookback:
            raise ValueError("skip must be smaller than lookback")
        self.lookback, self.skip, self.top_n, self.long_short = lookback, skip, top_n, long_short

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.shift(self.skip) / prices.shift(self.lookback) - 1

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.signal(prices)
        valid = (
            sig.notna().sum(axis=1) >= 2 * self.top_n
            if self.long_short
            else sig.notna().any(axis=1)
        )
        rank_desc = sig.rank(axis=1, ascending=False)
        longs = (rank_desc <= self.top_n).astype(float)
        w = longs.div(longs.sum(axis=1), axis=0)
        if self.long_short:
            rank_asc = sig.rank(axis=1, ascending=True)
            shorts = (rank_asc <= self.top_n).astype(float)
            w = 0.5 * w - 0.5 * shorts.div(shorts.sum(axis=1), axis=0)
        return w.where(valid, 0.0).fillna(0.0)
