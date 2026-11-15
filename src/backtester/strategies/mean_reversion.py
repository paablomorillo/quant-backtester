"""Short-term cross-sectional mean reversion (Lehmann, 1990; Jegadeesh, 1990)."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Strategy


class MeanReversion(Strategy):
    """Buys the `top_n` most oversold assets: those with the lowest price z-score.

    The z-score measures how far the price is from its own recent average, in units of its
    own recent dispersion:

        z_t = (P_t - mean(P_{t-lookback+1..t})) / std(P_{t-lookback+1..t})

    The idea is that a short-term drop below the average is partly a liquidity effect that
    reverts over the following days or weeks. Lehmann (1990) and Jegadeesh (1990) document
    this reversal for weekly and monthly stock returns; it is the effect that 12-1 momentum
    skips with its last month.

    long_short=True also shorts the `top_n` most overbought (a market-neutral portfolio).
    An asset whose price was flat over the whole window has no z-score and is not traded.
    """

    name = "mean_reversion"

    def __init__(self, lookback: int = 20, top_n: int = 3, long_short: bool = False):
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self.lookback, self.top_n, self.long_short = lookback, top_n, long_short

    def signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        window = prices.rolling(self.lookback)
        std = window.std()
        return (prices - window.mean()) / std.where(std > 0)

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.signal(prices)
        valid = (
            sig.notna().sum(axis=1) >= 2 * self.top_n
            if self.long_short
            else sig.notna().any(axis=1)
        )
        rank_asc = sig.rank(axis=1, ascending=True)
        longs = (rank_asc <= self.top_n).astype(float)
        w = longs.div(longs.sum(axis=1), axis=0)
        if self.long_short:
            rank_desc = sig.rank(axis=1, ascending=False)
            shorts = (rank_desc <= self.top_n).astype(float)
            w = 0.5 * w - 0.5 * shorts.div(shorts.sum(axis=1), axis=0)
        return w.where(valid, 0.0).fillna(0.0)
