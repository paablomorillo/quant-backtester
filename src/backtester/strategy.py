"""Common interface for all strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """A strategy turns prices into target portfolio weights.

    Contract:
    - `weights(prices)` returns a DataFrame with the same index and columns as `prices`.
    - Row t may only use information available at the close of t.
    - The engine applies the weights of t to the returns of t+1 (the lag is added by the
      engine, not by the strategy).
    - Sum of |weights| = gross leverage; whatever is missing up to 1 is cash.
    - An all-NaN row means "no order": the engine does not trade and lets the weights
      drift. A single NaN in a row with other values is treated as 0.
    """

    name: str = "strategy"

    @abstractmethod
    def weights(self, prices: pd.DataFrame) -> pd.DataFrame: ...


class EqualWeight(Strategy):
    """Equal-weighted and always invested: back to 1/n at each rebalance. Used as a benchmark."""

    name = "equal_weight"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        available = prices.notna()
        return available.div(available.sum(axis=1), axis=0).fillna(0.0)


class BuyAndHold(Strategy):
    """Buys equal parts on the first day and never trades again; the weights drift.

    It only gives an order on the first row, which the engine executes with any `rebalance`.
    Assets that are not trading yet on that day are never bought.
    """

    name = "buy_and_hold"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        equal = EqualWeight().weights(prices)
        is_entry = pd.Series(prices.index == prices.index[0], index=prices.index)
        return equal.where(is_entry, axis=0)
