"""Vectorized backtesting engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import pandas as pd

from backtester.costs import ProportionalCosts
from backtester.strategy import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series  # daily net portfolio return
    gross_returns: pd.Series  # before costs
    weights: pd.DataFrame  # effective (already lagged) weights each day
    turnover: pd.Series  # daily sum |Δw|
    costs: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def first_active(self) -> pd.Timestamp:
        """First day with an open position (before that, the strategy is in cash)."""
        invested = self.weights.abs().sum(axis=1) > 0
        return invested.idxmax()

    def since(self, start) -> BacktestResult:
        """Copy of the result trimmed to dates >= start (equity restarts at base 1).

        Used to compare strategies over the same period, e.g. without the warm-up in which
        one of them has no signal yet.
        """
        return BacktestResult(**{f.name: getattr(self, f.name).loc[start:] for f in fields(self)})


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    costs: Callable[[pd.Series], pd.Series] | None = None,
    rebalance: str | None = None,
) -> BacktestResult:
    """Run a strategy over a price matrix.

    rebalance: None = trade every day; "W", "ME", ... = only on the last day of each period.
    The first row with an order is always executed, even off a rebalance date: it is the
    entry. No trade happens on dates where the strategy gives no order (an all-NaN row).

    On a trade, the portfolio is set to the target weights. Between trades it is left alone:
    each weight drifts with its asset's return and cash (1 - sum of weights) earns nothing.
    Turnover is measured against the drifted weights, so rebalancing has a cost even when
    the target does not change. It is booked on the first day the new weights are held.

    If an asset is missing a price, it is valued at the last known one and the return over
    the gap is earned when it trades again. Before its first price, its return is 0.
    """
    costs = costs or ProportionalCosts()
    # A price gap is valued at the last known price (ffill only looks backwards);
    # the accumulated return is earned on the day a price is available again.
    asset_returns = prices.ffill().pct_change(fill_method=None).fillna(0.0)
    growth = (1 + asset_returns).cumprod()

    orders = strategy.weights(prices).reindex_like(prices)
    has_order = orders.notna().any(axis=1)
    is_trade = has_order
    if rebalance is not None:
        rebal_dates = orders.groupby(pd.Grouper(freq=rebalance)).tail(1).index
        is_entry = has_order & (has_order.cumsum() == 1)
        is_trade = has_order & (orders.index.isin(rebal_dates) | is_entry)
    target = orders.fillna(0.0)

    # Weights and price level of the last trade before each close (shift: excluding today).
    last_weights = target.where(is_trade, axis=0).ffill().shift(1).fillna(0.0)
    last_growth = growth.where(is_trade, axis=0).ffill().shift(1).fillna(1.0)

    # Weights at the close of t before trading: revalued positions / portfolio value.
    positions = last_weights * growth / last_growth
    cash = 1 - last_weights.sum(axis=1)
    before_trade = positions.div(cash + positions.sum(axis=1), axis=0)
    after_trade = target.where(is_trade, before_trade, axis=0)

    # Anti look-ahead key: the decision at the close of t is applied to the return of t+1.
    held = after_trade.shift(1).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)
    turnover = (after_trade - before_trade).abs().sum(axis=1).shift(1).fillna(0.0)
    cost = costs(turnover)

    return BacktestResult(
        returns=gross - cost,
        gross_returns=gross,
        weights=held,
        turnover=turnover,
        costs=cost,
    )
