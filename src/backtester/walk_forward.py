"""Walk-forward validation: choose a configuration on past data, trade it the next year.

A parameter grid shows what each configuration earned over one history, in sample. It does not
say what choosing the best one would have earned afterwards. Walk-forward answers that: at the
end of each year the configuration with the best Sharpe ratio so far is selected and held for
the following year, so every return of the result comes from a choice made before it.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping

import numpy as np
import pandas as pd

from backtester.engine import run_backtest
from backtester.metrics import PERIODS_PER_YEAR, sharpe
from backtester.strategy import Strategy

MIN_TRAINING_DAYS = 20


def select_by_year(
    returns: pd.DataFrame, min_days: int, rolling_days: int | None = None
) -> pd.DataFrame:
    """Candidate with the best Sharpe ratio at the end of each year, using only the past.

    `returns` holds the daily net returns of the candidates, one column each. A selection is
    made on the last session of every year that is followed by another year in the data (the
    last row of the sample is the end of the data, not of a year). The training window ends on
    the selection day, included, and is:
    - expanding (default): every row since the first one;
    - rolling: the last `rolling_days` rows.
    Years with fewer than `min_days` rows of history are skipped.

    Returns one row per selection day with the label of the chosen column ("selected") and its
    annualized Sharpe ratio over the training window ("train_sharpe"). Candidates without a
    Sharpe ratio (constant returns) are ignored and ties go to the first column.
    """
    if min_days < MIN_TRAINING_DAYS:
        raise ValueError(f"min_days must be at least {MIN_TRAINING_DAYS}, got {min_days}")
    if rolling_days is not None and rolling_days < MIN_TRAINING_DAYS:
        raise ValueError(f"rolling_days must be at least {MIN_TRAINING_DAYS}, got {rolling_days}")

    years = returns.index.year
    is_year_end = np.append(years[1:] != years[:-1], False)  # the last row never qualifies
    rows = []
    for position in np.flatnonzero(is_year_end):
        if position + 1 < min_days:
            continue
        first = 0 if rolling_days is None else max(0, position + 1 - rolling_days)
        training = returns.iloc[first : position + 1].apply(sharpe)
        if training.isna().all():
            raise ValueError(
                f"no candidate has a Sharpe ratio in the window ending {returns.index[position]}"
            )
        rows.append((returns.index[position], training.idxmax(), float(training.max())))
    selected = pd.DataFrame(rows, columns=["date", "selected", "train_sharpe"])
    return selected.set_index("date").rename_axis(returns.index.name)


def splice_returns(returns: pd.DataFrame, selected: pd.Series) -> pd.Series:
    """Returns of the selected candidates joined end to end, as if switching were free.

    `returns` holds the candidates' own daily returns and `selected` the label chosen on each
    selection day (the "selected" column of `select_by_year`). Each choice earns its candidate's
    return from the day after its selection day up to the next selection day, included.

    This is NOT the out-of-sample result: each candidate's return assumes it was already held,
    so the trade from one candidate to the next is never paid. It is a reference to measure
    what that trade costs when the weights go through the engine (see `WalkForward`).
    """
    pieces = []
    ends = [*selected.index[1:], returns.index[-1]]
    for day, label, end in zip(selected.index, selected, ends, strict=True):
        pieces.append(returns.loc[(returns.index > day) & (returns.index <= end), label])
    return pd.concat(pieces).rename("spliced")


class _Orders(Strategy):
    """Returns weights computed beforehand, so a candidate is not evaluated twice."""

    def __init__(self, orders: pd.DataFrame):
        self.orders = orders

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.orders


class WalkForward(Strategy):
    """Each year, holds the candidate with the best Sharpe ratio on the data before that year.

    `candidates` maps a label to a strategy. Each one is backtested with `costs` and
    `rebalance`, all of them from the same day, and `select_by_year` picks one at the end of
    each year. That first day is `start` if given, otherwise the latest first invested day among
    the candidates; give it to make several walk-forwards cover the same periods. `min_days` and
    `rolling_days` are those of `select_by_year`; the default is an expanding window with at
    least five years of data.

    The weights are those of the selected candidate from its selection day on, so the choice
    made at the last close of a year earns the returns of the next year. Before the first
    selection the portfolio is in cash. A candidate that gives no order on its selection day
    (an all-NaN row) is entered on its next order.

    Run the result through `run_backtest` with the same `costs` and `rebalance`: moving from
    one candidate to the next is then a trade like any other and pays its turnover. Splicing
    the candidates' own backtests instead would switch for free.
    """

    name = "walk_forward"

    def __init__(
        self,
        candidates: Mapping[Hashable, Strategy],
        costs: Callable[[pd.Series], pd.Series] | None = None,
        rebalance: str | None = None,
        min_days: int = 5 * PERIODS_PER_YEAR,
        rolling_days: int | None = None,
        start: pd.Timestamp | None = None,
    ):
        if not candidates:
            raise ValueError("at least one candidate is required")
        self.candidates = dict(candidates)
        self.costs, self.rebalance = costs, rebalance
        self.min_days, self.rolling_days, self.start = min_days, rolling_days, start

    def _evaluate(self, prices: pd.DataFrame) -> tuple[dict[Hashable, pd.DataFrame], pd.DataFrame]:
        """Orders of every candidate and their net returns from the common first day."""
        orders = {label: s.weights(prices) for label, s in self.candidates.items()}
        results = {
            label: run_backtest(prices, _Orders(o), self.costs, self.rebalance)
            for label, o in orders.items()
        }
        start = self.start
        if start is None:
            start = max(result.first_active for result in results.values())
        returns = pd.DataFrame({label: r.returns.loc[start:] for label, r in results.items()})
        return orders, returns

    def candidate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Daily net returns of every candidate over the days used for training."""
        return self._evaluate(prices)[1]

    def selection(self, prices: pd.DataFrame) -> pd.DataFrame:
        """The candidate chosen at the end of each year and its training Sharpe ratio."""
        return select_by_year(self.candidate_returns(prices), self.min_days, self.rolling_days)

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        orders, returns = self._evaluate(prices)
        chosen = select_by_year(returns, self.min_days, self.rolling_days)["selected"]
        if chosen.empty:
            raise ValueError("not enough history to make a single selection")

        target = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ends = [*chosen.index[1:], None]
        for day, label, end in zip(chosen.index, chosen, ends, strict=True):
            # From the selection day up to, but not including, the next one.
            held = (prices.index >= day) & (prices.index < end if end is not None else True)
            target.loc[held] = orders[label].reindex_like(prices).loc[held]
        return target
