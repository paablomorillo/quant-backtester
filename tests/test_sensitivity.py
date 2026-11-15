"""Tests for the parameter sensitivity grid, on synthetic prices."""

import numpy as np
import pandas as pd
import pytest

from backtester import ProportionalCosts, run_backtest
from backtester.data import synthetic_prices
from backtester.metrics import sharpe
from backtester.sensitivity import default_in_context, run_grid
from backtester.strategies import CrossSectionalMomentum, TrendFollowing

GRID = {"lookback": [40, 60, 90], "top_n": [1, 2]}


def build_momentum(lookback: int, top_n: int) -> CrossSectionalMomentum:
    return CrossSectionalMomentum(lookback=lookback, skip=5, top_n=top_n)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_prices(n_assets=5, n_days=400)


@pytest.fixture(scope="module")
def grid(prices):
    return run_grid(prices, build_momentum, GRID, ProportionalCosts(bps=5), rebalance="ME")


def test_grid_has_one_cell_per_combination(grid):
    expected = pd.MultiIndex.from_product(GRID.values(), names=GRID.keys())

    assert grid.sharpe.index.equals(expected)
    assert grid.returns.columns.equals(expected)
    assert grid.sharpe.notna().all()


def test_every_cell_is_measured_from_the_latest_first_active_day(prices, grid):
    """The slowest cell (lookback 90) sets the start, so all cells cover the same days."""
    slowest = run_backtest(prices, build_momentum(90, 1), ProportionalCosts(bps=5), rebalance="ME")

    assert grid.start == slowest.first_active
    assert grid.returns.index[0] == grid.start
    assert grid.returns.notna().all().all()


def test_a_cell_matches_a_backtest_run_by_hand(prices, grid):
    by_hand = run_backtest(
        prices, build_momentum(60, 2), ProportionalCosts(bps=5), rebalance="ME"
    ).since(grid.start)

    pd.testing.assert_series_equal(grid.returns[(60, 2)], by_hand.returns, check_names=False)
    assert grid.sharpe[(60, 2)] == pytest.approx(sharpe(by_hand.returns))


def test_an_explicit_start_is_used_as_given(prices):
    start = prices.index[200]
    grid = run_grid(prices, build_momentum, GRID, rebalance="ME", start=start)
    assert grid.start == start
    assert grid.returns.index[0] == start


def test_a_one_parameter_grid_works(prices):
    grid = run_grid(prices, TrendFollowing, {"window": [20, 50, 100]})

    assert list(grid.sharpe.index) == [20, 50, 100]
    assert grid.sharpe.index.name == "window"
    assert grid.start == run_backtest(prices, TrendFollowing(window=100)).first_active


def test_a_cell_that_never_trades_has_no_sharpe(prices):
    """A strategy that stays in cash has constant returns: its Sharpe ratio is NaN, not 0."""

    class Cash(TrendFollowing):
        def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
            return super().weights(prices) * 0.0

    grid = run_grid(prices, Cash, {"window": [20, 50]}, start=prices.index[60])
    assert grid.sharpe.isna().all()


def test_an_empty_grid_is_rejected(prices):
    with pytest.raises(ValueError):
        run_grid(prices, build_momentum, {})
    with pytest.raises(ValueError):
        run_grid(prices, build_momentum, {"lookback": [], "top_n": [1]})


def test_default_in_context_matches_a_hand_computed_table():
    """3 x 3 table with the default in the centre: its neighbours are the four adjacent cells.

            b=1   b=2   b=3
    a=10    0.1   0.2   0.3
    a=20    0.4   0.9   0.5
    a=30    0.6   0.7   0.8
    """
    index = pd.MultiIndex.from_product([[10, 20, 30], [1, 2, 3]], names=["a", "b"])
    table = pd.Series([0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.6, 0.7, 0.8], index=index)

    context = default_in_context(table, (20, 2))

    assert context["Default"] == 0.9
    assert context["Rank"] == 1
    assert context["Cells"] == 9
    assert context["Grid median"] == 0.5
    assert (context["Grid min"], context["Grid max"]) == (0.1, 0.9)
    assert context["Neighbours mean"] == pytest.approx((0.2 + 0.4 + 0.5 + 0.7) / 4)
    assert context["Neighbours min"] == 0.2
    assert context["Share of cells above zero"] == 1.0


def test_default_in_context_at_a_corner_and_with_missing_cells():
    index = pd.MultiIndex.from_product([[10, 20, 30], [1, 2, 3]], names=["a", "b"])
    table = pd.Series([0.1, np.nan, 0.3, 0.4, 0.9, 0.5, -0.6, 0.7, 0.8], index=index)

    corner = default_in_context(table, (10, 1))  # neighbours: (10, 2) is NaN, (20, 1) = 0.4

    assert corner["Rank"] == 7  # of the 8 cells with a Sharpe ratio
    assert corner["Cells"] == 8
    assert corner["Neighbours mean"] == pytest.approx(0.4)
    assert corner["Share of cells above zero"] == pytest.approx(7 / 8)


def test_default_in_context_on_a_one_parameter_grid():
    table = pd.Series([0.2, 0.5, 0.3, 0.1], index=pd.Index([50, 100, 150, 200], name="window"))

    context = default_in_context(table, 100)

    assert context["Rank"] == 1
    assert context["Neighbours mean"] == pytest.approx(0.25)


def test_default_in_context_rejects_a_default_outside_the_grid():
    table = pd.Series([0.2, 0.5], index=pd.Index([50, 100], name="window"))
    with pytest.raises(ValueError):
        default_in_context(table, 75)
