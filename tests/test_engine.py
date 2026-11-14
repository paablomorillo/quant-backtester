import numpy as np
import pandas as pd
import pytest

from backtester import ProportionalCosts, Strategy, run_backtest
from backtester.data import synthetic_prices
from backtester.strategies import CrossSectionalMomentum
from backtester.strategy import BuyAndHold, EqualWeight


class AllInFirst(Strategy):
    def weights(self, prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w.iloc[:, 0] = 1.0
        return w


class FixedWeights(Strategy):
    """Constant target weights for the first two assets."""

    def __init__(self, w_a: float, w_b: float):
        self.w_a, self.w_b = w_a, w_b

    def weights(self, prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w.iloc[:, 0], w.iloc[:, 1] = self.w_a, self.w_b
        return w


def drift_prices() -> pd.DataFrame:
    """A doubles on February 1 and rises 10% on the 2nd; B does not move.

    With monthly rebalancing, trades happen at the close of January 31 and February 29.
    """
    dates = pd.to_datetime(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-29", "2024-03-01"]
    )
    return pd.DataFrame(
        {"A": [100.0, 100.0, 200.0, 220.0, 220.0, 220.0], "B": [100.0] * 6}, index=dates
    )


class TodaysWinners(Strategy):
    """Buys what went up today. If the engine applied the weights on the same day
    (no lag), it would win impossibly, because it would earn the return it already knows."""

    def weights(self, prices):
        today = prices.pct_change(fill_method=None)
        return (today > 0).astype(float).div(len(prices.columns))


def test_weights_are_applied_next_day():
    prices = synthetic_prices(n_assets=2, n_days=50)
    res = run_backtest(prices, AllInFirst(), costs=ProportionalCosts(0))
    expected = prices.iloc[:, 0].pct_change().fillna(0.0)
    expected.iloc[0] = 0.0  # no position yet on the first day
    pd.testing.assert_series_equal(res.returns, expected, check_names=False)


def test_engine_shift_prevents_same_day_lookahead():
    """In a random walk, buying what rose today gives no edge if you trade tomorrow."""
    prices = synthetic_prices(n_assets=10, n_days=2000, mu=0.0, seed=1)
    res = run_backtest(prices, TodaysWinners(), costs=ProportionalCosts(0))
    assert abs(res.returns.mean() * 252) < 0.05


def test_costs_reduce_returns():
    prices = synthetic_prices()
    strat = CrossSectionalMomentum(lookback=60, skip=5, top_n=2)
    free = run_backtest(prices, strat, costs=ProportionalCosts(0))
    paid = run_backtest(prices, strat, costs=ProportionalCosts(20))
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert np.isclose((free.returns - paid.returns).sum(), paid.costs.sum())


def test_buy_and_hold_turnover_only_on_entry():
    prices = synthetic_prices(n_assets=4)
    res = run_backtest(prices, BuyAndHold())
    assert res.turnover.iloc[1] == pytest.approx(1.0)
    assert res.turnover.iloc[2:].sum() == pytest.approx(0.0)


def test_buy_and_hold_never_rebalances():
    """Buys once and lets it drift: equity is the mean of the normalized prices."""
    prices = synthetic_prices(n_assets=4, n_days=300)
    res = run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0))
    expected = prices.div(prices.iloc[0]).mean(axis=1)
    pd.testing.assert_series_equal(res.equity, expected, check_names=False)
    assert not np.allclose(res.weights.iloc[-1], 0.25)


def test_first_order_is_executed_even_off_rebalance_date():
    """The entry does not wait for month end: monthly buy & hold is not left in cash."""
    prices = drift_prices()
    res = run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0), rebalance="ME")
    jan31, feb2 = prices.index[1], prices.index[3]

    # order at the Jan 30 close (not a month end): 50/50 from Jan 31
    assert tuple(res.weights.loc[jan31]) == pytest.approx((0.5, 0.5))
    assert res.turnover.loc[jan31] == pytest.approx(1.0)
    # no trades afterwards: A doubles on Feb 1 and the weights drift to (2/3, 1/3)
    assert tuple(res.weights.loc[feb2]) == pytest.approx((2 / 3, 1 / 3))
    assert res.turnover.loc[feb2:].sum() == pytest.approx(0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.6)


def test_only_the_first_order_skips_the_rebalance_calendar():
    """After the entry, orders on non-rebalance dates are still ignored."""
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(0.5, 0.5), costs=ProportionalCosts(0), rebalance="ME")
    # entry on Jan 31 (order of the 30th); no rebalance on Feb 1 despite the daily order
    assert res.turnover.loc[prices.index[1]] == pytest.approx(1.0)
    assert res.turnover.loc[prices.index[2] : prices.index[4]].sum() == pytest.approx(0.0)


def test_equal_weight_rebalances_back_to_target():
    prices = synthetic_prices(n_assets=4, n_days=300)
    res = run_backtest(prices, EqualWeight(), rebalance="ME")
    month_ends = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    day_after = prices.index[prices.index.get_loc(month_ends[2]) + 1]
    assert np.allclose(res.weights.loc[day_after], 0.25)
    assert res.turnover.loc[day_after] > 0


@pytest.mark.parametrize("strategy", [BuyAndHold(), EqualWeight()])
def test_benchmarks_have_no_lookahead(strategy):
    """Changing future prices must not alter the weights of past dates."""
    prices = synthetic_prices(n_assets=4, n_days=100)
    cut = 60
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= 1.5
    pd.testing.assert_frame_equal(
        strategy.weights(prices).iloc[: cut + 1], strategy.weights(tampered).iloc[: cut + 1]
    )


def test_monthly_rebalance_trades_only_after_month_end():
    """Between rebalances the weights drift, but trades only happen the day after month end."""
    prices = synthetic_prices()
    res = run_backtest(prices, CrossSectionalMomentum(lookback=60, skip=5), rebalance="ME")
    trade_days = res.turnover[res.turnover > 1e-12].index
    month_ends = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    next_day = {prices.index[prices.index.get_loc(d) + 1] for d in month_ends[:-1]}
    assert len(trade_days) > 0
    assert set(trade_days) <= next_day


# Hand-worked cases on drift_prices(). After A doubles, the positions are worth (2·wA, wB)
# and cash (1 - wA - wB) is unchanged; drifted weights are position / portfolio value.
DRIFT_CASES = [
    # wA, wB, weights on Feb 2, return on Feb 2, turnover when going back to target
    pytest.param(0.5, 0.5, (2 / 3, 1 / 3), 1 / 15, 0.375, id="fully_invested"),
    pytest.param(0.25, 0.25, (0.4, 0.2), 0.04, 0.3 / 1.3, id="with_cash"),
    pytest.param(0.5, -0.5, (2 / 3, -1 / 3), 1 / 15, 0.375, id="long_short"),
]


@pytest.mark.parametrize(("w_a", "w_b", "drifted", "ret", "turnover"), DRIFT_CASES)
def test_weights_drift_between_rebalances(w_a, w_b, drifted, ret, turnover):
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(w_a, w_b), costs=ProportionalCosts(0), rebalance="ME")
    jan31, feb1, feb2, mar1 = prices.index[1], prices.index[2], prices.index[3], prices.index[5]

    # Jan 31: entry (order of the 30th). Prices do not move, so the month-end rebalance
    # has nothing to adjust
    assert res.turnover.loc[jan31] == pytest.approx(abs(w_a) + abs(w_b))

    # Feb 1: at the target weights; A doubles
    assert tuple(res.weights.loc[feb1]) == pytest.approx((w_a, w_b))
    assert res.gross_returns.loc[feb1] == pytest.approx(w_a * 1.0)
    assert res.turnover.loc[feb1] == pytest.approx(0.0)

    # Feb 2: no trade; the weights have drifted and A rises 10%
    assert tuple(res.weights.loc[feb2]) == pytest.approx(drifted)
    assert res.gross_returns.loc[feb2] == pytest.approx(ret)
    assert res.turnover.loc[feb2] == pytest.approx(0.0)

    # Mar 1: back to target; turnover is measured against the drifted weights
    assert tuple(res.weights.loc[mar1]) == pytest.approx((w_a, w_b))
    assert res.turnover.loc[mar1] == pytest.approx(turnover)


def test_drifted_equity_matches_untouched_positions():
    """Without trading, the portfolio is worth cash + the revalued initial positions."""
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(0.25, 0.25), costs=ProportionalCosts(0), rebalance="ME")
    # 0.5 in cash + 0.25·2.2 in A + 0.25 in B
    assert res.equity.loc[prices.index[4]] == pytest.approx(1.3)


def test_daily_rebalance_keeps_target_weights_and_gross_returns():
    prices = synthetic_prices(n_assets=5, n_days=300)
    strat = CrossSectionalMomentum(lookback=60, skip=5, top_n=2)
    res = run_backtest(prices, strat, costs=ProportionalCosts(0))

    held = strat.weights(prices).shift(1).fillna(0.0)
    expected = (held * prices.pct_change().fillna(0.0)).sum(axis=1)
    pd.testing.assert_frame_equal(res.weights, held)
    pd.testing.assert_series_equal(res.gross_returns, expected)


def test_daily_rebalance_pays_for_undoing_one_day_of_drift():
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(0.5, 0.5), costs=ProportionalCosts(0))
    feb2 = prices.index[3]
    # Feb 1: A doubles -> (2/3, 1/3). Back to 50/50 at the close moves 2 · 1/6, booked Feb 2
    assert res.turnover.loc[prices.index[1]] == pytest.approx(1.0)
    assert res.turnover.loc[feb2] == pytest.approx(1 / 3)
    assert res.gross_returns.loc[feb2] == pytest.approx(0.05)


def test_price_gap_does_not_lose_the_return():
    """A lone NaN is valued at the last known price: A's 10% arrives when the price returns."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, np.nan, 110.0, 110.0, 110.0], "B": [100.0] * 5}, index=idx)
    res = run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0))
    # 50/50 at the close of day 0; A rises 10% between day 0 and day 2
    assert res.gross_returns.tolist() == pytest.approx([0.0, 0.0, 0.05, 0.0, 0.0])
    assert res.equity.iloc[-1] == pytest.approx(1.05)


def test_asset_without_history_has_no_return_on_its_first_price():
    """Before the first price there is no return to earn, even if the asset has weight."""
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [np.nan, np.nan, 100.0, 110.0], "B": [100.0] * 4}, index=idx)
    res = run_backtest(prices, FixedWeights(0.5, 0.5), costs=ProportionalCosts(0))
    assert res.gross_returns.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.05])


def test_trailing_nans_warn_with_ticker_and_last_date():
    """An asset that stops trading before the end is flagged by name and last price date."""
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, np.nan, np.nan], "B": [100.0] * 5}, index=idx)
    with pytest.warns(UserWarning, match=r"A.*2024-01-03") as record:
        run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0))
    assert len(record) == 1
    assert "B" not in str(record[0].message)


def test_trailing_nans_warning_names_every_affected_ticker():
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "A": [100.0, np.nan, np.nan, np.nan],
            "B": [100.0, 100.0, np.nan, np.nan],
            "C": [100.0] * 4,
        },
        index=idx,
    )
    with pytest.warns(UserWarning) as record:
        run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0))
    message = str(record[0].message)
    assert "A" in message and "2024-01-01" in message
    assert "B" in message and "2024-01-02" in message
    assert "C" not in message


def test_interior_gaps_and_leading_nans_do_not_warn(recwarn):
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {"A": [100.0, np.nan, 110.0, 110.0, 110.0], "B": [np.nan, np.nan, 100.0, 100.0, 100.0]},
        index=idx,
    )
    run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0))
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_engine_drift_has_no_lookahead():
    """Changing future prices does not alter past weights, turnover or returns."""
    prices = synthetic_prices(n_assets=5, n_days=300)
    strat = CrossSectionalMomentum(lookback=60, skip=5, top_n=2)
    cut = 200
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= np.random.default_rng(3).uniform(
        0.5, 2.0, tampered.iloc[cut + 1 :].shape
    )
    a = run_backtest(prices, strat, rebalance="ME")
    b = run_backtest(tampered, strat, rebalance="ME")
    pd.testing.assert_frame_equal(a.weights.iloc[: cut + 1], b.weights.iloc[: cut + 1])
    pd.testing.assert_series_equal(a.returns.iloc[: cut + 1], b.returns.iloc[: cut + 1])
    pd.testing.assert_series_equal(a.turnover.iloc[: cut + 1], b.turnover.iloc[: cut + 1])


def test_first_active_is_first_day_with_position():
    prices = synthetic_prices(n_assets=5, n_days=400)
    res = run_backtest(prices, CrossSectionalMomentum(lookback=100, skip=10, top_n=2))

    # the signal exists from row 100 (index 100) and the engine applies it one day later
    assert res.first_active == prices.index[101]
    assert (res.weights.loc[: prices.index[100]] == 0).all().all()


def test_since_trims_every_series_and_restarts_equity():
    prices = synthetic_prices(n_assets=3, n_days=60)
    res = run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(5))
    start = prices.index[20]

    cut = res.since(start)

    for name in ("returns", "gross_returns", "weights", "turnover", "costs"):
        pd.testing.assert_index_equal(getattr(cut, name).index, prices.index[20:])
    pd.testing.assert_series_equal(cut.returns, res.returns.loc[start:])
    assert cut.equity.iloc[0] == pytest.approx(1 + res.returns.loc[start])
    assert len(res.returns) == 60  # the original is not modified
