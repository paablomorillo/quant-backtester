"""Tests for walk-forward selection, on synthetic data with a known answer."""

import numpy as np
import pandas as pd
import pytest

from backtester import ProportionalCosts, run_backtest
from backtester.data import synthetic_prices
from backtester.metrics import sharpe
from backtester.strategies import TrendFollowing
from backtester.strategy import Strategy
from backtester.walk_forward import WalkForward, select_by_year, splice_returns


class Tilt(Strategy):
    """Holds `share` of the portfolio in the first asset and the rest in the second."""

    def __init__(self, share: float):
        self.share = share

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w.iloc[:, 0], w.iloc[:, 1] = self.share, 1.0 - self.share
        return w


def two_candidate_returns() -> pd.DataFrame:
    """Daily returns of two candidates over 2020-2022 with the same +-1% noise.

    A earns an extra 0.2% a day in 2020 and nothing after; B earns an extra 0.1% a day from
    2021 on. Over 2020-2021 together A is still ahead (0.1% a day against 0.05%); over the
    last year alone B is.
    """
    index = pd.bdate_range("2020-01-01", "2022-06-30")
    noise = np.where(np.arange(len(index)) % 2 == 0, 0.01, -0.01)
    year = index.year
    return pd.DataFrame(
        {"A": noise + np.where(year == 2020, 0.002, 0.0), "B": noise + (year >= 2021) * 0.001},
        index=index,
    )


def year_end(returns: pd.DataFrame, year: int) -> pd.Timestamp:
    return returns.index[returns.index.year == year][-1]


# --- select_by_year ---------------------------------------------------------------------


def test_selection_happens_on_the_last_session_of_each_complete_year():
    returns = two_candidate_returns()

    chosen = select_by_year(returns, min_days=200)

    # 2022 stops in June: its last row is the end of the sample, not the end of a year.
    assert list(chosen.index) == [year_end(returns, 2020), year_end(returns, 2021)]
    assert list(chosen.columns) == ["selected", "train_sharpe"]


def test_expanding_window_keeps_the_candidate_that_is_ahead_since_the_start():
    returns = two_candidate_returns()

    chosen = select_by_year(returns, min_days=200)

    assert list(chosen["selected"]) == ["A", "A"]
    upto_2021 = returns.loc[: year_end(returns, 2021)]
    assert chosen["train_sharpe"].iloc[1] == pytest.approx(sharpe(upto_2021["A"]))


def test_rolling_window_switches_to_the_candidate_that_is_ahead_lately():
    returns = two_candidate_returns()

    chosen = select_by_year(returns, min_days=200, rolling_days=250)

    assert list(chosen["selected"]) == ["A", "B"]
    last_250 = returns.loc[: year_end(returns, 2021)].iloc[-250:]
    assert chosen["train_sharpe"].iloc[1] == pytest.approx(sharpe(last_250["B"]))


def test_years_without_enough_history_are_not_selected():
    returns = two_candidate_returns()

    assert list(select_by_year(returns, min_days=300).index) == [year_end(returns, 2021)]
    assert select_by_year(returns, min_days=5000).empty


def test_selection_does_not_change_when_later_returns_change():
    returns = two_candidate_returns()
    tampered = returns.copy()
    tampered.loc[tampered.index > year_end(returns, 2020), "B"] += 0.05  # B becomes far better

    before = select_by_year(returns, min_days=200)
    after = select_by_year(tampered, min_days=200)

    pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])
    assert after["selected"].iloc[1] == "B"  # and the change is visible once it is in the past


def test_ties_go_to_the_first_candidate_and_constant_candidates_are_skipped():
    returns = two_candidate_returns()
    tied = pd.DataFrame({"first": returns["A"], "second": returns["A"]})
    with_cash = returns.assign(cash=0.0)[["cash", "A", "B"]]

    assert list(select_by_year(tied, min_days=200)["selected"]) == ["first", "first"]
    assert list(select_by_year(with_cash, min_days=200)["selected"]) == ["A", "A"]
    with pytest.raises(ValueError):
        select_by_year(with_cash[["cash"]], min_days=200)


def test_invalid_windows_are_rejected():
    returns = two_candidate_returns()
    with pytest.raises(ValueError):
        select_by_year(returns, min_days=1)
    with pytest.raises(ValueError):
        select_by_year(returns, min_days=200, rolling_days=1)


# --- WalkForward ------------------------------------------------------------------------


def good_and_noisy_prices(seed: int = 0) -> pd.DataFrame:
    """Two assets over 2020-2024: one with a clear upward drift, one with no drift, noisier.

    Any money moved from the first to the second lowers the mean and adds independent variance,
    so the truly best `Tilt` is share = 1.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", "2024-12-31")
    good = rng.normal(0.001, 0.005, len(index))
    noisy = rng.normal(0.0, 0.02, len(index))
    return pd.DataFrame(
        {"good": 100 * np.cumprod(1 + good), "noisy": 100 * np.cumprod(1 + noisy)}, index=index
    )


def tilts() -> dict[float, Tilt]:
    return {share: Tilt(share) for share in (0.0, 0.5, 1.0)}


def test_the_truly_better_parameter_is_selected_every_year():
    prices = good_and_noisy_prices()
    walk = WalkForward(tilts(), min_days=250)

    chosen = walk.selection(prices)

    assert len(chosen) == 4  # end of 2020, 2021, 2022 and 2023; 2024 is the end of the sample
    assert (chosen["selected"] == 1.0).all()


def test_weights_are_those_of_the_selected_candidate_from_the_selection_day_on():
    prices = good_and_noisy_prices()
    walk = WalkForward(tilts(), min_days=250)

    weights = walk.weights(prices)
    first_selection = walk.selection(prices).index[0]

    assert (weights.loc[weights.index < first_selection] == 0.0).all().all()  # cash until then
    expected = Tilt(1.0).weights(prices).loc[first_selection:]
    pd.testing.assert_frame_equal(weights.loc[first_selection:], expected)


def test_out_of_sample_returns_start_the_session_after_the_first_selection():
    prices = good_and_noisy_prices()
    walk = WalkForward(tilts(), min_days=250)

    result = run_backtest(prices, walk)
    first_selection = walk.selection(prices).index[0]

    assert result.first_active == prices.index[prices.index.get_loc(first_selection) + 1]
    assert result.first_active.year == 2021


def regime_prices() -> pd.DataFrame:
    """Asset "early" rises 0.2% a day in 2020-2021 and is flat after; "late" does the opposite.

    Both carry the same +-1% alternating noise so that their Sharpe ratios are defined.
    """
    index = pd.bdate_range("2020-01-01", "2024-12-31")
    noise = np.where(np.arange(len(index)) % 2 == 0, 0.01, -0.01)
    first_half = index.year <= 2021
    early = noise + np.where(first_half, 0.002, 0.0)
    late = noise + np.where(first_half, 0.0, 0.002)
    return pd.DataFrame(
        {"early": 100 * np.cumprod(1 + early), "late": 100 * np.cumprod(1 + late)}, index=index
    )


def test_switching_between_candidates_pays_turnover_once_through_the_engine():
    prices = regime_prices()
    candidates = {"early": Tilt(1.0), "late": Tilt(0.0)}
    costs = ProportionalCosts(bps=10)
    walk = WalkForward(candidates, costs=costs, min_days=250, rolling_days=250)

    chosen = walk.selection(prices)
    result = run_backtest(prices, walk, costs)

    # Selected at the end of 2020 and 2021: "early". At the end of 2022 and 2023: "late".
    assert list(chosen["selected"]) == ["early", "early", "late", "late"]
    switch_day = chosen.index[2]
    booked = prices.index[prices.index.get_loc(switch_day) + 1]  # first day of the new weights
    assert result.turnover[booked] == pytest.approx(2.0)  # sell all of one, buy all of the other
    assert result.costs[booked] == pytest.approx(2.0 * 10 / 10_000)
    # Turnover happens only on entry (1.0) and on the switch (2.0).
    assert result.turnover.sum() == pytest.approx(3.0)

    # Splicing the candidates' own backtests misses the entry and the switch: 3.0 of turnover.
    early = run_backtest(prices, candidates["early"], costs).returns
    late = run_backtest(prices, candidates["late"], costs).returns
    spliced = pd.concat([early.loc[result.first_active : switch_day], late.loc[booked:]])
    out_of_sample = result.returns.loc[result.first_active :]
    assert out_of_sample.sum() == pytest.approx(spliced.sum() - 3.0 * 10 / 10_000)


def test_the_switch_is_traded_at_the_year_end_with_monthly_rebalancing():
    """The configuration of the notebooks: orders every day, trades only at month ends."""
    prices = regime_prices()
    candidates = {"early": Tilt(1.0), "late": Tilt(0.0)}
    walk = WalkForward(candidates, rebalance="ME", min_days=250, rolling_days=250)

    chosen = walk.selection(prices)
    weights = walk.weights(prices)
    result = run_backtest(prices, walk, rebalance="ME")

    switch_day = chosen.index[2]
    booked = prices.index[prices.index.get_loc(switch_day) + 1]
    # The boundary is exact: "early" up to the day before the switch, "late" from that day on.
    before = (weights.index >= chosen.index[0]) & (weights.index < switch_day)
    assert (weights.loc[before, "early"] == 1.0).all()
    assert (weights.loc[switch_day:, "late"] == 1.0).all()
    assert result.turnover[booked] == pytest.approx(2.0)
    assert result.weights.loc[booked, "late"] == pytest.approx(1.0)


class SilentAtYearEnd(Tilt):
    """A `Tilt` that gives no order (an all-NaN row) on the last session of each year."""

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        years = prices.index.year
        last_of_year = np.append(years[1:] != years[:-1], True)
        orders = super().weights(prices)
        orders.loc[last_of_year] = np.nan
        return orders


def test_a_selected_candidate_without_an_order_closes_the_previous_one():
    """No order on the selection day must not leave the previous candidate held."""
    prices = regime_prices()
    candidates = {"early": Tilt(1.0), "late": SilentAtYearEnd(0.0)}
    walk = WalkForward(candidates, min_days=250, rolling_days=250)

    chosen = walk.selection(prices)
    result = run_backtest(prices, walk)

    switch_day = chosen.index[2]
    assert chosen["selected"].iloc[2] == "late"
    booked = prices.index[prices.index.get_loc(switch_day) + 1]
    assert (result.weights.loc[booked] == 0.0).all()  # in cash for one day, not still in "early"
    assert result.weights.iloc[prices.index.get_loc(booked) + 1]["late"] == pytest.approx(1.0)


def test_splicing_an_empty_selection_is_rejected():
    returns = two_candidate_returns()
    with pytest.raises(ValueError):
        splice_returns(returns, select_by_year(returns, min_days=5000)["selected"])


def test_no_lookahead_in_selection_or_weights():
    """Changing prices after a selection day alters neither earlier selections nor weights."""
    prices = synthetic_prices(n_assets=5, n_days=1500)  # 2020-01 to 2025-09
    candidates = {window: TrendFollowing(window=window) for window in (20, 50, 100, 200)}
    walk = WalkForward(candidates, ProportionalCosts(bps=5), rebalance="ME", min_days=300)

    selection = walk.selection(prices)
    cut = selection.index[2]
    later = prices.index > cut
    tampered = prices.copy()
    tampered.loc[later] *= np.linspace(1.0, 3.0, later.sum())[:, None]

    assert len(selection) >= 4
    pd.testing.assert_frame_equal(walk.selection(tampered).loc[:cut], selection.loc[:cut])
    pd.testing.assert_frame_equal(walk.weights(tampered).loc[:cut], walk.weights(prices).loc[:cut])
    # The tampering is not a no-op: the weights after the cut do change.
    assert not walk.weights(tampered).loc[later].equals(walk.weights(prices).loc[later])


def test_candidates_are_compared_over_the_same_days():
    """The slowest candidate sets the start of the training data for all of them."""
    prices = synthetic_prices(n_assets=5, n_days=1500)
    candidates = {window: TrendFollowing(window=window) for window in (20, 200)}
    walk = WalkForward(candidates, min_days=300)

    returns = walk.candidate_returns(prices)
    selection = walk.selection(prices)

    assert returns.index[0] == run_backtest(prices, candidates[200]).first_active
    assert list(returns.columns) == [20, 200]
    first_training = returns.loc[: selection.index[0]]
    assert selection["train_sharpe"].iloc[0] == pytest.approx(first_training.apply(sharpe).max())


def test_an_explicit_start_sets_the_first_training_day():
    """Used to give several walk-forwards the same training and out-of-sample periods."""
    prices = synthetic_prices(n_assets=5, n_days=1500)
    candidates = {window: TrendFollowing(window=window) for window in (20, 50)}
    start = prices.index[400]  # 2021-07: the end of 2022 is the first year end with 300 days

    walk = WalkForward(candidates, min_days=300, start=start)

    assert walk.candidate_returns(prices).index[0] == start
    assert walk.selection(prices).index[0].year == 2022
    assert WalkForward(candidates, min_days=300).selection(prices).index[0].year == 2021


def test_spliced_returns_are_the_selected_candidates_own_returns_year_by_year():
    prices = regime_prices()
    candidates = {"early": Tilt(1.0), "late": Tilt(0.0)}
    walk = WalkForward(candidates, min_days=250, rolling_days=250)
    returns, chosen = walk.candidate_returns(prices), walk.selection(prices)

    spliced = splice_returns(returns, chosen["selected"])

    switch_day = chosen.index[2]
    assert spliced.index[0] == returns.index[returns.index.get_loc(chosen.index[0]) + 1]
    assert spliced.index[-1] == returns.index[-1]
    # The choice made on a selection day earns the candidate's return from the next day on.
    before = returns["early"].loc[spliced.index[0] : switch_day]
    pd.testing.assert_series_equal(spliced.loc[:switch_day], before, check_names=False)
    after = returns.index[returns.index > switch_day]
    pd.testing.assert_series_equal(
        spliced.loc[after], returns.loc[after, "late"], check_names=False
    )


def test_walk_forward_rejects_bad_input():
    prices = good_and_noisy_prices()
    with pytest.raises(ValueError):
        WalkForward({}, min_days=250)
    with pytest.raises(ValueError):
        WalkForward(tilts(), min_days=5000).weights(prices)  # never enough history to select
