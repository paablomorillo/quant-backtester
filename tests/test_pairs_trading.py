"""Tests for pairs trading, on synthetic pairs whose cointegration is known by construction."""

import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_backtest
from backtester.strategies import PairsTrading
from backtester.strategies.pairs_trading import hedge_ratio, walk_positions

# Log prices built so that the regression can be solved on paper:
#   log Y = 1 + 2 * log X + e,  with e = (0.1, -0.1, -0.1, 0.1).
# e sums to 0 and is orthogonal to log X = (0, 1, 2, 3), so OLS recovers alpha = 1, beta = 2
# exactly and e is the residual. Its sample std (ddof=1) is sqrt(0.04 / 3), so the z-score of
# the last day is 0.1 / sqrt(0.04 / 3) = sqrt(3) / 2.
HAND_LOG_X = np.array([0.0, 1.0, 2.0, 3.0])
HAND_RESIDUAL = np.array([0.1, -0.1, -0.1, 0.1])
HAND_LOG_Y = 1.0 + 2.0 * HAND_LOG_X + HAND_RESIDUAL


def cointegrated_pair(
    n_days: int = 1500, beta: float = 1.5, phi: float = 0.7, seed: int = 0
) -> pd.DataFrame:
    """X is a random walk in logs; Y = 0.5 + beta * X + a stationary AR(1) spread.

    The spread reverts fast (phi = 0.7). With phi = 0.9 the Engle-Granger test on 120 days
    only detects the cointegration in about half of the windows.
    """
    rng = np.random.default_rng(seed)
    log_x = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, n_days))
    spread = np.zeros(n_days)
    shocks = rng.normal(0.0, 0.01, n_days)
    for t in range(1, n_days):
        spread[t] = phi * spread[t - 1] + shocks[t]
    index = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame({"Y": np.exp(0.5 + beta * log_x + spread), "X": np.exp(log_x)}, index=index)


def random_walk_pair(n_days: int, seed: int) -> pd.DataFrame:
    """Two independent random walks in logs: no long-run relationship at all."""
    rng = np.random.default_rng(seed)
    logs = np.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, (n_days, 2)), axis=0)
    index = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(np.exp(logs), index=index, columns=["Y", "X"])


def fast_strategy(**kwargs) -> PairsTrading:
    """Shorter windows than the defaults, to keep the tests quick."""
    params = {"formation": 120, "reestimate": 20, "max_holding": 40} | kwargs
    return PairsTrading([("Y", "X")], **params)


# --- no look-ahead ----------------------------------------------------------------------


def test_no_lookahead_in_signal():
    """Changing future prices must not alter the weights of past dates."""
    prices = cointegrated_pair(n_days=700)
    strat = fast_strategy()
    w = strat.weights(prices)

    cut = 500
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= np.random.default_rng(5).uniform(
        0.5, 2.0, tampered.iloc[cut + 1 :].shape
    )
    w2 = strat.weights(tampered)

    assert w.iloc[: cut + 1].notna().any(axis=None)  # there is something to compare
    pd.testing.assert_frame_equal(w.iloc[: cut + 1], w2.iloc[: cut + 1])


def test_no_lookahead_in_zscore_and_tradable():
    prices = cointegrated_pair(n_days=500)
    strat = fast_strategy()
    cut = 350
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= 1.3

    pd.testing.assert_frame_equal(
        strat.zscore(prices).iloc[: cut + 1], strat.zscore(tampered).iloc[: cut + 1]
    )
    pd.testing.assert_frame_equal(
        strat.tradable(prices).iloc[: cut + 1], strat.tradable(tampered).iloc[: cut + 1]
    )


# --- spread and z-score by hand ---------------------------------------------------------


def test_hedge_ratio_by_hand():
    alpha, beta, residual_std = hedge_ratio(HAND_LOG_Y, HAND_LOG_X)
    assert alpha == pytest.approx(1.0)
    assert beta == pytest.approx(2.0)
    assert residual_std == pytest.approx(np.sqrt(0.04 / 3))


def test_zscore_by_hand():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"Y": np.exp(HAND_LOG_Y), "X": np.exp(HAND_LOG_X)}, index=index)
    z = PairsTrading([("Y", "X")], formation=4, reestimate=1).zscore(prices)

    assert list(z.columns) == ["Y/X"]
    assert z["Y/X"].iloc[:3].isna().all()  # no full formation window yet
    assert z["Y/X"].iloc[-1] == pytest.approx(np.sqrt(3) / 2)


def test_zscore_keeps_formation_parameters_until_next_reestimation():
    """Between re-estimations the spread is measured with the frozen alpha, beta and std."""
    prices = cointegrated_pair(n_days=200)
    z = PairsTrading([("Y", "X")], formation=100, reestimate=50).zscore(prices)["Y/X"]

    log_y, log_x = np.log(prices["Y"].to_numpy()), np.log(prices["X"].to_numpy())
    alpha, beta, residual_std = hedge_ratio(log_y[:100], log_x[:100])  # rows 0..99
    day = 130  # inside the block that starts at row 99
    assert z.iloc[day] == pytest.approx((log_y[day] - alpha - beta * log_x[day]) / residual_std)


# --- position rules on constructed z paths ----------------------------------------------


def positions_for(z: list[float], tradable: list[bool] | None = None, **kwargs):
    params = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_holding": 10} | kwargs
    flags = np.ones(len(z), dtype=bool) if tradable is None else np.array(tradable)
    return walk_positions(np.array(z), flags, **params)


def test_enters_short_spread_above_entry_and_exits_inside_band():
    position, reason = positions_for([0.0, 2.5, 1.5, 0.6, 0.4, 0.0])
    assert position.tolist() == [0, -1, -1, -1, 0, 0]
    assert reason.tolist() == ["", "", "", "", "target", ""]


def test_enters_long_spread_below_minus_entry():
    position, _ = positions_for([0.0, -2.1, -1.0, -0.4])
    assert position.tolist() == [0, 1, 1, 0]


def test_exits_when_spread_overshoots_to_the_other_side():
    position, reason = positions_for([2.5, -3.0, -3.0])
    # The short-spread trade has more than converged, so it is closed. A trade in the other
    # direction can only start on a later day.
    assert position.tolist() == [-1, 0, 1]
    assert reason.tolist() == ["", "target", ""]


def test_stop_loss_closes_and_blocks_reentry_until_spread_normalizes():
    position, reason = positions_for([2.5, 4.5, 3.0, 2.5, 0.3, 2.5])
    assert position.tolist() == [-1, 0, 0, 0, 0, -1]
    assert reason.tolist() == ["", "stop", "", "", "", ""]


def test_no_entry_beyond_the_stop_level():
    position, _ = positions_for([5.0, 3.0])
    assert position.tolist() == [0, -1]


def test_max_holding_closes_and_blocks_reentry():
    position, reason = positions_for([2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 2.5], max_holding=3)
    # Entered at day 0 and held 3 days (1, 2, 3): the close of day 3 is the time-out.
    assert position.tolist() == [-1, -1, -1, 0, 0, 0, -1]
    assert reason.tolist() == ["", "", "", "time", "", "", ""]


def test_position_is_closed_when_pair_stops_being_tradable():
    position, reason = positions_for([2.5, 2.5, 2.5, 2.5], tradable=[True, True, False, False])
    assert position.tolist() == [-1, -1, 0, 0]
    assert reason.tolist() == ["", "", "not_cointegrated", ""]


def test_no_entry_while_not_tradable_or_without_zscore():
    position, _ = positions_for([3.0, np.nan, 3.0], tradable=[False, True, True])
    assert position.tolist() == [0, 0, -1]


# --- synthetic pairs --------------------------------------------------------------------


def test_cointegrated_pair_is_tradable_and_trades():
    prices = cointegrated_pair()
    strat = fast_strategy()

    tradable = strat.tradable(prices)["Y/X"]
    assert tradable.iloc[:119].sum() == 0  # warm-up: no formation window yet
    assert tradable.iloc[119:].mean() > 0.7

    trades = strat.trades(prices)
    assert len(trades) >= 5
    assert set(trades["side"]) == {"long", "short"}


def test_random_walk_pair_is_rarely_tradable():
    """A 5% test passes by chance now and then; over a long sample it must stay rare."""
    prices = random_walk_pair(n_days=3000, seed=1)
    tradable = fast_strategy().tradable(prices)["Y/X"]
    assert tradable.iloc[119:].mean() < 0.15


def test_random_walk_pair_does_not_trade():
    """In this seeded sample no formation window passes the test, so nothing is traded."""
    prices = random_walk_pair(n_days=600, seed=3)
    strat = fast_strategy()

    assert not strat.tradable(prices)["Y/X"].any()
    assert strat.trades(prices).empty
    assert strat.weights(prices).isna().all(axis=None)  # never an order
    assert (run_backtest(prices, strat).returns == 0).all()


# --- weights ----------------------------------------------------------------------------


def test_legs_have_opposite_signs_in_proportion_one_to_beta_and_gross_one():
    prices = cointegrated_pair()
    strat = fast_strategy()
    w = strat.weights(prices)
    beta = strat.hedge_ratios(prices)["Y/X"]

    open_orders = w[(w.abs().sum(axis=1) > 0)]
    assert len(open_orders) > 0
    assert (np.sign(open_orders["Y"]) == -np.sign(open_orders["X"])).all()
    assert open_orders.abs().sum(axis=1).to_numpy() == pytest.approx(1.0)
    ratio = (open_orders["X"].abs() / open_orders["Y"].abs()).to_numpy()
    assert ratio == pytest.approx(beta.loc[open_orders.index].to_numpy())


def test_orders_only_on_days_the_target_changes():
    """No order (all-NaN row) while nothing changes, so the engine lets positions drift."""
    prices = cointegrated_pair()
    strat = fast_strategy()
    w = strat.weights(prices)
    position = strat.positions(prices)["Y/X"]
    beta = strat.hedge_ratios(prices)["Y/X"]

    changed = position.ne(position.shift(1, fill_value=0)) | (
        (position != 0) & beta.ne(beta.shift(1))
    )
    pd.testing.assert_series_equal(w.notna().all(axis=1), changed, check_names=False)
    assert (w[position == 0].fillna(0.0) == 0).all(axis=None)


def test_short_spread_is_short_y_and_long_x():
    prices = cointegrated_pair()
    strat = fast_strategy()
    w = strat.weights(prices)
    position = strat.positions(prices)["Y/X"]

    entries = w[(position == -1) & w.notna().all(axis=1)]
    assert (entries["Y"] < 0).all()
    assert (entries["X"] > 0).all()


def test_gross_exposure_is_shared_among_active_pairs():
    """Two copies of the same pair are always active together: each gets half the book."""
    base = cointegrated_pair()
    prices = base.assign(Y2=base["Y"], X2=base["X"])
    strat = PairsTrading([("Y", "X"), ("Y2", "X2")], formation=120, reestimate=20, max_holding=40)
    w = strat.weights(prices)

    open_orders = w[w.abs().sum(axis=1) > 0]
    assert open_orders.abs().sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert open_orders[["Y", "X"]].abs().sum(axis=1).to_numpy() == pytest.approx(0.5)
    pd.testing.assert_series_equal(w["Y"], w["Y2"], check_names=False)


def test_assets_outside_the_pairs_get_no_weight():
    prices = cointegrated_pair().assign(OTHER=100.0)
    w = fast_strategy().weights(prices)
    assert list(w.columns) == ["Y", "X", "OTHER"]
    assert (w["OTHER"].fillna(0.0) == 0).all()


def test_missing_prices_make_the_window_untradable():
    prices = cointegrated_pair(n_days=400)
    prices.iloc[:150, 0] = np.nan  # Y lists late
    strat = fast_strategy()
    tradable = strat.tradable(prices)["Y/X"]
    assert not tradable.iloc[: 150 + 119].any()
    assert strat.weights(prices).iloc[: 150 + 119].isna().all(axis=None)


def test_backtest_on_cointegrated_pair_earns_before_costs():
    """Sanity check of the signs end to end: a truly mean-reverting spread must make money."""
    prices = cointegrated_pair()
    result = run_backtest(prices, fast_strategy())
    assert np.isfinite(result.returns).all()
    assert result.gross_returns.sum() > 0


# --- validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"formation": 2},
        {"reestimate": 0},
        {"entry_z": 0.5, "exit_z": 0.5},
        {"stop_z": 2.0},
        {"exit_z": -0.1},
        {"max_holding": 0},
        {"p_value": 0.0},
        {"p_value": 1.0},
    ],
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        PairsTrading([("Y", "X")], **kwargs)


@pytest.mark.parametrize("pairs", [[], [("Y", "Y")], [("Y", "X"), ("Y", "X")], [("Y",)]])
def test_invalid_pairs(pairs):
    with pytest.raises(ValueError):
        PairsTrading(pairs)


def test_unknown_ticker_is_reported():
    with pytest.raises(ValueError, match="MISSING"):
        PairsTrading([("Y", "MISSING")]).weights(cointegrated_pair(n_days=300))
