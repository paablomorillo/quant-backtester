import numpy as np
import pandas as pd
import pytest

from backtester.data import synthetic_prices
from backtester.strategies import TrendFollowing

# Three assets, three days. With window=3 only the last row has a signal:
#   UP:   3 > mean(1, 2, 3) = 2 -> long
#   DOWN: 1 < mean(3, 2, 1) = 2 -> cash
#   FLAT: 2 = mean(2, 2, 2) = 2 -> cash (the price must be strictly above its average)
HAND_PRICES = pd.DataFrame(
    {"UP": [1.0, 2.0, 3.0], "DOWN": [3.0, 2.0, 1.0], "FLAT": [2.0, 2.0, 2.0]},
    index=pd.bdate_range("2024-01-01", periods=3),
)


def test_no_lookahead_in_signal():
    """Changing future prices must not alter the weights of past dates."""
    prices = synthetic_prices(n_assets=6, n_days=400)
    strat = TrendFollowing(window=50)
    w = strat.weights(prices)

    cut = 300
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= np.random.default_rng(5).uniform(
        0.5, 2.0, tampered.iloc[cut + 1 :].shape
    )
    w2 = strat.weights(tampered)

    pd.testing.assert_frame_equal(w.iloc[: cut + 1], w2.iloc[: cut + 1])


def test_weights_by_hand():
    w = TrendFollowing(window=3).weights(HAND_PRICES)
    assert (w.iloc[:2] == 0).all(axis=None)  # warm-up: no average yet, all cash
    assert w.iloc[-1].to_dict() == pytest.approx({"UP": 1 / 3, "DOWN": 0.0, "FLAT": 0.0})


def test_signal_is_per_asset_not_cross_sectional():
    """Changing the prices of one asset must not change the weight of any other asset."""
    prices = synthetic_prices(n_assets=5, n_days=300)
    strat = TrendFollowing(window=50)
    w = strat.weights(prices)

    tampered = prices.copy()
    tampered["A1"] *= np.random.default_rng(7).uniform(0.5, 2.0, len(tampered))
    w2 = strat.weights(tampered)

    others = [c for c in prices.columns if c != "A1"]
    pd.testing.assert_frame_equal(w[others], w2[others])
    assert not w["A1"].equals(w2["A1"])  # the tampering did change A1 itself


def test_each_weight_is_zero_or_one_over_n():
    prices = synthetic_prices(n_assets=4)
    w = TrendFollowing(window=50).weights(prices)
    assert (w.iloc[:49] == 0).all(axis=None)
    assert np.isin(w.to_numpy(), [0.0, 0.25]).all()
    assert (w == 0.25).any(axis=None)
    assert (w.sum(axis=1) <= 1.0 + 1e-12).all()


def test_always_rising_assets_are_fully_invested():
    index = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame({"A": np.arange(1.0, 31.0), "B": np.arange(10.0, 40.0)}, index=index)
    w = TrendFollowing(window=5).weights(prices)
    assert (w.iloc[4:] == 0.5).all(axis=None)


def test_asset_without_history_counts_neither_in_signal_nor_in_n():
    """An asset that has not started trading gets no weight and does not dilute the others."""
    index = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": np.arange(1.0, 11.0), "B": [np.nan] * 8 + [5.0, 6.0]}, index=index)
    w = TrendFollowing(window=3).weights(prices)
    assert (w["B"] == 0).all()
    assert (w["A"].iloc[2:8] == 1.0).all()  # A is the only listed asset
    assert (w["A"].iloc[8:] == 0.5).all()  # B is listed now, but has no 3-day average yet


def test_invalid_params():
    with pytest.raises(ValueError):
        TrendFollowing(window=1)
