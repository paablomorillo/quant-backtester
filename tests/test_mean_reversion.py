import numpy as np
import pandas as pd
import pytest

from backtester.data import synthetic_prices
from backtester.strategies import MeanReversion

# Four assets, three days. With lookback=3 only the last row has a signal:
#   A: mean 9,     std sqrt(3) -> z = (7 - 9) / sqrt(3)   = -1.155  (most oversold)
#   B: mean 11,    std 1       -> z = (12 - 11) / 1       = +1.000  (most overbought)
#   C: mean 11,    std 1       -> z = (11 - 11) / 1       =  0.000
#   D: mean 9.167, std 1.041   -> z = (9.5 - 9.167)/1.041 = +0.320
HAND_PRICES = pd.DataFrame(
    {
        "A": [10.0, 10.0, 7.0],
        "B": [10.0, 11.0, 12.0],
        "C": [10.0, 12.0, 11.0],
        "D": [10.0, 8.0, 9.5],
    },
    index=pd.bdate_range("2024-01-01", periods=3),
)


def test_no_lookahead_in_signal():
    """Changing future prices must not alter the weights of past dates."""
    prices = synthetic_prices(n_assets=6, n_days=400)
    strat = MeanReversion(lookback=20, top_n=2)
    w = strat.weights(prices)

    cut = 300
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= np.random.default_rng(5).uniform(
        0.5, 2.0, tampered.iloc[cut + 1 :].shape
    )
    w2 = strat.weights(tampered)

    pd.testing.assert_frame_equal(w.iloc[: cut + 1], w2.iloc[: cut + 1])


def test_zscore_by_hand():
    """Prices 1, 2, 3 with a 3-day window: mean 2, sample std 1, so z = (3 - 2) / 1 = 1."""
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    z = MeanReversion(lookback=3, top_n=1).signal(prices)["A"]
    assert z.iloc[:2].isna().all()
    assert z.iloc[2] == pytest.approx(1.0)


def test_signal_matches_hand_values():
    z = MeanReversion(lookback=3, top_n=1).signal(HAND_PRICES).iloc[-1]
    assert z["A"] == pytest.approx(-2 / np.sqrt(3))
    assert z["B"] == pytest.approx(1.0)
    assert z["C"] == pytest.approx(0.0)
    assert z["D"] == pytest.approx(0.320, abs=1e-3)


def test_long_only_buys_the_most_oversold():
    w = MeanReversion(lookback=3, top_n=1).weights(HAND_PRICES)
    assert (w.iloc[:2] == 0).all(axis=None)  # warm-up: no signal, all cash
    assert w.iloc[-1].to_dict() == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}


def test_long_only_splits_equally_among_top_n():
    w = MeanReversion(lookback=3, top_n=2).weights(HAND_PRICES)
    assert w.iloc[-1].to_dict() == {"A": 0.5, "B": 0.0, "C": 0.5, "D": 0.0}


def test_long_short_also_shorts_the_most_overbought():
    w = MeanReversion(lookback=3, top_n=1, long_short=True).weights(HAND_PRICES)
    assert w.iloc[-1].to_dict() == {"A": 0.5, "B": -0.5, "C": 0.0, "D": 0.0}


def test_long_only_weights_sum_to_one_once_warmed_up():
    prices = synthetic_prices(n_assets=6)
    w = MeanReversion(lookback=20, top_n=2).weights(prices)
    assert (w.iloc[:19].sum(axis=1) == 0).all()
    assert np.allclose(w.iloc[19:].sum(axis=1), 1.0)
    assert (w >= 0).all(axis=None)


def test_long_short_is_dollar_neutral():
    prices = synthetic_prices(n_assets=8)
    w = MeanReversion(lookback=20, top_n=2, long_short=True).weights(prices)
    live = w.iloc[19:]
    assert np.allclose(live.sum(axis=1), 0.0)
    assert np.allclose(live.abs().sum(axis=1), 1.0)


def test_long_short_stays_in_cash_without_enough_assets():
    """Long-short with top_n=2 needs 4 valid signals; 3 assets are not enough."""
    prices = synthetic_prices(n_assets=3)
    w = MeanReversion(lookback=20, top_n=2, long_short=True).weights(prices)
    assert (w == 0).all(axis=None)


def test_flat_price_has_no_signal_and_is_never_held():
    """A constant price has zero deviation: z is undefined (0/0), not 'oversold'."""
    prices = synthetic_prices(n_assets=4, n_days=60)
    prices["A0"] = 100.0
    strat = MeanReversion(lookback=20, top_n=1)
    assert strat.signal(prices)["A0"].isna().all()
    assert (strat.weights(prices)["A0"] == 0).all()


@pytest.mark.parametrize("kwargs", [{"lookback": 1}, {"top_n": 0}])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        MeanReversion(**kwargs)
