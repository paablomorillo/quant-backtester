import numpy as np
import pandas as pd
import pytest

from backtester.data import synthetic_prices
from backtester.strategies import CrossSectionalMomentum


def test_no_lookahead_in_signal():
    """Cambiar precios futuros no debe alterar los pesos de fechas pasadas."""
    prices = synthetic_prices(n_assets=6, n_days=400)
    strat = CrossSectionalMomentum(lookback=120, skip=10, top_n=2)
    w = strat.weights(prices)

    cut = 300
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= np.random.default_rng(5).uniform(
        0.5, 2.0, tampered.iloc[cut + 1 :].shape
    )
    w2 = strat.weights(tampered)

    pd.testing.assert_frame_equal(w.iloc[: cut + 1], w2.iloc[: cut + 1])


def test_long_only_weights_sum_to_one_once_warmed_up():
    prices = synthetic_prices(n_assets=6)
    w = CrossSectionalMomentum(lookback=120, skip=10, top_n=2).weights(prices)
    assert (w.iloc[:120].sum(axis=1) == 0).all()
    assert np.allclose(w.iloc[120:].sum(axis=1), 1.0)


def test_long_short_is_dollar_neutral():
    prices = synthetic_prices(n_assets=8)
    w = CrossSectionalMomentum(lookback=120, skip=10, top_n=2, long_short=True).weights(prices)
    live = w.iloc[120:]
    assert np.allclose(live.sum(axis=1), 0.0)
    assert np.allclose(live.abs().sum(axis=1), 1.0)


def test_invalid_params():
    with pytest.raises(ValueError):
        CrossSectionalMomentum(lookback=10, skip=10)
