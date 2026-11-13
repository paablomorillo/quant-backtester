import numpy as np
import pandas as pd
import pytest

from backtester import metrics


def test_cagr_constant_daily_return():
    r = pd.Series([0.001] * 252)
    assert metrics.cagr(r) == pytest.approx(1.001**252 - 1)


def test_max_drawdown_known_path():
    # equity: 1.0 -> 1.2 -> 0.9 -> 1.1  => peor caída 0.9/1.2 - 1 = -25 %
    r = pd.Series([0.2, 0.9 / 1.2 - 1, 1.1 / 0.9 - 1])
    assert metrics.max_drawdown(r) == pytest.approx(-0.25)


def test_drawdown_counts_a_loss_from_the_starting_capital():
    # equity: 1.0 -> 0.5 -> 0.5 -> 0.75. El máximo previo es el capital inicial (1,0).
    r = pd.Series([-0.5, 0.0, 0.5])
    assert metrics.drawdown(r).tolist() == pytest.approx([-0.5, -0.5, -0.25])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_sharpe_scales_with_sqrt_time():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 5000))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe(r) == pytest.approx(expected)


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(metrics.sharpe(pd.Series([0.0] * 10)))
