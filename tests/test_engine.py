import numpy as np
import pandas as pd
import pytest

from backtester import ProportionalCosts, Strategy, run_backtest
from backtester.data import synthetic_prices
from backtester.strategies import CrossSectionalMomentum
from backtester.strategy import BuyAndHold


class AllInFirst(Strategy):
    def weights(self, prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w.iloc[:, 0] = 1.0
        return w


class TodaysWinners(Strategy):
    """Compra lo que ha subido hoy. Si el motor aplicara los pesos el mismo día
    (sin desfase), ganaría de forma imposible porque cobraría el rendimiento que ya conoce."""

    def weights(self, prices):
        today = prices.pct_change(fill_method=None)
        return (today > 0).astype(float).div(len(prices.columns))


def test_weights_are_applied_next_day():
    prices = synthetic_prices(n_assets=2, n_days=50)
    res = run_backtest(prices, AllInFirst(), costs=ProportionalCosts(0))
    expected = prices.iloc[:, 0].pct_change().fillna(0.0)
    expected.iloc[0] = 0.0  # el primer día aún no hay posición
    pd.testing.assert_series_equal(res.returns, expected, check_names=False)


def test_engine_shift_prevents_same_day_lookahead():
    """En un paseo aleatorio, comprar lo que subió hoy no da ventaja si se opera mañana."""
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


def test_monthly_rebalance_changes_weights_only_at_month_end():
    prices = synthetic_prices()
    res = run_backtest(prices, CrossSectionalMomentum(lookback=60, skip=5), rebalance="ME")
    changes = res.weights.diff().abs().sum(axis=1)
    change_days = changes[changes > 0].index
    month_ends = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    next_day = {prices.index[prices.index.get_loc(d) + 1] for d in month_ends[:-1]}
    assert set(change_days) <= next_day
