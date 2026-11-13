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
    """Pesos objetivo constantes para los dos primeros activos."""

    def __init__(self, w_a: float, w_b: float):
        self.w_a, self.w_b = w_a, w_b

    def weights(self, prices):
        w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        w.iloc[:, 0], w.iloc[:, 1] = self.w_a, self.w_b
        return w


def drift_prices() -> pd.DataFrame:
    """A dobla el 1 de febrero y sube un 10 % el 2; B no se mueve.

    Con rebalanceo mensual se opera al cierre del 31 de enero y del 29 de febrero.
    """
    dates = pd.to_datetime(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-29", "2024-03-01"]
    )
    return pd.DataFrame(
        {"A": [100.0, 100.0, 200.0, 220.0, 220.0, 220.0], "B": [100.0] * 6}, index=dates
    )


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


def test_buy_and_hold_never_rebalances():
    """Compra una vez y deja derivar: la equity es la media de los precios normalizados."""
    prices = synthetic_prices(n_assets=4, n_days=300)
    res = run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0))
    expected = prices.div(prices.iloc[0]).mean(axis=1)
    pd.testing.assert_series_equal(res.equity, expected, check_names=False)
    assert not np.allclose(res.weights.iloc[-1], 0.25)


def test_first_order_is_executed_even_off_rebalance_date():
    """La entrada no espera a fin de mes: buy & hold con rebalanceo mensual no queda en efectivo."""
    prices = drift_prices()
    res = run_backtest(prices, BuyAndHold(), costs=ProportionalCosts(0), rebalance="ME")
    jan31, feb2 = prices.index[1], prices.index[3]

    # orden al cierre del 30-ene (no es fin de mes): 50/50 desde el 31-ene
    assert tuple(res.weights.loc[jan31]) == pytest.approx((0.5, 0.5))
    assert res.turnover.loc[jan31] == pytest.approx(1.0)
    # después no se opera: A dobla el 1-feb y los pesos derivan a (2/3, 1/3)
    assert tuple(res.weights.loc[feb2]) == pytest.approx((2 / 3, 1 / 3))
    assert res.turnover.loc[feb2:].sum() == pytest.approx(0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.6)


def test_only_the_first_order_skips_the_rebalance_calendar():
    """Tras la entrada, las órdenes de fechas que no son de rebalanceo se siguen ignorando."""
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(0.5, 0.5), costs=ProportionalCosts(0), rebalance="ME")
    # entrada el 31-ene (orden del 30); el 1-feb no se reequilibra aunque haya orden diaria
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
    """Cambiar precios futuros no debe alterar los pesos de fechas pasadas."""
    prices = synthetic_prices(n_assets=4, n_days=100)
    cut = 60
    tampered = prices.copy()
    tampered.iloc[cut + 1 :] *= 1.5
    pd.testing.assert_frame_equal(
        strategy.weights(prices).iloc[: cut + 1], strategy.weights(tampered).iloc[: cut + 1]
    )


def test_monthly_rebalance_trades_only_after_month_end():
    """Entre rebalanceos los pesos derivan, pero solo se opera el día siguiente a fin de mes."""
    prices = synthetic_prices()
    res = run_backtest(prices, CrossSectionalMomentum(lookback=60, skip=5), rebalance="ME")
    trade_days = res.turnover[res.turnover > 1e-12].index
    month_ends = prices.groupby(pd.Grouper(freq="ME")).tail(1).index
    next_day = {prices.index[prices.index.get_loc(d) + 1] for d in month_ends[:-1]}
    assert len(trade_days) > 0
    assert set(trade_days) <= next_day


# Casos a mano sobre drift_prices(). Tras doblar A, las posiciones valen (2·wA, wB) y el
# efectivo (1 - wA - wB) no cambia; los pesos derivados son posición / valor de la cartera.
DRIFT_CASES = [
    # wA, wB, pesos el 2-feb, rendimiento el 2-feb, turnover al volver al objetivo
    pytest.param(0.5, 0.5, (2 / 3, 1 / 3), 1 / 15, 0.375, id="invertida"),
    pytest.param(0.25, 0.25, (0.4, 0.2), 0.04, 0.3 / 1.3, id="con_efectivo"),
    pytest.param(0.5, -0.5, (2 / 3, -1 / 3), 1 / 15, 0.375, id="long_short"),
]


@pytest.mark.parametrize(("w_a", "w_b", "drifted", "ret", "turnover"), DRIFT_CASES)
def test_weights_drift_between_rebalances(w_a, w_b, drifted, ret, turnover):
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(w_a, w_b), costs=ProportionalCosts(0), rebalance="ME")
    jan31, feb1, feb2, mar1 = prices.index[1], prices.index[2], prices.index[3], prices.index[5]

    # 31-ene: entrada (orden del 30). Los precios no se mueven, así que el rebalanceo de fin
    # de mes no tiene nada que ajustar
    assert res.turnover.loc[jan31] == pytest.approx(abs(w_a) + abs(w_b))

    # 1-feb: a los pesos objetivo; A dobla
    assert tuple(res.weights.loc[feb1]) == pytest.approx((w_a, w_b))
    assert res.gross_returns.loc[feb1] == pytest.approx(w_a * 1.0)
    assert res.turnover.loc[feb1] == pytest.approx(0.0)

    # 2-feb: no se ha operado; los pesos han derivado y A sube un 10 %
    assert tuple(res.weights.loc[feb2]) == pytest.approx(drifted)
    assert res.gross_returns.loc[feb2] == pytest.approx(ret)
    assert res.turnover.loc[feb2] == pytest.approx(0.0)

    # 1-mar: vuelta al objetivo; el turnover se mide contra los pesos ya derivados
    assert tuple(res.weights.loc[mar1]) == pytest.approx((w_a, w_b))
    assert res.turnover.loc[mar1] == pytest.approx(turnover)


def test_drifted_equity_matches_untouched_positions():
    """Sin operar, la cartera vale efectivo + posiciones iniciales revalorizadas."""
    prices = drift_prices()
    res = run_backtest(prices, FixedWeights(0.25, 0.25), costs=ProportionalCosts(0), rebalance="ME")
    # 0,5 de efectivo + 0,25·2,2 en A + 0,25 en B
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
    # 1-feb: A dobla -> (2/3, 1/3). Volver a 50/50 al cierre mueve 2 · 1/6, anotado el 2-feb
    assert res.turnover.loc[prices.index[1]] == pytest.approx(1.0)
    assert res.turnover.loc[feb2] == pytest.approx(1 / 3)
    assert res.gross_returns.loc[feb2] == pytest.approx(0.05)


def test_engine_drift_has_no_lookahead():
    """Cambiar precios futuros no altera pesos, turnover ni rendimientos pasados."""
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

    # la señal existe desde la fila 100 (índice 100) y el motor la aplica un día después
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
    assert len(res.returns) == 60  # el original no se modifica
