"""Motor de backtesting vectorizado."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import pandas as pd

from backtester.costs import ProportionalCosts
from backtester.strategy import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series  # rendimiento neto diario de la cartera
    gross_returns: pd.Series  # antes de costes
    weights: pd.DataFrame  # pesos efectivos (ya desfasados) en cada día
    turnover: pd.Series  # sum |Δw| diario
    costs: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def first_active(self) -> pd.Timestamp:
        """Primer día con posición abierta (antes, la estrategia está en efectivo)."""
        invested = self.weights.abs().sum(axis=1) > 0
        return invested.idxmax()

    def since(self, start) -> BacktestResult:
        """Copia del resultado recortada a las fechas >= start (la equity vuelve a base 1).

        Sirve para comparar estrategias sobre el mismo periodo, p. ej. sin el calentamiento
        en el que una de ellas aún no tiene señal.
        """
        return BacktestResult(**{f.name: getattr(self, f.name).loc[start:] for f in fields(self)})


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    costs: Callable[[pd.Series], pd.Series] | None = None,
    rebalance: str | None = None,
) -> BacktestResult:
    """Ejecuta una estrategia sobre una matriz de precios.

    rebalance: None = se opera cada día; "W", "ME", ... = solo en el último día de cada periodo.
    La primera fila con orden se ejecuta siempre, aunque no caiga en fecha de rebalanceo: es la
    entrada. No se opera en las fechas en que la estrategia no da orden (fila entera de NaN).

    Al operar, la cartera se ajusta a los pesos objetivo. Entre operaciones no se toca: cada
    peso deriva con el rendimiento de su activo y el efectivo (1 - suma de pesos) no rinde.
    El turnover se mide contra los pesos ya derivados, así que reequilibrar tiene coste aunque
    el objetivo no cambie. Se anota el primer día en que se mantienen los pesos nuevos.
    """
    costs = costs or ProportionalCosts()
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    growth = (1 + asset_returns).cumprod()

    orders = strategy.weights(prices).reindex_like(prices)
    has_order = orders.notna().any(axis=1)
    is_trade = has_order
    if rebalance is not None:
        rebal_dates = orders.groupby(pd.Grouper(freq=rebalance)).tail(1).index
        is_entry = has_order & (has_order.cumsum() == 1)
        is_trade = has_order & (orders.index.isin(rebal_dates) | is_entry)
    target = orders.fillna(0.0)

    # Pesos y nivel de precios de la última operación anterior a cada cierre (shift: sin hoy).
    last_weights = target.where(is_trade, axis=0).ffill().shift(1).fillna(0.0)
    last_growth = growth.where(is_trade, axis=0).ffill().shift(1).fillna(1.0)

    # Pesos al cierre de t antes de operar: posiciones revalorizadas / valor de la cartera.
    positions = last_weights * growth / last_growth
    cash = 1 - last_weights.sum(axis=1)
    before_trade = positions.div(cash + positions.sum(axis=1), axis=0)
    after_trade = target.where(is_trade, before_trade, axis=0)

    # Clave anti look-ahead: la decisión del cierre de t se aplica al rendimiento de t+1.
    held = after_trade.shift(1).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)
    turnover = (after_trade - before_trade).abs().sum(axis=1).shift(1).fillna(0.0)
    cost = costs(turnover)

    return BacktestResult(
        returns=gross - cost,
        gross_returns=gross,
        weights=held,
        turnover=turnover,
        costs=cost,
    )
