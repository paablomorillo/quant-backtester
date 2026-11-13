"""Motor de backtesting vectorizado."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    costs: Callable[[pd.Series], pd.Series] | None = None,
    rebalance: str | None = None,
) -> BacktestResult:
    """Ejecuta una estrategia sobre una matriz de precios.

    rebalance: None = rebalanceo diario; "W", "ME", ... = solo en el último día de cada periodo
    (entre medias se mantienen los pesos objetivo; la deriva de pesos se ignora en la v0).
    """
    costs = costs or ProportionalCosts()
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

    target = strategy.weights(prices).reindex_like(prices).fillna(0.0)
    if rebalance is not None:
        rebal_dates = target.groupby(pd.Grouper(freq=rebalance)).tail(1).index
        is_rebal = pd.Series(target.index.isin(rebal_dates), index=target.index)
        target = target.where(is_rebal, axis=0).ffill().fillna(0.0)

    # Clave anti look-ahead: la decisión del cierre de t se aplica al rendimiento de t+1.
    held = target.shift(1).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    cost = costs(turnover)

    return BacktestResult(
        returns=gross - cost,
        gross_returns=gross,
        weights=held,
        turnover=turnover,
        costs=cost,
    )
