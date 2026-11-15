"""Parameter sensitivity: the same strategy over a grid of parameter values.

A result that only holds for one parameter value is more likely luck than a real effect. The
grid shows whether the chosen value sits on a plateau (its neighbours do about as well) or on
an isolated spike. It is a diagnostic, not a way of choosing parameters: every cell is one more
trial, and picking the best one needs the correction in `significance.deflated_sharpe_ratio`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import pandas as pd

from backtester.engine import run_backtest
from backtester.metrics import sharpe
from backtester.strategy import Strategy


@dataclass(frozen=True)
class GridResult:
    """Outcome of `run_grid`: one backtest per combination of parameter values.

    `sharpe` and the columns of `returns` share the same index: the parameter values, as a
    MultiIndex named after the parameters (a plain Index if there is only one parameter).
    Every cell covers the same days, from `start` on.
    """

    sharpe: pd.Series  # annualized Sharpe ratio after costs of each cell
    returns: pd.DataFrame  # daily net returns of each cell, one column per cell
    start: pd.Timestamp


def run_grid(
    prices: pd.DataFrame,
    build: Callable[..., Strategy],
    grid: Mapping[str, Sequence],
    costs: Callable[[pd.Series], pd.Series] | None = None,
    rebalance: str | None = None,
    start: pd.Timestamp | None = None,
) -> GridResult:
    """Backtest `build(**parameters)` for every combination of the values in `grid`.

    `grid` maps each parameter name to the values to try, e.g. {"lookback": [126, 252],
    "top_n": [1, 3, 5]}. `costs` and `rebalance` are passed to `run_backtest` unchanged.

    All cells are measured over the same period, otherwise a cell with a short warm-up would be
    judged on days that a slow one never traded. By default the period starts on the latest
    first invested day among the cells. Pass `start` for strategies whose first position is not
    the end of a warm-up (e.g. pairs trading, which may wait a long time for its first trade).

    A cell that never trades has constant returns and a NaN Sharpe ratio.
    """
    if not grid or any(len(values) == 0 for values in grid.values()):
        raise ValueError("the grid needs at least one parameter and one value for each")
    if any(len(set(values)) != len(values) for values in grid.values()):
        raise ValueError("the values of a parameter must not be repeated")

    names = list(grid)
    cells = list(product(*grid.values()))
    results = [
        run_backtest(prices, build(**dict(zip(names, cell, strict=True))), costs, rebalance)
        for cell in cells
    ]
    if start is None:
        start = max(result.first_active for result in results)

    if len(names) == 1:
        index = pd.Index([cell[0] for cell in cells], name=names[0])
    else:
        index = pd.MultiIndex.from_tuples(cells, names=names)
    returns = pd.concat([result.since(start).returns for result in results], axis=1)
    returns.columns = index
    return GridResult(
        sharpe=returns.apply(sharpe).rename("Sharpe"), returns=returns, start=pd.Timestamp(start)
    )


def _neighbours(index: pd.Index, cell: tuple) -> list[Hashable]:
    """Cells one step away from `cell` along a single parameter, in the order of the grid."""
    levels = [list(index.get_level_values(level).unique()) for level in range(index.nlevels)]
    found = []
    for level, values in enumerate(levels):
        position = values.index(cell[level])
        for step in (-1, 1):
            if 0 <= position + step < len(values):
                moved = (*cell[:level], values[position + step], *cell[level + 1 :])
                found.append(moved if index.nlevels > 1 else moved[0])
    return [label for label in found if label in index]


def default_in_context(table: pd.Series, default: Hashable) -> pd.Series:
    """Where one cell of a Sharpe grid stands among the rest and among its neighbours.

    `table` is the `sharpe` of a `GridResult` and `default` the label of the cell of interest
    (a tuple for a grid with several parameters). The neighbours are the cells one step away
    along a single parameter: up to two per parameter.

    A plateau shows as a default close to the mean of its neighbours and to the grid median.
    A spike shows as a default far above both. Cells without a Sharpe ratio are ignored, and
    a default without one is rejected.
    """
    if default not in table.index:
        raise ValueError(f"{default} is not a cell of the grid")
    if pd.isna(table.loc[default]):
        raise ValueError(f"the cell {default} has no Sharpe ratio, so it cannot be ranked")
    cell = default if isinstance(default, tuple) else (default,)
    valid = table.dropna()
    neighbours = table.loc[_neighbours(table.index, cell)].dropna()
    return pd.Series(
        {
            "Default": table.loc[default],
            "Rank": int((valid > table.loc[default]).sum()) + 1,
            "Cells": len(valid),
            "Grid min": valid.min(),
            "Grid median": valid.median(),
            "Grid max": valid.max(),
            "Neighbours mean": neighbours.mean(),
            "Neighbours min": neighbours.min(),
            "Share of cells above zero": float((valid > 0).mean()),
        }
    )
