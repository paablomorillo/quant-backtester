"""Plots and report of a backtest."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.engine import BacktestResult
from backtester.metrics import drawdown, summary


def plot_result(
    result: BacktestResult,
    benchmark: BacktestResult | None = None,
    title: str = "",
    labels: tuple[str, str] = ("Strategy", "Benchmark"),
):
    """Equity (top) and drawdown (bottom) of the strategy and, if given, of the benchmark."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    result.equity.plot(ax=ax1, label=labels[0])
    if benchmark is not None:
        benchmark.equity.plot(ax=ax1, label=labels[1], alpha=0.7)
    ax1.set_ylabel("Equity (base 1)")
    ax1.set_title(title)
    ax1.legend()
    drawdown(result.returns).plot(
        ax=ax2, kind="area", alpha=0.4, color="tab:red", label=labels[0], legend=False
    )
    if benchmark is not None:
        drawdown(benchmark.returns).plot(ax=ax2, color="tab:orange", lw=1, label=labels[1])
        ax2.legend(loc="lower right")
        ax2.autoscale(axis="y")  # the area pins the axis to the strategy; benchmark may fall more
        ax2.set_ylim(top=0)
    ax2.set_xlabel("")
    ax2.set_ylabel("Drawdown")
    fig.tight_layout()
    return fig


def plot_sharpe_heatmap(
    table: pd.DataFrame,
    default: tuple | None = None,
    title: str = "",
    limit: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of a Sharpe table (rows and columns are the values of two parameters).

    Every cell shows its Sharpe ratio and `default`, a (row label, column label) pair, is
    outlined in black. Colours are centred on zero: blue above, red below, from -limit to
    +limit (by default the largest absolute value in the table). Pass the same `limit` to
    several heatmaps to make their colours comparable. A cell without a value reads "n/a".
    """
    if default is not None and (default[0] not in table.index or default[1] not in table.columns):
        raise ValueError(f"{default} is not a cell of the table")
    values = table.to_numpy(dtype=float)
    if np.isnan(values).all():
        raise ValueError("the table has no values to plot")
    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 2, 0.6 * table.shape[0] + 1.5))
    if limit is None:
        limit = float(np.nanmax(np.abs(values)))
    ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")

    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        text = "n/a" if np.isnan(value) else f"{value:.2f}"
        # Dark cells (far from zero) need white text to stay readable.
        color = "white" if abs(value) > 0.6 * limit else "black"
        ax.text(col, row, text, ha="center", va="center", color=color, fontsize=9)
    if default is not None:
        row, col = table.index.get_loc(default[0]), table.columns.get_loc(default[1])
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="black", lw=2.5))

    ax.set_xticks(range(table.shape[1]), [str(label) for label in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(label) for label in table.index])
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    ax.set_title(title)
    return ax


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Metrics table with one column per strategy."""
    return pd.DataFrame({name: summary(r) for name, r in results.items()})
