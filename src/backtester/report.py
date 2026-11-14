"""Plots and report of a backtest."""

from __future__ import annotations

import matplotlib.pyplot as plt
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


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Metrics table with one column per strategy."""
    return pd.DataFrame({name: summary(r) for name, r in results.items()})
