"""Gráficas e informe de un backtest."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from backtester.engine import BacktestResult
from backtester.metrics import drawdown, summary


def plot_result(result: BacktestResult, benchmark: BacktestResult | None = None, title: str = ""):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    result.equity.plot(ax=ax1, label="Estrategia")
    if benchmark is not None:
        benchmark.equity.plot(ax=ax1, label="Benchmark", alpha=0.7)
    ax1.set_ylabel("Equity (base 1)")
    ax1.set_title(title)
    ax1.legend()
    drawdown(result.returns).plot(ax=ax2, kind="area", alpha=0.4, color="tab:red")
    ax2.set_ylabel("Drawdown")
    fig.tight_layout()
    return fig


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Tabla de métricas con una columna por estrategia."""
    return pd.DataFrame({name: summary(r) for name, r in results.items()})
