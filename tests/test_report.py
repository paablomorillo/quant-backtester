import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest
from backtester.data import synthetic_prices
from backtester.report import plot_result, plot_sharpe_heatmap
from backtester.strategies import CrossSectionalMomentum
from backtester.strategy import BuyAndHold


def _results():
    prices = synthetic_prices(n_assets=4, n_days=200)
    mom = run_backtest(prices, CrossSectionalMomentum(lookback=60, skip=5, top_n=2))
    return mom, run_backtest(prices, BuyAndHold())


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def _sharpe_table():
    return pd.DataFrame(
        [[0.10, 0.25, np.nan], [0.40, -0.30, 0.55]],
        index=pd.Index([126, 252], name="lookback"),
        columns=pd.Index([1, 3, 5], name="top_n"),
    )


def test_heatmap_writes_every_sharpe_and_names_the_axes():
    ax = plot_sharpe_heatmap(_sharpe_table(), title="Momentum")

    assert [text.get_text() for text in ax.texts] == [
        "0.10",
        "0.25",
        "n/a",
        "0.40",
        "-0.30",
        "0.55",
    ]
    assert (ax.get_ylabel(), ax.get_xlabel(), ax.get_title()) == ("lookback", "top_n", "Momentum")
    assert [label.get_text() for label in ax.get_xticklabels()] == ["1", "3", "5"]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["126", "252"]
    assert len(ax.patches) == 0  # no default to mark
    plt.close(ax.figure)


def test_heatmap_outlines_the_default_cell():
    ax = plot_sharpe_heatmap(_sharpe_table(), default=(252, 3))

    (outline,) = ax.patches
    assert outline.get_xy() == (0.5, 0.5)  # second row, second column; cells are 1 x 1
    assert (outline.get_width(), outline.get_height()) == (1.0, 1.0)
    plt.close(ax.figure)


def test_heatmap_colours_are_centred_on_zero():
    ax = plot_sharpe_heatmap(_sharpe_table())
    low, high = ax.images[0].get_clim()
    assert -low == high == pytest.approx(0.55)

    ax_fixed = plot_sharpe_heatmap(_sharpe_table(), limit=1.0)
    assert ax_fixed.images[0].get_clim() == (-1.0, 1.0)
    plt.close("all")


def test_heatmap_rejects_a_default_outside_the_table():
    with pytest.raises(ValueError):
        plot_sharpe_heatmap(_sharpe_table(), default=(252, 4))
    plt.close("all")


def test_plot_result_shows_benchmark_drawdown_with_custom_labels():
    mom, bench = _results()

    fig = plot_result(mom, bench, labels=("Momentum", "Equal weight"))
    ax_equity, ax_drawdown = fig.axes

    assert _legend_labels(ax_equity) == ["Momentum", "Equal weight"]
    assert _legend_labels(ax_drawdown) == ["Momentum", "Equal weight"]
    plt.close(fig)


def test_plot_result_without_benchmark_keeps_single_series():
    mom, _ = _results()

    fig = plot_result(mom)
    ax_equity, ax_drawdown = fig.axes

    assert _legend_labels(ax_equity) == ["Strategy"]
    assert ax_drawdown.get_legend() is None
    plt.close(fig)


def test_plot_result_drawdown_axis_covers_benchmark():
    mom, bench = _results()

    fig = plot_result(mom, bench)
    low, high = fig.axes[1].get_ylim()

    worst = (
        min(
            mom.equity.div(mom.equity.cummax()).min(), bench.equity.div(bench.equity.cummax()).min()
        )
        - 1
    )
    assert low <= worst
    assert high == 0
    plt.close(fig)
