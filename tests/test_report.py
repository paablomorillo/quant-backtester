import matplotlib.pyplot as plt

from backtester import run_backtest
from backtester.data import synthetic_prices
from backtester.report import plot_result
from backtester.strategies import CrossSectionalMomentum
from backtester.strategy import BuyAndHold


def _results():
    prices = synthetic_prices(n_assets=4, n_days=200)
    mom = run_backtest(prices, CrossSectionalMomentum(lookback=60, skip=5, top_n=2))
    return mom, run_backtest(prices, BuyAndHold())


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_plot_result_shows_benchmark_drawdown_with_custom_labels():
    mom, bench = _results()

    fig = plot_result(mom, bench, labels=("Momentum", "Equiponderado"))
    ax_equity, ax_drawdown = fig.axes

    assert _legend_labels(ax_equity) == ["Momentum", "Equiponderado"]
    assert _legend_labels(ax_drawdown) == ["Momentum", "Equiponderado"]
    plt.close(fig)


def test_plot_result_without_benchmark_keeps_single_series():
    mom, _ = _results()

    fig = plot_result(mom)
    ax_equity, ax_drawdown = fig.axes

    assert _legend_labels(ax_equity) == ["Estrategia"]
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
