"""Ejemplo: momentum 12-1 sobre sectores del S&P 500 vs. cartera equiponderada.

Uso:  python examples/run_momentum.py            (datos reales, necesita internet)
      python examples/run_momentum.py --synthetic (datos simulados)
"""

import sys
from pathlib import Path

from backtester import ProportionalCosts, run_backtest
from backtester.data import load_prices, synthetic_prices
from backtester.report import compare, plot_result
from backtester.strategies import CrossSectionalMomentum
from backtester.strategy import EqualWeight

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def main():
    if "--synthetic" in sys.argv:
        prices = synthetic_prices(n_assets=9, n_days=252 * 10)
    else:
        prices = load_prices(SECTOR_ETFS, start="2005-01-01")

    costs = ProportionalCosts(bps=5)
    mom = run_backtest(prices, CrossSectionalMomentum(top_n=3), costs=costs, rebalance="ME")
    bench = run_backtest(prices, EqualWeight(), costs=costs, rebalance="ME")

    print(compare({"Momentum 12-1": mom, "Equiponderado": bench}).round(3))

    out = Path("reports")
    out.mkdir(exist_ok=True)
    plot_result(mom, bench, title="Momentum 12-1 sectores vs. equiponderado").savefig(
        out / "momentum.png", dpi=150
    )
    print(f"Gráfica guardada en {out / 'momentum.png'}")


if __name__ == "__main__":
    main()
