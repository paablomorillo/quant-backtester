"""Example: 12-1 momentum on S&P 500 sectors vs. an equal-weight portfolio.

Usage: python examples/run_momentum.py             (real data, needs internet)
       python examples/run_momentum.py --synthetic (simulated data)
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

    print(compare({"Momentum 12-1": mom, "Equal weight": bench}).round(3))

    out = Path("reports")
    out.mkdir(exist_ok=True)
    plot_result(mom, bench, title="12-1 sector momentum vs. equal weight").savefig(
        out / "momentum.png", dpi=150
    )
    print(f"Plot saved to {out / 'momentum.png'}")


if __name__ == "__main__":
    main()
