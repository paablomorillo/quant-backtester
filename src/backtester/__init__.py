"""Vectorized backtesting engine for quantitative strategies."""

from backtester.costs import ProportionalCosts
from backtester.engine import BacktestResult, run_backtest
from backtester.metrics import summary
from backtester.strategy import Strategy

__all__ = ["BacktestResult", "ProportionalCosts", "Strategy", "run_backtest", "summary"]
