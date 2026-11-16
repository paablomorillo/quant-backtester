# Roadmap

Version 1.0 closes the project as planned: an engine, four strategies and a five-notebook study
with its statistical validation. What was left out is listed under "Future work".

## Phase 0 — Skeleton (done)
- [x] Vectorized engine with a t → t+1 lag (anti look-ahead)
- [x] Proportional costs in bps on turnover
- [x] Metrics: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover
- [x] 12-1 cross-sectional momentum (long-only and long-short)
- [x] Daily / weekly / monthly rebalancing
- [x] Tests (including a test that changing the future does not alter the past) and CI

## Phase 1 — First real result (done)
- [x] Run `examples/run_momentum.py` on real sector ETF data
- [x] Notebook `01_sector_momentum.ipynb` with results, plots and interpretation
- [x] README with the equity plot and the metrics table (since v1.0 the table lives in the notebook)
- [x] Weight drift between rebalances

## Phase 2 — More strategies (done)
- [x] Short-term mean reversion (z-score over a moving average)
- [x] Pairs trading: cointegration test (Engle-Granger), spread and entry/exit z-score
- [x] Notebook `03_pairs_trading.ipynb`: four pre-selected pairs, how often each is tradable, and a one-sample Sharpe test against cash
- [x] Time-series trend following (a signal per asset, not cross-sectional)
- [x] Notebook `02_strategies_comparison.ipynb`: momentum, mean reversion and trend following vs. equal weight, with Holm-corrected Sharpe difference tests

## Phase 3 — Statistical rigor (done)
- [x] Walk-forward validation: optimize on a training window, evaluate out of sample (`walk_forward.py`)
- [x] Notebook `05_walk_forward.ipynb`: yearly selection within each sector grid and across all of them, against the pre-fixed parameters and equal weight
- [x] Block bootstrap for Sharpe confidence intervals (paired stationary bootstrap for Sharpe differences, cross-checked with Ledoit-Wolf, and a one-sample version against zero, in `significance.py`)
- [x] Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to correct for the number of trials, in `significance.py`
- [x] Parameter sensitivity analysis: Sharpe grids for the four strategies (`sensitivity.py`, heatmaps in `report.py`)
- [x] Notebook `04_robustness.ipynb`: the four grids, the count of every configuration tried (98) and the DSR of the best one

## Phase 4 — Presentation (done)
- [x] Final README explaining the results, including the negative ones

## Future work
Not done in v1.0. Each item would be a new set of trials and would have to be counted as such.

- **Volatility targeting** as a layer on top of any strategy. Trend following's advantage is
  lower risk, so the fair comparison is against equal weight scaled to the same volatility.
- **More realistic costs** (spread plus impact proportional to volatility). The gross edge of
  momentum is the same size as 5 to 20 bps of costs, so the cost model decides the result.
- **Automatic HTML report per strategy.** It would make every run produce the same tables and
  plots as the notebooks, without copying cells.
- **Interactive demo (Streamlit)** to choose assets, strategy and parameters. It would let a
  reader see how flat the parameter grids are instead of taking the heatmaps on trust.
- **Factors (value, quality, low-vol) with fundamental data.** A universe of hundreds of stocks
  is where these premia are documented; nine sector ETFs are a poor setting for them.
- **Portfolio optimization (minimum variance, risk parity) as a weights layer.** Equal weight was
  the benchmark no strategy beat, so the natural next question is whether a better-built passive
  portfolio does.
- **Event-driven engine** for limit orders and intraday data. Pairs trading and short-term mean
  reversion decay within days, and a daily close-to-close engine cannot test them properly.
