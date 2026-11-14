# Roadmap

Each phase ends with: tests green, CI green, README updated and a commit/tag.

## Phase 0 — Skeleton (done)
- [x] Vectorized engine with a t → t+1 lag (anti look-ahead)
- [x] Proportional costs in bps on turnover
- [x] Metrics: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover
- [x] 12-1 cross-sectional momentum (long-only and long-short)
- [x] Daily / weekly / monthly rebalancing
- [x] Tests (including a test that changing the future does not alter the past) and CI

## Phase 1 — First real result (1-2 weeks)
- [x] Run `examples/run_momentum.py` on real sector ETF data
- [x] Notebook `01_sector_momentum.ipynb` with results, plots and interpretation
- [x] README with the equity plot and the metrics table
- [x] Weight drift between rebalances

## Phase 2 — More strategies (2-3 weeks)
- [ ] Short-term mean reversion (z-score over a moving average)
- [ ] Pairs trading: cointegration test (Engle-Granger), spread and entry/exit z-score
- [ ] Time-series trend following (a signal per asset, not cross-sectional)
- [ ] Volatility targeting as a layer on top of any strategy

## Phase 3 — Statistical rigor (what sets the project apart)
- [ ] Walk-forward validation: optimize on a training window, evaluate out of sample
- [ ] Block bootstrap for Sharpe confidence intervals
- [ ] Deflated Sharpe Ratio (López de Prado) to correct for the number of trials
- [ ] Parameter sensitivity analysis (lookback × top_n heatmap)
- [ ] More realistic costs: spread + impact proportional to volatility

## Phase 4 — Presentation
- [ ] Automatic HTML report per strategy
- [ ] Interactive demo (Streamlit) to choose assets, strategy and parameters
- [ ] Final post/README explaining the results, including the negative ones

## Ideas for later
- Factors (value, quality, low-vol) with fundamental data
- Portfolio optimization (minimum variance, risk parity) as a weights layer
- Event-driven engine for limit orders and intraday
