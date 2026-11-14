# CLAUDE.md

Personal portfolio project: a vectorized backtesting engine in Python for quantitative
strategies. Author: Pablo (Mathematical Engineering in Data Science, UPF).
The goal is to learn and to show rigor: an honest, modest result is better than a spectacular
Sharpe caused by a methodological error.

## Language
- Everything written to the repo is in **English**: code, comments, docstrings, error messages,
  labels, docs, notebook text and commit messages.
- Pablo may talk to you in Spanish. Reply in the language he uses, but keep the repo in English.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest`
- Lint and format: `ruff check . --fix && ruff format .`
- Example: `python examples/run_momentum.py` (or `--synthetic` without internet)

## Architecture
- `src/backtester/data.py`: download (yfinance, parquet cache in `data/`) and synthetic data
- `src/backtester/strategy.py`: `Strategy` base class; `weights(prices) -> DataFrame` of weights
- `src/backtester/strategies/`: one strategy per file
- `src/backtester/engine.py`: `run_backtest` applies weights, costs and rebalancing
- `src/backtester/metrics.py`: metrics on simple daily returns (252 days/year)
- `src/backtester/report.py`: plots and comparison tables
- `docs/ROADMAP.md`: project phases; tick the boxes as each item is finished

## Non-negotiable rules
1. **No look-ahead.** A strategy at row t only uses data up to the close of t. The lag to t+1
   is done by the engine; strategies NEVER call `shift(-k)`.
2. **Every new strategy gets a "changing the future does not alter the past" test**
   (see `tests/test_momentum.py::test_no_lookahead_in_signal`).
3. **Test before code** for new engine or metrics logic, with cases that can be worked out by
   hand.
4. **No network data in tests**: use `synthetic_prices`.
5. Notebooks only import and present; the logic lives in `src/`.
6. Do not rewrite correct code or shorten it at the expense of clarity; keep the existing structure.
7. When finishing a task: `pytest`, `ruff check .` and `ruff format --check .` green before
   calling anything done.

## Style
- Python 3.11+, type hints, short docstrings in English
- Vectorized pandas; no per-date loops unless unavoidable
- Small commits in English, imperative mood: "Add pairs trading strategy"
