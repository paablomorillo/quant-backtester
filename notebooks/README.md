# Notebooks

The five analyses, in reading order: 01_sector_momentum.ipynb, 02_strategies_comparison.ipynb,
03_pairs_trading.ipynb, 04_robustness.ipynb and 05_walk_forward.ipynb. Each one loads its own
data and can be run alone.

Rule: notebooks only import from `backtester` and present results. New logic goes in `src/`
with its test, never defined inside the notebook.
