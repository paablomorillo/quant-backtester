"""Price loading (with a local cache) and synthetic data generation for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Adjusted close prices (rows = dates, columns = tickers).

    Downloads from Yahoo Finance the first time and saves a parquet file in data/.
    """
    key = f"{'_'.join(sorted(tickers))}_{start}_{end or 'today'}.parquet"
    path = CACHE_DIR / key
    if use_cache and path.exists():
        return pd.read_parquet(path)

    import yfinance as yf  # lazy import: tests need no network

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.dropna(how="all").sort_index()
    check_complete(prices, tickers)  # before caching: a broken download must not be saved

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        prices.to_parquet(path)
    return prices


def check_complete(prices: pd.DataFrame, tickers: list[str]) -> None:
    """Raise ValueError if a ticker is missing or its column is empty.

    yfinance does not raise when one ticker fails to download: it returns an all-NaN column.
    """
    missing = [t for t in tickers if t not in prices.columns or prices[t].notna().sum() == 0]
    if missing:
        raise ValueError(f"Incomplete download, no data for: {', '.join(missing)}")


def _ticker_quality(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    inside = series.loc[valid.index[0] : valid.index[-1]]
    return pd.Series(
        {
            "First date": valid.index[0],
            "Last date": valid.index[-1],
            "Observations": len(valid),
            "Internal NaNs": int(inside.isna().sum()),
            "Non-positive prices": int((valid <= 0).sum()),
            "Largest |daily return|": float(valid.pct_change().abs().max()),
            "Largest gap (days)": int(valid.index.to_series().diff().dt.days.max()),
        }
    )


def quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker quality check (one row per asset).

    "Internal NaNs" counts gaps between each asset's first and last valid date (a late
    start is not a gap). "Largest gap" is calendar days between two consecutive
    observations: 3-4 is normal (weekends and holidays).
    """
    return pd.DataFrame(
        {ticker: _ticker_quality(prices[ticker]) for ticker in prices.columns}
    ).T.infer_objects()


def synthetic_prices(
    n_assets: int = 5,
    n_days: int = 756,
    mu: float = 0.07,
    sigma: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Prices simulated with geometric Brownian motion (252 days/year)."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), (n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=[f"A{i}" for i in range(n_assets)])
