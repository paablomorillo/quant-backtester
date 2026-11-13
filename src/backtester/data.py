"""Carga de precios (con caché local) y generación de datos sintéticos para tests."""

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
    """Precios de cierre ajustados (filas = fechas, columnas = tickers).

    Descarga de Yahoo Finance la primera vez y guarda un parquet en data/.
    """
    key = f"{'_'.join(sorted(tickers))}_{start}_{end or 'today'}.parquet"
    path = CACHE_DIR / key
    if use_cache and path.exists():
        return pd.read_parquet(path)

    import yfinance as yf  # import perezoso: los tests no necesitan red

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.dropna(how="all").sort_index()

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        prices.to_parquet(path)
    return prices


def synthetic_prices(
    n_assets: int = 5,
    n_days: int = 756,
    mu: float = 0.07,
    sigma: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Precios simulados con movimiento browniano geométrico (252 días/año)."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), (n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=[f"A{i}" for i in range(n_assets)])
