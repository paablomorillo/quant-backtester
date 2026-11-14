"""Performance and risk metrics. All assume simple daily returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def cagr(returns: pd.Series) -> float:
    equity = (1 + returns).prod()
    years = len(returns) / PERIODS_PER_YEAR
    return float(equity ** (1 / years) - 1) if years > 0 else np.nan


def annual_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / PERIODS_PER_YEAR
    std = excess.std(ddof=1)
    return float(excess.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else np.nan


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / PERIODS_PER_YEAR
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(PERIODS_PER_YEAR)) if downside > 0 else np.nan


def drawdown(returns: pd.Series) -> pd.Series:
    """Decline from the previous peak, counting the starting capital (1) as the first peak."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return cagr(returns) / abs(mdd) if mdd < 0 else np.nan


def summary(result) -> pd.Series:
    """Summary table of a BacktestResult."""
    r = result.returns
    return pd.Series(
        {
            "CAGR": cagr(r),
            "Annual volatility": annual_volatility(r),
            "Sharpe": sharpe(r),
            "Sortino": sortino(r),
            "Max drawdown": max_drawdown(r),
            "Calmar": calmar(r),
            "Annual turnover": float(result.turnover.mean() * PERIODS_PER_YEAR),
            "Annual costs": float(result.costs.mean() * PERIODS_PER_YEAR),
        }
    )
