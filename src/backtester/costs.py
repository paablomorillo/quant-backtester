"""Transaction cost models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ProportionalCosts:
    """Cost proportional to traded volume: `bps` basis points per unit of turnover."""

    bps: float = 5.0

    def __call__(self, turnover: pd.Series) -> pd.Series:
        return turnover * self.bps / 10_000
