"""Modelos de costes de transacción."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ProportionalCosts:
    """Coste proporcional al volumen negociado: `bps` puntos básicos por unidad de turnover."""

    bps: float = 5.0

    def __call__(self, turnover: pd.Series) -> pd.Series:
        return turnover * self.bps / 10_000
