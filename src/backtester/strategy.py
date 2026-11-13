"""Interfaz común de todas las estrategias."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Una estrategia transforma precios en pesos objetivo de cartera.

    Contrato:
    - `weights(prices)` devuelve un DataFrame con el mismo índice y columnas que `prices`.
    - La fila t solo puede usar información disponible al cierre de t.
    - El motor aplica los pesos de t a los rendimientos de t+1 (el desfase lo pone el motor,
      no la estrategia).
    - Suma de |pesos| = apalancamiento bruto. NaN se trata como 0.
    """

    name: str = "strategy"

    @abstractmethod
    def weights(self, prices: pd.DataFrame) -> pd.DataFrame: ...


class BuyAndHold(Strategy):
    """Equiponderada y siempre invertida. Sirve de benchmark."""

    name = "buy_and_hold"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        available = prices.notna()
        return available.div(available.sum(axis=1), axis=0).fillna(0.0)
