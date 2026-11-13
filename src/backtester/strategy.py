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
    - Suma de |pesos| = apalancamiento bruto; lo que falta hasta 1 es efectivo.
    - Una fila entera de NaN significa "sin orden": el motor no opera y deja derivar los
      pesos. Un NaN suelto en una fila con otros valores se trata como 0.
    """

    name: str = "strategy"

    @abstractmethod
    def weights(self, prices: pd.DataFrame) -> pd.DataFrame: ...


class EqualWeight(Strategy):
    """Equiponderada y siempre invertida: en cada rebalanceo vuelve a 1/n. Sirve de benchmark."""

    name = "equal_weight"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        available = prices.notna()
        return available.div(available.sum(axis=1), axis=0).fillna(0.0)


class BuyAndHold(Strategy):
    """Compra a partes iguales el primer día y no vuelve a operar; los pesos derivan.

    Solo da orden en la primera fila, así que se usa con rebalanceo diario (`rebalance=None`).
    Los activos que aún no cotizan ese día no se compran nunca.
    """

    name = "buy_and_hold"

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        equal = EqualWeight().weights(prices)
        is_entry = pd.Series(prices.index == prices.index[0], index=prices.index)
        return equal.where(is_entry, axis=0)
