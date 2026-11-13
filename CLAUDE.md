# CLAUDE.md

Proyecto personal de portfolio: motor de backtesting vectorizado en Python para estrategias
cuantitativas. Autor: Pablo (Ingeniería Matemática en Ciencia de Datos, UPF).
El objetivo es aprender y enseñar rigor: es preferible un resultado honesto y modesto a un
Sharpe espectacular por un error metodológico.

## Comandos
- Instalar: `pip install -e ".[dev]"`
- Tests: `pytest`
- Lint y formato: `ruff check . --fix && ruff format .`
- Ejemplo: `python examples/run_momentum.py` (o `--synthetic` sin internet)

## Arquitectura
- `src/backtester/data.py`: descarga (yfinance, caché parquet en `data/`) y datos sintéticos
- `src/backtester/strategy.py`: clase base `Strategy`; `weights(prices) -> DataFrame` de pesos
- `src/backtester/strategies/`: una estrategia por fichero
- `src/backtester/engine.py`: `run_backtest` aplica pesos, costes y rebalanceo
- `src/backtester/metrics.py`: métricas sobre rendimientos diarios simples (252 días/año)
- `src/backtester/report.py`: gráficas y tablas comparativas
- `docs/ROADMAP.md`: fases del proyecto; marcar casillas al terminar cada punto

## Reglas no negociables
1. **Sin look-ahead.** Una estrategia en la fila t solo usa datos hasta el cierre de t. El
   desfase a t+1 lo hace el motor; las estrategias NUNCA hacen `shift(-k)`.
2. **Cada estrategia nueva lleva un test de "cambiar el futuro no altera el pasado"**
   (ver `tests/test_momentum.py::test_no_lookahead_in_signal`).
3. **Test antes que código** para lógica nueva del motor o de métricas, con casos calculables
   a mano.
4. **Sin datos de red en tests**: usar `synthetic_prices`.
5. Los notebooks solo importan y presentan; la lógica vive en `src/`.
6. No reescribir código correcto ni acortarlo perdiendo claridad; mantener la estructura existente.
7. Al terminar una tarea: `pytest`, `ruff check .` y `ruff format --check .` en verde antes de
   dar algo por hecho.

## Estilo
- Python 3.11+, type hints, docstrings breves en español
- pandas vectorizado; nada de bucles por fecha salvo que sea imprescindible
- Commits pequeños en español, en imperativo: "Añade estrategia de pairs trading"
