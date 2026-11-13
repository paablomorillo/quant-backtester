# quant-backtester

Motor de backtesting vectorizado en Python para estrategias cuantitativas, centrado en el rigor
metodológico: sin look-ahead bias, con costes de transacción y con validación fuera de muestra.

> 🚧 En desarrollo. Ver [docs/ROADMAP.md](docs/ROADMAP.md).

## Qué hace
- Convierte una estrategia (precios → pesos de cartera) en una serie de rendimientos netos
- Aplica las decisiones del cierre de *t* a los rendimientos de *t+1*, así que no hay look-ahead por construcción
- Descuenta costes proporcionales al turnover
- Calcula CAGR, volatilidad, Sharpe, Sortino, max drawdown, Calmar y turnover
- Rebalanceo diario, semanal o mensual

## Estrategias
| Estrategia | Estado |
|---|---|
| Momentum de sección cruzada 12-1 (long-only / long-short) | ✅ |
| Mean reversion | ⏳ |
| Pairs trading (cointegración) | ⏳ |
| Trend following por series temporales | ⏳ |

## Instalación
```bash
git clone https://github.com/<tu-usuario>/quant-backtester.git
cd quant-backtester
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Uso rápido
```python
from backtester import ProportionalCosts, run_backtest, summary
from backtester.data import load_prices
from backtester.strategies import CrossSectionalMomentum

prices = load_prices(
    ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"], start="2005-01-01"
)
result = run_backtest(
    prices, CrossSectionalMomentum(top_n=3), ProportionalCosts(bps=5), rebalance="ME"
)
print(summary(result))
```

O directamente: `python examples/run_momentum.py`

## Resultados
*(Fase 1: aquí irán la gráfica de equity y la tabla de métricas con datos reales.)*

## Tests
```bash
pytest
```
Además de los tests unitarios de métricas, hay tests específicos contra el look-ahead bias:
modificar precios futuros no puede cambiar los pesos pasados, y una estrategia que "compra lo
que ha subido hoy" no puede ganar en un paseo aleatorio.

## Estructura
```
src/backtester/
├── data.py          # descarga + caché + datos sintéticos
├── strategy.py      # interfaz Strategy
├── strategies/      # una estrategia por fichero
├── engine.py        # run_backtest
├── costs.py         # modelos de costes
├── metrics.py       # métricas de rendimiento y riesgo
└── report.py        # gráficas y comparativas
tests/               # pytest, sin red
examples/            # scripts ejecutables
notebooks/           # análisis y demos
docs/ROADMAP.md      # plan por fases
```

## Licencia
MIT
