# Hoja de ruta

Cada fase termina con: tests en verde, CI en verde, README actualizado y un commit/tag.

## Fase 0 — Esqueleto (hecha)
- [x] Motor vectorizado con desfase t → t+1 (anti look-ahead)
- [x] Costes proporcionales en bps sobre turnover
- [x] Métricas: CAGR, volatilidad, Sharpe, Sortino, max drawdown, Calmar, turnover
- [x] Momentum de sección cruzada 12-1 (long-only y long-short)
- [x] Rebalanceo diario / semanal / mensual
- [x] Tests (incluido test de que cambiar el futuro no altera el pasado) y CI

## Fase 1 — Primer resultado real (1-2 semanas)
- [x] Ejecutar `examples/run_momentum.py` con datos reales de ETFs sectoriales
- [x] Notebook `01_momentum_sectores.ipynb` con resultados, gráficas e interpretación
- [x] README con la gráfica de equity y la tabla de métricas
- [x] Deriva de pesos entre rebalanceos

## Fase 2 — Más estrategias (2-3 semanas)
- [ ] Mean reversion de corto plazo (z-score sobre media móvil)
- [ ] Pairs trading: test de cointegración (Engle-Granger), spread y z-score de entrada/salida
- [ ] Trend following por series temporales (señal por activo, no de sección cruzada)
- [ ] Volatility targeting como capa encima de cualquier estrategia

## Fase 3 — Rigor estadístico (lo que diferencia el proyecto)
- [ ] Validación walk-forward: optimizar en ventana de entrenamiento, evaluar fuera de muestra
- [ ] Bootstrap de bloques para intervalos de confianza del Sharpe
- [ ] Deflated Sharpe Ratio (López de Prado) para corregir por número de pruebas
- [ ] Análisis de sensibilidad de parámetros (heatmap lookback × top_n)
- [ ] Costes más realistas: spread + impacto proporcional a la volatilidad

## Fase 4 — Presentación
- [ ] Informe HTML automático por estrategia
- [ ] Demo interactiva (Streamlit) para elegir activos, estrategia y parámetros
- [ ] Post/README final explicando los resultados, incluidos los negativos

## Ideas para más adelante
- Factores (value, quality, low-vol) con datos fundamentales
- Optimización de cartera (mínima varianza, risk parity) como capa de pesos
- Motor event-driven para órdenes limitadas e intradía
