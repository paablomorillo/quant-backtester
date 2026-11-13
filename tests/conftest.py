"""Configuración común de los tests."""

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: los tests de gráficas no necesitan pantalla
