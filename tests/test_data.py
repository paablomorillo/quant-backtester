import numpy as np
import pandas as pd
import pytest

from backtester.data import check_complete, quality_report, synthetic_prices


def test_quality_report_clean_data():
    prices = synthetic_prices(n_assets=3, n_days=100)

    report = quality_report(prices)

    assert list(report.index) == list(prices.columns)
    assert (report["Primera fecha"] == prices.index[0]).all()
    assert (report["Última fecha"] == prices.index[-1]).all()
    assert (report["Observaciones"] == 100).all()
    assert (report["NaN internos"] == 0).all()
    assert (report["Precios no positivos"] == 0).all()
    # bdate_range: el mayor hueco entre días hábiles es el fin de semana (3 días naturales)
    assert (report["Mayor hueco (días)"] == 3).all()


def test_quality_report_detects_late_start_and_internal_gaps():
    prices = synthetic_prices(n_assets=2, n_days=50)
    prices.iloc[:10, 0] = np.nan  # A0 empieza tarde: no es un hueco interno
    prices.iloc[20:23, 1] = np.nan  # A1 tiene 3 NaN en mitad de la serie

    report = quality_report(prices)

    assert report.loc["A0", "Primera fecha"] == prices.index[10]
    assert report.loc["A0", "Observaciones"] == 40
    assert report.loc["A0", "NaN internos"] == 0
    assert report.loc["A1", "Primera fecha"] == prices.index[0]
    assert report.loc["A1", "NaN internos"] == 3


def test_quality_report_max_abs_return_and_bad_prices():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"X": [100.0, 150.0, 75.0, 0.0]}, index=index)

    report = quality_report(prices)

    # rendimientos: +50 %, -50 %, -100 %
    assert report.loc["X", "Mayor |rend. diario|"] == pytest.approx(1.0)
    assert report.loc["X", "Precios no positivos"] == 1


def test_quality_report_calendar_gap():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-12", "2024-01-15"])
    prices = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]}, index=index)

    assert quality_report(prices).loc["X", "Mayor hueco (días)"] == 10


def test_check_complete_accepts_full_download():
    prices = synthetic_prices(n_assets=2, n_days=10)

    check_complete(prices, ["A0", "A1"])  # no lanza


def test_check_complete_rejects_empty_or_missing_ticker():
    prices = synthetic_prices(n_assets=2, n_days=10)
    prices["A1"] = np.nan  # descarga fallida: columna entera a NaN

    with pytest.raises(ValueError, match="A1"):
        check_complete(prices, ["A0", "A1"])
    with pytest.raises(ValueError, match="A2"):
        check_complete(prices[["A0"]], ["A0", "A2"])
