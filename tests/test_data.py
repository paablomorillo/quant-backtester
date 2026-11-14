import numpy as np
import pandas as pd
import pytest

from backtester.data import check_complete, quality_report, synthetic_prices


def test_quality_report_clean_data():
    prices = synthetic_prices(n_assets=3, n_days=100)

    report = quality_report(prices)

    assert list(report.index) == list(prices.columns)
    assert (report["First date"] == prices.index[0]).all()
    assert (report["Last date"] == prices.index[-1]).all()
    assert (report["Observations"] == 100).all()
    assert (report["Internal NaNs"] == 0).all()
    assert (report["Non-positive prices"] == 0).all()
    # bdate_range: the largest gap between business days is the weekend (3 calendar days)
    assert (report["Largest gap (days)"] == 3).all()


def test_quality_report_detects_late_start_and_internal_gaps():
    prices = synthetic_prices(n_assets=2, n_days=50)
    prices.iloc[:10, 0] = np.nan  # A0 starts late: not an internal gap
    prices.iloc[20:23, 1] = np.nan  # A1 has 3 NaNs in the middle of the series

    report = quality_report(prices)

    assert report.loc["A0", "First date"] == prices.index[10]
    assert report.loc["A0", "Observations"] == 40
    assert report.loc["A0", "Internal NaNs"] == 0
    assert report.loc["A1", "First date"] == prices.index[0]
    assert report.loc["A1", "Internal NaNs"] == 3


def test_quality_report_max_abs_return_and_bad_prices():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"X": [100.0, 150.0, 75.0, 0.0]}, index=index)

    report = quality_report(prices)

    # returns: +50%, -50%, -100%
    assert report.loc["X", "Largest |daily return|"] == pytest.approx(1.0)
    assert report.loc["X", "Non-positive prices"] == 1


def test_quality_report_calendar_gap():
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-12", "2024-01-15"])
    prices = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]}, index=index)

    assert quality_report(prices).loc["X", "Largest gap (days)"] == 10


def test_check_complete_accepts_full_download():
    prices = synthetic_prices(n_assets=2, n_days=10)

    check_complete(prices, ["A0", "A1"])  # does not raise


def test_check_complete_rejects_empty_or_missing_ticker():
    prices = synthetic_prices(n_assets=2, n_days=10)
    prices["A1"] = np.nan  # failed download: all-NaN column

    with pytest.raises(ValueError, match="A1"):
        check_complete(prices, ["A0", "A1"])
    with pytest.raises(ValueError, match="A2"):
        check_complete(prices[["A0"]], ["A0", "A2"])
