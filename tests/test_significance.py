"""Tests for the Sharpe-difference significance tests, on synthetic data with a known answer."""

import dataclasses
import math
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import sharpe
from backtester.significance import (
    EULER_GAMMA,
    _politis_white,
    _sharpe_gradient,
    bootstrap_sharpe,
    bootstrap_sharpe_difference,
    deflated_sharpe_ratio,
    effective_trials,
    expected_max_sharpe,
    holm_adjust,
    ledoit_wolf_sharpe_test,
    optimal_block_length,
    probabilistic_sharpe_ratio,
    return_moments,
    stationary_bootstrap_indices,
)

N_DAYS = 2500


def paired_returns(
    n: int, drift_1: float, drift_2: float, seed: int, idio_vol: float = 0.004
) -> tuple[pd.Series, pd.Series]:
    """Two daily return series sharing a common factor, with chosen daily drifts."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-01", periods=n)
    common = rng.normal(0.0, 0.01, n)
    r1 = common + rng.normal(drift_1, idio_vol, n)
    r2 = common + rng.normal(drift_2, idio_vol, n)
    return pd.Series(r1, index=index), pd.Series(r2, index=index)


def clustered_returns(n: int, seed: int, regime_days: int = 100) -> tuple[pd.Series, pd.Series]:
    """Two equal-drift series whose volatility switches between calm and stormy regimes."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-01", periods=n)
    regimes = rng.choice([0.006, 0.02], size=n // regime_days + 1)
    vol = np.repeat(regimes, regime_days)[:n]
    common = rng.normal(0.0, 1.0, n) * vol
    r1 = common + rng.normal(0.0004, 0.004, n) * vol / 0.01
    r2 = common + rng.normal(0.0004, 0.004, n) * vol / 0.01
    return pd.Series(r1, index=index), pd.Series(r2, index=index)


# --- stationary bootstrap indices -------------------------------------------------------


def test_indices_have_right_shape_and_range():
    idx = stationary_bootstrap_indices(50, 5.0, 20, np.random.default_rng(0))

    assert idx.shape == (20, 50)
    assert idx.min() >= 0
    assert idx.max() < 50


def test_very_long_blocks_are_contiguous_runs_with_wraparound():
    n = 40
    idx = stationary_bootstrap_indices(n, 1e12, 10, np.random.default_rng(1))

    steps = np.diff(idx, axis=1) % n
    assert (steps == 1).all()


def test_mean_block_one_is_iid_resampling():
    n = 50
    idx = stationary_bootstrap_indices(n, 1.0, 2000, np.random.default_rng(2))

    # With iid draws, a step "previous + 1" only happens by chance, with probability 1/n.
    continues = (np.diff(idx, axis=1) % n == 1).mean()
    assert continues == pytest.approx(1 / n, abs=0.005)


def test_mean_block_length_matches_the_requested_one():
    n, b = 1000, 10.0
    idx = stationary_bootstrap_indices(n, b, 200, np.random.default_rng(3))

    # A new block starts with probability 1/b; it only looks like a break if the random start
    # differs from "previous + 1", which has probability 1 - 1/n.
    breaks = (np.diff(idx, axis=1) % n != 1).mean()
    assert breaks == pytest.approx((1 / b) * (1 - 1 / n), abs=0.005)


def test_indices_reject_a_mean_block_below_one():
    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap_indices(50, 0.5, 10, np.random.default_rng(0))


# --- block length -----------------------------------------------------------------------


def test_block_length_is_larger_for_dependent_data_than_for_iid():
    rng = np.random.default_rng(4)
    iid = rng.normal(0.0, 0.01, 5000)
    ar = np.zeros(5000)
    shocks = rng.normal(0.0, 0.01, 5000)
    for t in range(1, 5000):  # AR(1) generator for the test only
        ar[t] = 0.7 * ar[t - 1] + shocks[t]

    b_iid = optimal_block_length(pd.Series(iid), pd.Series(rng.normal(0.0, 0.01, 5000)))
    b_ar = optimal_block_length(pd.Series(ar), pd.Series(ar * 0.5))

    assert b_iid >= 2.0
    assert b_ar > 2 * b_iid


def test_politis_white_matches_the_closed_form_for_an_ar1_process():
    # For AR(1) with coefficient rho: b = (G / g0)^(2/3) * n^(1/3), with
    # G / g0 = 2 rho / ((1 - rho) (1 + rho)). For rho = 0.7 and n = 5000 that is about 33.
    rho, n = 0.7, 5000
    shocks = np.random.default_rng(14).normal(0.0, 1.0, n)
    ar = np.zeros(n)
    for t in range(1, n):  # AR(1) generator for the test only
        ar[t] = rho * ar[t - 1] + shocks[t]

    theory = (2 * rho / ((1 - rho) * (1 + rho))) ** (2 / 3) * n ** (1 / 3)

    assert _politis_white(ar) == pytest.approx(theory, rel=0.4)


# --- the bootstrap test -----------------------------------------------------------------


def test_point_estimate_is_the_hand_computed_sharpe_difference():
    # Both series alternate between two values 0.02 apart, so std = 0.01 * sqrt(40 / 39) with
    # 40 observations. The means are 0.02 and 0.01, so the daily Sharpe gap is 0.01 / std.
    r1 = pd.Series([0.01, 0.03] * 20)
    r2 = pd.Series([0.00, 0.02] * 20)

    result = bootstrap_sharpe_difference(r1, r2, mean_block=2.0, n_boot=50, seed=0)

    expected = 0.01 / (0.01 * np.sqrt(40 / 39)) * np.sqrt(252)
    assert result.estimate == pytest.approx(expected)
    assert result.estimate == pytest.approx(sharpe(r1) - sharpe(r2))


def test_identical_series_give_zero_difference_and_are_never_significant():
    r, _ = paired_returns(N_DAYS, 0.0005, 0.0005, seed=5)

    boot = bootstrap_sharpe_difference(r, r.copy(), n_boot=500, seed=0)
    hac = ledoit_wolf_sharpe_test(r, r.copy())

    assert boot.estimate == 0.0
    assert (boot.ci_low, boot.ci_high) == (0.0, 0.0)
    assert boot.p_value == 1.0
    assert hac.estimate == 0.0
    assert hac.p_value == 1.0
    assert hac.z_score == 0.0


def test_resampling_is_paired_so_a_rescaled_copy_has_no_difference_in_any_replicate():
    # Sharpe is scale-invariant. If both series were resampled with the same blocks the
    # difference is 0 in every replicate; independent resampling would make it noisy.
    r, _ = paired_returns(N_DAYS, 0.0005, 0.0005, seed=6)

    result = bootstrap_sharpe_difference(r, 0.5 * r, n_boot=300, seed=0)

    assert result.ci_low == pytest.approx(0.0, abs=1e-9)
    assert result.ci_high == pytest.approx(0.0, abs=1e-9)


def test_clearly_higher_drift_is_significant_in_both_tests():
    r1, r2 = paired_returns(N_DAYS, 0.0010, 0.0002, seed=7)

    boot = bootstrap_sharpe_difference(r1, r2, n_boot=2000, seed=0)
    hac = ledoit_wolf_sharpe_test(r1, r2)

    assert boot.estimate > 0
    assert boot.p_value < 0.01
    assert boot.ci_low > 0
    assert hac.p_value < 0.01
    assert hac.ci_low > 0


def test_equal_drift_series_are_rejected_about_as_often_as_the_nominal_level():
    n_sims = 100
    boot_rejections = 0
    hac_rejections = 0
    for seed in range(n_sims):
        r1, r2 = paired_returns(750, 0.0004, 0.0004, seed=1000 + seed)
        boot_rejections += bootstrap_sharpe_difference(r1, r2, n_boot=100, seed=seed).p_value < 0.05
        hac_rejections += ledoit_wolf_sharpe_test(r1, r2).p_value < 0.05

    # Nominal level is 5%; with 100 simulations 12% is three standard errors away. The lower
    # bound rules out a test that never rejects.
    assert 0.01 <= boot_rejections / n_sims <= 0.12
    assert 0.01 <= hac_rejections / n_sims <= 0.12


def test_bootstrap_and_ledoit_wolf_agree_on_estimate_and_interval_width():
    r1, r2 = paired_returns(N_DAYS, 0.0006, 0.0002, seed=8)

    boot = bootstrap_sharpe_difference(r1, r2, n_boot=4000, seed=0)
    hac = ledoit_wolf_sharpe_test(r1, r2)

    assert boot.estimate == pytest.approx(hac.estimate)
    boot_width = boot.ci_high - boot.ci_low
    hac_width = hac.ci_high - hac.ci_low
    assert boot_width == pytest.approx(hac_width, rel=0.3)


def test_a_rescaled_copy_is_never_significant_in_the_hac_test():
    # Sharpe is scale-invariant: the true difference is exactly 0, whatever the scale factor.
    # Floating-point noise must not be turned into a huge z-score.
    r, _ = paired_returns(N_DAYS, 0.0005, 0.0005, seed=15)

    for scale in (0.3, 0.5, 0.7, 2.0):
        hac = ledoit_wolf_sharpe_test(r, scale * r)
        boot = bootstrap_sharpe_difference(r, scale * r, n_boot=100, seed=0)

        assert hac.estimate == 0.0
        assert hac.p_value == 1.0
        assert boot.p_value == 1.0


def test_sharpe_gradient_matches_finite_differences_on_a_high_sharpe_sample():
    # High daily Sharpe (about 1), so the second-moment terms of the gradient are not negligible.
    rng = np.random.default_rng(16)
    x1 = rng.normal(0.01, 0.01, 1000)
    x2 = rng.normal(0.005, 0.012, 1000)

    def f(m1: float, m2: float, g1: float, g2: float) -> float:
        return m1 / np.sqrt(g1 - m1**2) - m2 / np.sqrt(g2 - m2**2)

    point = np.array([x1.mean(), x2.mean(), (x1**2).mean(), (x2**2).mean()])
    numeric = np.empty(4)
    for i in range(4):
        step = np.zeros(4)
        step[i] = 1e-7
        numeric[i] = (f(*(point + step)) - f(*(point - step))) / 2e-7

    np.testing.assert_allclose(_sharpe_gradient(x1, x2), numeric, rtol=1e-5)


def test_risk_free_rate_is_applied_to_both_tests():
    r1, r2 = paired_returns(N_DAYS, 0.0006, 0.0002, seed=17)
    rf = 0.03

    boot = bootstrap_sharpe_difference(r1, r2, n_boot=200, seed=0, risk_free=rf)
    hac = ledoit_wolf_sharpe_test(r1, r2, risk_free=rf)
    hac_shifted = ledoit_wolf_sharpe_test(r1 - rf / 252, r2 - rf / 252)

    assert boot.estimate == pytest.approx(sharpe(r1, rf) - sharpe(r2, rf))
    assert boot.estimate != pytest.approx(sharpe(r1) - sharpe(r2))
    assert hac.estimate == pytest.approx(hac_shifted.estimate)
    assert hac.std_error == pytest.approx(hac_shifted.std_error)


def test_same_seed_reproduces_the_result():
    r1, r2 = paired_returns(500, 0.0005, 0.0002, seed=9)

    a = bootstrap_sharpe_difference(r1, r2, n_boot=200, seed=42)
    b = bootstrap_sharpe_difference(r1, r2, n_boot=200, seed=42)

    assert a == b


def test_explicit_block_length_is_reported_back():
    r1, r2 = paired_returns(500, 0.0005, 0.0002, seed=10)

    result = bootstrap_sharpe_difference(r1, r2, mean_block=7.0, n_boot=100, seed=0)

    assert result.block_length == 7.0
    assert result.n_boot == 100


def test_series_are_aligned_on_dates_without_a_warning_when_one_is_just_shorter():
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=11)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = bootstrap_sharpe_difference(r1, r2.iloc[50:], n_boot=100, seed=0)

    expected = sharpe(r1.iloc[50:]) - sharpe(r2.iloc[50:])
    assert result.estimate == pytest.approx(expected)


def test_a_gap_inside_the_sample_triggers_a_warning():
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=11)
    gapped = r2.copy()
    gapped.iloc[100:103] = np.nan

    with pytest.warns(UserWarning, match="gap"):
        bootstrap_sharpe_difference(r1, gapped, n_boot=100, seed=0)


def test_leading_missing_values_are_dropped_without_a_warning():
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=11)
    warm_up = r2.copy()
    warm_up.iloc[:30] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = bootstrap_sharpe_difference(r1, warm_up, n_boot=100, seed=0)

    assert result.estimate == pytest.approx(sharpe(r1.iloc[30:]) - sharpe(r2.iloc[30:]))


def test_a_nearly_constant_series_is_rejected_like_an_exactly_constant_one():
    # The mean of 30 copies of 0.01 is not exactly 0.01 in floating point, so the std is ~1e-18.
    flat = pd.Series([0.01] * 30)
    noisy = pd.Series(np.random.default_rng(18).normal(0.0, 0.01, 30))

    with pytest.raises(ValueError, match="constant"):
        bootstrap_sharpe_difference(flat, noisy, n_boot=50)
    with pytest.raises(ValueError, match="constant"):
        ledoit_wolf_sharpe_test(noisy, flat)


def test_an_unsorted_index_is_rejected_because_time_order_matters():
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=19)

    with pytest.raises(ValueError, match="sorted"):
        bootstrap_sharpe_difference(r1.iloc[::-1], r2.iloc[::-1], n_boot=50)


def test_bootstrap_indices_validate_their_arguments():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap_indices(50, float("nan"), 10, rng)
    with pytest.raises(ValueError, match="n must"):
        stationary_bootstrap_indices(0, 5.0, 10, rng)
    with pytest.raises(ValueError, match="n_boot"):
        stationary_bootstrap_indices(50, 5.0, 0, rng)
    with pytest.raises(ValueError, match="mean_block"):
        bootstrap_sharpe_difference(
            *paired_returns(300, 0.0005, 0.0002, seed=20), mean_block=float("nan")
        )


def test_all_replicates_non_finite_raises_a_clear_error(monkeypatch):
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=21)
    monkeypatch.setattr(
        "backtester.significance._bootstrap_differences",
        lambda x1, x2, mean_block, n_boot, seed: np.full(n_boot, np.nan),
    )

    with pytest.warns(UserWarning, match="dropped"), pytest.raises(ValueError, match="replicates"):
        bootstrap_sharpe_difference(r1, r2, n_boot=50)


def test_block_length_grows_with_volatility_clustering():
    iid, other = paired_returns(5000, 0.0004, 0.0004, seed=22)
    clustered, other_clustered = clustered_returns(5000, seed=22)

    assert optimal_block_length(clustered, other_clustered) > 5 * optimal_block_length(iid, other)


def test_equal_drift_series_with_volatility_clustering_are_not_badly_oversized():
    n_sims = 60
    boot_rejections = 0
    hac_rejections = 0
    for seed in range(n_sims):
        r1, r2 = clustered_returns(1000, seed=2000 + seed)
        boot_rejections += bootstrap_sharpe_difference(r1, r2, n_boot=100, seed=seed).p_value < 0.05
        hac_rejections += ledoit_wolf_sharpe_test(r1, r2).p_value < 0.05

    # Both tests are mildly oversized under volatility clustering (about 8-10% at T = 750 in
    # earlier checks), so the bound is looser than for iid data but still catches a broken
    # block length or lag choice.
    assert 0.01 <= boot_rejections / n_sims <= 0.15
    assert 0.01 <= hac_rejections / n_sims <= 0.15


def test_invalid_inputs_raise_clear_errors():
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=12)

    with pytest.raises(ValueError, match="observations"):
        bootstrap_sharpe_difference(r1.iloc[:5], r2.iloc[:5])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_sharpe_difference(r1, r2, confidence=1.5)
    with pytest.raises(ValueError, match="mean_block"):
        bootstrap_sharpe_difference(r1, r2, mean_block=0.5)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_sharpe_difference(r1, r2, n_boot=0)
    with pytest.raises(ValueError, match="constant"):
        ledoit_wolf_sharpe_test(r1, r1 * 0.0)
    duplicated = r1.copy()
    duplicated.index = [duplicated.index[0]] * len(duplicated)
    with pytest.raises(ValueError, match="duplicate"):
        bootstrap_sharpe_difference(duplicated, r2)


def test_results_are_immutable():
    r1, r2 = paired_returns(300, 0.0005, 0.0002, seed=13)
    result = bootstrap_sharpe_difference(r1, r2, n_boot=100, seed=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.0  # type: ignore[misc]


def test_holm_adjust_by_hand():
    """Three tests, p = 0.01, 0.04, 0.03. Sorted: 0.01, 0.03, 0.04, multiplied by 3, 2, 1:
    0.03, 0.06, 0.04. The adjusted values cannot decrease, so the last one is raised to 0.06.
    """
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})

    adjusted = holm_adjust(p)

    assert list(adjusted.index) == ["a", "b", "c"]  # same order as the input
    assert adjusted.to_dict() == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_holm_adjust_caps_at_one_and_leaves_a_single_test_alone():
    capped = holm_adjust(pd.Series([0.6, 0.9]))  # 2 * 0.6 = 1.2 -> 1.0, then max(1.0, 0.9)
    assert capped.tolist() == pytest.approx([1.0, 1.0])
    assert holm_adjust(pd.Series([0.03])).tolist() == pytest.approx([0.03])


def test_holm_adjust_is_between_raw_and_bonferroni():
    p = pd.Series(np.random.default_rng(0).uniform(0, 1, 20))

    adjusted = holm_adjust(p)

    assert (adjusted >= p - 1e-12).all()
    assert (adjusted <= np.minimum(1.0, p * len(p)) + 1e-12).all()


@pytest.mark.parametrize("values", [[], [0.2, np.nan], [0.2, 1.5], [-0.1, 0.2]])
def test_holm_adjust_rejects_invalid_p_values(values):
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust(pd.Series(values, dtype=float))


# --- one-sample bootstrap of the Sharpe ratio -------------------------------------------


def single_returns(n: int, drift: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, 0.01, n), index=pd.bdate_range("2010-01-01", periods=n))


def test_one_sample_estimate_is_the_hand_computed_sharpe():
    # Alternates between 0.01 and 0.03: mean 0.02, std = 0.01 * sqrt(40 / 39) with 40 values.
    r = pd.Series([0.01, 0.03] * 20)

    result = bootstrap_sharpe(r, mean_block=2.0, n_boot=50, seed=0)

    assert result.estimate == pytest.approx(0.02 / (0.01 * np.sqrt(40 / 39)) * np.sqrt(252))
    assert result.estimate == pytest.approx(sharpe(r))


def test_one_sample_clearly_positive_drift_is_significant():
    result = bootstrap_sharpe(single_returns(N_DAYS, 0.0015, seed=11), n_boot=2000, seed=0)

    assert result.p_value < 0.01
    assert result.ci_low > 0
    assert result.ci_low < result.estimate < result.ci_high


def test_one_sample_zero_drift_is_rejected_about_as_often_as_the_nominal_level():
    n_sims = 100
    rejections = sum(
        bootstrap_sharpe(single_returns(750, 0.0, seed=2000 + seed), n_boot=100, seed=seed).p_value
        < 0.05
        for seed in range(n_sims)
    )
    # Same bounds as the paired test: 12% is three standard errors above the nominal 5%.
    assert 0.01 <= rejections / n_sims <= 0.12


def test_one_sample_negated_returns_mirror_the_result():
    """With the same seed the same days are resampled, so -r gives the mirror image."""
    r = single_returns(500, 0.0005, seed=12)

    up = bootstrap_sharpe(r, mean_block=5.0, n_boot=300, seed=3)
    down = bootstrap_sharpe(-r, mean_block=5.0, n_boot=300, seed=3)

    assert down.estimate == pytest.approx(-up.estimate)
    assert (down.ci_low, down.ci_high) == pytest.approx((-up.ci_high, -up.ci_low))
    assert down.p_value == up.p_value


def test_one_sample_risk_free_rate_lowers_the_sharpe():
    r = single_returns(500, 0.0005, seed=13)
    result = bootstrap_sharpe(r, n_boot=50, seed=0, risk_free=0.03)
    assert result.estimate == pytest.approx(sharpe(r, 0.03))


def test_one_sample_block_length_covers_volatility_clustering():
    calm, _ = paired_returns(N_DAYS, 0.0004, 0.0004, seed=14)
    stormy, _ = clustered_returns(N_DAYS, seed=14)
    assert (
        bootstrap_sharpe(stormy, n_boot=20).block_length
        > bootstrap_sharpe(calm, n_boot=20).block_length
    )


def test_one_sample_ignores_a_leading_warm_up_and_rejects_bad_input():
    r = single_returns(300, 0.0005, seed=15)
    padded = pd.concat([pd.Series([np.nan] * 5, index=r.index[:5] - pd.Timedelta(days=30)), r])

    assert bootstrap_sharpe(padded, n_boot=50, seed=1) == bootstrap_sharpe(r, n_boot=50, seed=1)
    with pytest.raises(ValueError):
        bootstrap_sharpe(pd.Series([0.01] * 50), n_boot=50)
    with pytest.raises(ValueError):
        bootstrap_sharpe(r, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_sharpe(r, mean_block=0.5)
    with pytest.raises(ValueError):
        bootstrap_sharpe(r, confidence=1.0)


# --- Deflated Sharpe Ratio (Bailey & López de Prado, 2014) ------------------------------


def test_psr_matches_a_hand_computed_case():
    """Normal returns, per-period Sharpe 0.1, T = 101: z = 0.1 * 10 / sqrt(1 + 0.1^2 / 2)."""
    psr = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=101, periods_per_year=1)
    assert psr == pytest.approx(NormalDist().cdf(1.0 / math.sqrt(1.005)))
    assert psr == pytest.approx(0.8407, abs=1e-4)


def test_psr_is_one_half_when_the_sharpe_equals_the_benchmark():
    assert probabilistic_sharpe_ratio(0.7, 0.7, n_obs=500, skew=-1.0, kurtosis=6.0) == 0.5


def test_psr_penalizes_negative_skew_and_fat_tails():
    normal = probabilistic_sharpe_ratio(1.0, 0.0, n_obs=1000)
    assert probabilistic_sharpe_ratio(1.0, 0.0, n_obs=1000, skew=-1.0) < normal
    assert probabilistic_sharpe_ratio(1.0, 0.0, n_obs=1000, kurtosis=9.0) < normal


def test_expected_max_sharpe_matches_a_hand_computed_case():
    """N = 2, unit variance: (1 - g) * z(1/2) + g * z(1 - 1/(2e)), and z(1/2) = 0."""
    expected = EULER_GAMMA * NormalDist().inv_cdf(1.0 - 1.0 / (2.0 * math.e))
    assert expected_max_sharpe(2, 1.0) == pytest.approx(expected)
    assert expected_max_sharpe(2, 4.0) == pytest.approx(2.0 * expected)  # scales with the std


def test_expected_max_sharpe_is_zero_for_a_single_trial_or_identical_trials():
    assert expected_max_sharpe(1, 0.3) == 0.0
    assert expected_max_sharpe(50, 0.0) == 0.0


def test_expected_max_sharpe_is_floored_at_zero_for_barely_more_than_one_trial():
    """Below about 1.5 trials the approximation is negative; the DSR is then the plain PSR."""
    assert expected_max_sharpe(1.2, 1.0) == 0.0
    assert expected_max_sharpe(2.0, 1.0) > 0.0
    barely = deflated_sharpe_ratio(
        0.8, n_trials=1.2, sharpe_variance=0.1, skew=0.0, kurtosis=3.0, n_obs=2000
    )
    assert barely.dsr == probabilistic_sharpe_ratio(0.8, 0.0, 2000)


def test_expected_max_sharpe_is_close_to_a_simulated_maximum():
    """The mean of the largest of N standard normal draws; the formula is an approximation."""
    rng = np.random.default_rng(0)
    simulated = rng.standard_normal((20_000, 100)).max(axis=1).mean()
    assert expected_max_sharpe(100, 1.0) == pytest.approx(simulated, abs=0.03)


def test_dsr_reproduces_the_numerical_example_of_the_paper():
    """The numerical example of Bailey & López de Prado (2014).

    Annualized Sharpe 2.5 on 1250 daily observations (250 a year), best of 100 trials whose
    annualized Sharpe ratios have variance 1/2, skewness -3 and kurtosis 10. The paper reports
    a non-annualized expected maximum of 0.1132 and DSR = 0.9004.
    """
    result = deflated_sharpe_ratio(
        2.5,
        n_trials=100,
        sharpe_variance=0.5,
        skew=-3.0,
        kurtosis=10.0,
        n_obs=1250,
        periods_per_year=250,
    )
    assert result.expected_max_sharpe / math.sqrt(250) == pytest.approx(0.1132, abs=5e-5)
    assert result.dsr == pytest.approx(0.9004, abs=5e-5)


def test_dsr_with_a_single_trial_is_the_psr_against_zero():
    result = deflated_sharpe_ratio(
        0.8, n_trials=1, sharpe_variance=0.0, skew=-0.5, kurtosis=5.0, n_obs=2000
    )
    assert result.expected_max_sharpe == 0.0
    assert result.dsr == probabilistic_sharpe_ratio(0.8, 0.0, 2000, skew=-0.5, kurtosis=5.0)


def test_dsr_is_one_half_when_the_sharpe_is_the_expected_maximum():
    threshold = expected_max_sharpe(40, 0.09)
    result = deflated_sharpe_ratio(
        threshold, n_trials=40, sharpe_variance=0.09, skew=0.0, kurtosis=3.0, n_obs=3000
    )
    assert result.dsr == 0.5


def test_dsr_falls_with_more_trials_and_with_more_dispersion_across_trials():
    def dsr(n_trials: float, variance: float) -> float:
        return deflated_sharpe_ratio(
            1.0, n_trials=n_trials, sharpe_variance=variance, skew=0.0, kurtosis=3.0, n_obs=2500
        ).dsr

    assert dsr(100, 0.04) < dsr(10, 0.04) < dsr(1, 0.04)
    assert dsr(10, 0.16) < dsr(10, 0.04)


def test_dsr_gives_the_same_answer_for_annualized_and_per_period_inputs():
    """Annualizing multiplies the Sharpe by sqrt(252) and its variance across trials by 252."""
    common = {"n_trials": 30, "skew": -0.4, "kurtosis": 7.0, "n_obs": 4000}
    annualized = deflated_sharpe_ratio(0.9, sharpe_variance=0.05, periods_per_year=252, **common)
    per_period = deflated_sharpe_ratio(
        0.9 / math.sqrt(252), sharpe_variance=0.05 / 252, periods_per_year=1, **common
    )
    assert annualized.dsr == pytest.approx(per_period.dsr)
    assert annualized.expected_max_sharpe == pytest.approx(
        per_period.expected_max_sharpe * math.sqrt(252)
    )
    # Mixing the two (an annualized Sharpe read as a per-period one) is a different number.
    mixed = deflated_sharpe_ratio(0.9, sharpe_variance=0.05, periods_per_year=1, **common)
    assert abs(mixed.dsr - annualized.dsr) > 0.01


def test_dsr_rejects_invalid_inputs():
    valid = {
        "sharpe": 1.0,
        "n_trials": 10,
        "sharpe_variance": 0.1,
        "skew": 0.0,
        "kurtosis": 3.0,
        "n_obs": 500,
    }
    for bad in (
        {"n_trials": 0},
        {"sharpe_variance": -0.1},
        {"n_obs": 1},
        {"periods_per_year": 0},
        {"kurtosis": 0.5},  # below 1, impossible for any distribution
        {"sharpe": float("nan")},
    ):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(**{**valid, **bad})
    # Skewness 3 needs a kurtosis of at least 10; with 3 the estimator variance turns negative.
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(30.0, 0.0, n_obs=500, skew=3.0, kurtosis=3.0)


def test_return_moments_use_plain_kurtosis_not_excess():
    """Two-point +-1 returns have skewness 0 and kurtosis 1; pandas `.kurt()` reports excess."""
    skew, kurtosis = return_moments(pd.Series([1.0, -1.0] * 50))
    assert (skew, kurtosis) == pytest.approx((0.0, 1.0))

    values = pd.Series([0.0, 0.0, 0.0, 4.0])  # mean 1, m2 = 3, m3 = 6, m4 = 21
    skew, kurtosis = return_moments(values)
    assert skew == pytest.approx(6.0 / 3.0**1.5)
    assert kurtosis == pytest.approx(21.0 / 9.0)

    normal = pd.Series(np.random.default_rng(0).standard_normal(200_000))
    assert return_moments(normal) == pytest.approx((0.0, 3.0), abs=0.05)
    with pytest.raises(ValueError):
        return_moments(pd.Series([0.01] * 30))


def test_effective_trials_goes_from_one_to_n_with_the_correlation():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(5000)
    copies = pd.DataFrame({k: base * (k + 1) for k in range(6)})  # correlation 1
    independent = pd.DataFrame(rng.standard_normal((5000, 6)))

    assert effective_trials(copies) == pytest.approx(1.0)
    assert effective_trials(independent) == pytest.approx(6.0, abs=0.2)
    # Two columns with correlation 0.5: rho + (1 - rho) * N = 0.5 + 0.5 * 2.
    z = rng.standard_normal(200_000)
    pair = pd.DataFrame({"a": z, "b": 0.5 * z + math.sqrt(0.75) * rng.standard_normal(200_000)})
    assert effective_trials(pair) == pytest.approx(1.5, abs=0.01)
    assert effective_trials(copies[[0]]) == 1.0
