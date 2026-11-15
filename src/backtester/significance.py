"""Significance tests for the difference between the Sharpe ratios of two return series.

Two independent routes to the same question, "is Sharpe(r1) - Sharpe(r2) different from zero?":

- `bootstrap_sharpe_difference`: stationary bootstrap (Politis & Romano, 1994) on the PAIRED
  daily returns. Both series are resampled with the same blocks, so their correlation is kept.
- `ledoit_wolf_sharpe_test`: the HAC-based asymptotic test of Ledoit & Wolf (2008).

Both use the same statistic as `metrics.sharpe`: annualized (252 days), on simple daily returns.

`bootstrap_sharpe` is the one-sample version of the bootstrap: a single Sharpe ratio against
zero, for strategies whose benchmark is cash.

`holm_adjust` corrects the p-values when several such comparisons are made at once.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from backtester.metrics import PERIODS_PER_YEAR, sharpe

MIN_OBSERVATIONS = 20
# Bootstrap replicates are drawn in chunks of this many cells. Several arrays of this size are
# alive at once, so peak memory is a few hundred MB, not just 4M * 8 bytes.
_CHUNK_ELEMENTS = 4_000_000
_MIN_BLOCK = 2.0


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a stationary bootstrap test (paired or one-sample). Sharpe values are annualized.

    `n_boot` is the number of replicates actually used: those with a finite Sharpe in every
    series. It can be smaller than the number requested.
    """

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    block_length: float
    n_boot: int
    confidence: float


@dataclass(frozen=True)
class HACResult:
    """Outcome of the Ledoit-Wolf HAC test. Sharpe values are annualized."""

    estimate: float
    std_error: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float
    lags: int
    confidence: float


# --- input handling ---------------------------------------------------------------------


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")


def _align(r1: pd.Series, r2: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join the two series on their index and drop missing values."""
    first, second = pd.Series(r1), pd.Series(r2)
    for series in (first, second):
        if not series.index.is_unique:
            raise ValueError("return series must not have duplicate index labels")
        if not series.index.is_monotonic_increasing:
            raise ValueError("return series must be sorted by date, oldest first")
    paired = pd.concat([first, second], axis=1, join="inner")
    valid = paired.notna().all(axis=1).to_numpy()
    if valid.sum() < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} observations in common, got {int(valid.sum())}"
        )

    # Missing values before the first or after the last valid row are harmless (a warm-up
    # period). Those in between glue non-adjacent days together, which distorts the block
    # bootstrap and the dependence estimates, so they are reported.
    inside = valid[valid.argmax() : len(valid) - valid[::-1].argmax()]
    gaps = int((~inside).sum())
    if gaps > 0:
        warnings.warn(
            f"{gaps} rows with missing values inside the sample were dropped; this closes "
            "gaps and makes non-adjacent days adjacent",
            UserWarning,
            stacklevel=3,
        )
    joined = paired[valid]
    return joined.iloc[:, 0].astype(float), joined.iloc[:, 1].astype(float)


def _sharpe_gap(a: pd.Series, b: pd.Series, risk_free: float) -> float:
    """Sharpe(a) - Sharpe(b), snapped to exactly 0 when it is only floating-point noise.

    Sharpe is scale-invariant, so e.g. a series and its leveraged copy have a true gap of 0,
    but the computed one can be ~1e-16 and would otherwise be divided by a ~1e-17 std. error.
    """
    sharpe_a, sharpe_b = sharpe(a, risk_free), sharpe(b, risk_free)
    gap = sharpe_a - sharpe_b
    if abs(gap) <= 1e-9 * max(1.0, abs(sharpe_a), abs(sharpe_b)):
        return 0.0
    return float(gap)


def _excess(returns: pd.Series, risk_free: float) -> np.ndarray:
    values = returns.to_numpy(dtype=float) - risk_free / PERIODS_PER_YEAR
    # A relative tolerance: the mean of 30 copies of 0.01 is not exactly 0.01, so the std of a
    # constant series is ~1e-18 rather than exactly 0.
    if values.std(ddof=1) <= 1e-12 * np.abs(values).max():
        raise ValueError("a return series is constant, so its Sharpe ratio is undefined")
    return values


# --- block length (Politis & White, 2004; Patton, Politis & White, 2009) ----------------


def _autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    centered = x - x.mean()
    n = len(x)
    return np.array([centered[: n - k] @ centered[k:] / n for k in range(max_lag + 1)])


def _flat_top_weights(lags: np.ndarray, bandwidth: int) -> np.ndarray:
    """Trapezoid lag window: 1 up to half the bandwidth, then linearly down to 0."""
    t = lags / bandwidth
    return np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))


def _politis_white(x: np.ndarray) -> float:
    """Optimal mean block length of the stationary bootstrap for the mean of `x`."""
    n = len(x)
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = math.ceil(math.sqrt(n)) + k_n
    max_lag = min(m_max + k_n, n - 1)
    acov = _autocovariances(x, max_lag)
    if acov[0] <= 0:
        return 1.0

    threshold = 2.0 * math.sqrt(math.log10(n) / n)
    significant = np.abs(acov[1:] / acov[0]) > threshold
    m_hat = max_lag - k_n
    for m in range(max_lag - k_n + 1):
        if not significant[m : m + k_n].any():
            m_hat = m
            break

    bandwidth = min(2 * m_hat, m_max, max_lag)
    if bandwidth == 0:
        return 1.0
    lags = np.arange(1, bandwidth + 1)
    weights = _flat_top_weights(lags, bandwidth)
    long_run_var = acov[0] + 2.0 * (weights * acov[1 : bandwidth + 1]).sum()
    g_sum = 2.0 * (weights * lags * acov[1 : bandwidth + 1]).sum()
    if long_run_var <= 0:
        return 1.0
    d_sb = 2.0 * long_run_var**2
    block = (2.0 * g_sum**2 / d_sb) ** (1 / 3) * n ** (1 / 3)
    return float(max(1.0, min(block, math.ceil(min(3 * math.sqrt(n), n / 3)))))


def _paired_block_length(x1: np.ndarray, x2: np.ndarray) -> float:
    """Largest Politis-White block over both series and their squares, rounded up."""
    candidates = (x1, x2, x1**2, x2**2)
    return float(max(_MIN_BLOCK, math.ceil(max(_politis_white(c) for c in candidates))))


def optimal_block_length(r1: pd.Series, r2: pd.Series) -> float:
    """Mean block length for the paired stationary bootstrap of a Sharpe difference.

    The Sharpe ratio depends on the mean and on the second moment, and volatility clustering
    makes squared returns dependent even when returns are not. The Politis-White rule is
    therefore applied to both series and to both squared series, and the largest block wins.
    The result is rounded up and never below 2.
    """
    a, b = _align(r1, r2)
    return _paired_block_length(a.to_numpy(), b.to_numpy())


# --- stationary bootstrap ---------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Resampling indices, shape (n_boot, n), of the stationary bootstrap.

    At each position a new block starts with probability 1/mean_block, at a uniform random
    index. Otherwise the previous index is followed by the next one, wrapping around the end,
    so block lengths are geometric with mean `mean_block`.
    """
    if not mean_block >= 1.0:  # also rejects NaN
        raise ValueError(f"mean_block must be at least 1, got {mean_block}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    columns = np.arange(n)
    restart = rng.random((n_boot, n)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    block_start = np.maximum.accumulate(np.where(restart, columns, 0), axis=1)
    base = np.take_along_axis(starts, block_start, axis=1)
    return (base + columns - block_start) % n


def _annualized_sharpe_rows(samples: np.ndarray) -> np.ndarray:
    """Annualized Sharpe of each row (returns already in excess of the risk-free rate)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return samples.mean(axis=1) / samples.std(axis=1, ddof=1) * math.sqrt(PERIODS_PER_YEAR)


def _bootstrap_differences(
    x1: np.ndarray, x2: np.ndarray, mean_block: float, n_boot: int, seed: int
) -> np.ndarray:
    """Sharpe differences of every replicate; both series use the same resampled indices."""
    rng = np.random.default_rng(seed)
    n = len(x1)
    chunk = max(1, _CHUNK_ELEMENTS // n)
    pieces = []
    for done in range(0, n_boot, chunk):
        idx = stationary_bootstrap_indices(n, mean_block, min(chunk, n_boot - done), rng)
        pieces.append(_annualized_sharpe_rows(x1[idx]) - _annualized_sharpe_rows(x2[idx]))
    return np.concatenate(pieces)


def bootstrap_sharpe_difference(
    r1: pd.Series,
    r2: pd.Series,
    mean_block: float | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    risk_free: float = 0.0,
) -> BootstrapResult:
    """Paired stationary bootstrap test of Sharpe(r1) - Sharpe(r2).

    The confidence interval is the percentile interval of the bootstrap differences. The
    two-sided p-value imposes the null by recentering: with B replicates and k of them at least
    as far from the estimate as the estimate is from zero, p = (1 + k) / (B + 1).

    This is the plain (not studentized) bootstrap. Ledoit and Wolf (2008) recommend studentizing
    for better size under heteroskedasticity; with strong volatility clustering this test is
    mildly oversized (about 8-10% rejection at a 5% level in simulations with T = 750), so use
    `ledoit_wolf_sharpe_test` as a cross-check.
    """
    _check_confidence(confidence)
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if mean_block is not None and not mean_block >= 1.0:  # also rejects NaN
        raise ValueError(f"mean_block must be at least 1, got {mean_block}")

    a, b = _align(r1, r2)
    x1, x2 = _excess(a, risk_free), _excess(b, risk_free)
    block = _paired_block_length(x1, x2) if mean_block is None else float(mean_block)

    estimate = _sharpe_gap(a, b, risk_free)
    diffs = _bootstrap_differences(x1, x2, block, n_boot, seed)
    finite = np.isfinite(diffs)
    if not finite.all():
        warnings.warn(
            f"{int((~finite).sum())} bootstrap replicates had a constant series and were dropped",
            UserWarning,
            stacklevel=2,
        )
        diffs = diffs[finite]
    if diffs.size == 0:
        raise ValueError("all bootstrap replicates had a constant series; check the input data")

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    extreme = np.abs(diffs - estimate) >= abs(estimate)
    p_value = (1 + int(extreme.sum())) / (len(diffs) + 1)
    return BootstrapResult(
        estimate=float(estimate),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        block_length=block,
        n_boot=len(diffs),
        confidence=confidence,
    )


def bootstrap_sharpe(
    returns: pd.Series,
    mean_block: float | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    risk_free: float = 0.0,
) -> BootstrapResult:
    """Stationary bootstrap test of a single Sharpe ratio against zero.

    The one-sample version of `bootstrap_sharpe_difference`, for strategies whose natural
    benchmark is cash (e.g. a market-neutral one): the question is whether the Sharpe ratio
    is different from zero, not whether it beats another risky series. Same percentile interval
    and same recentered two-sided p-value. The block length is the largest Politis-White block
    of the returns and of their squares, rounded up and never below 2.
    """
    _check_confidence(confidence)
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if mean_block is not None and not mean_block >= 1.0:  # also rejects NaN
        raise ValueError(f"mean_block must be at least 1, got {mean_block}")

    clean, _ = _align(returns, returns)  # same checks and missing-value rules as the paired test
    x = _excess(clean, risk_free)
    block = _paired_block_length(x, x) if mean_block is None else float(mean_block)

    estimate = float(sharpe(clean, risk_free))
    rng = np.random.default_rng(seed)
    chunk = max(1, _CHUNK_ELEMENTS // len(x))
    pieces = []
    for done in range(0, n_boot, chunk):
        idx = stationary_bootstrap_indices(len(x), block, min(chunk, n_boot - done), rng)
        pieces.append(_annualized_sharpe_rows(x[idx]))
    sharpes = np.concatenate(pieces)
    finite = np.isfinite(sharpes)
    if not finite.all():
        warnings.warn(
            f"{int((~finite).sum())} bootstrap replicates had a constant series and were dropped",
            UserWarning,
            stacklevel=2,
        )
        sharpes = sharpes[finite]
    if sharpes.size == 0:
        raise ValueError("all bootstrap replicates had a constant series; check the input data")

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(sharpes, [alpha / 2.0, 1.0 - alpha / 2.0])
    extreme = np.abs(sharpes - estimate) >= abs(estimate)
    return BootstrapResult(
        estimate=estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float((1 + int(extreme.sum())) / (len(sharpes) + 1)),
        block_length=block,
        n_boot=len(sharpes),
        confidence=confidence,
    )


# --- Ledoit & Wolf (2008) HAC test ------------------------------------------------------


def _sharpe_gradient(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Gradient of mu1/sqrt(g1 - mu1^2) - mu2/sqrt(g2 - mu2^2) in (mu1, mu2, g1, g2)."""
    mu1, mu2 = x1.mean(), x2.mean()
    g1, g2 = (x1**2).mean(), (x2**2).mean()
    var1, var2 = g1 - mu1**2, g2 - mu2**2
    return np.array(
        [
            g1 / var1**1.5,
            -g2 / var2**1.5,
            -mu1 / (2.0 * var1**1.5),
            mu2 / (2.0 * var2**1.5),
        ]
    )


def _bartlett_long_run_variance(h: np.ndarray) -> tuple[float, int]:
    """Newey-West long-run variance of `h` with the Newey-West (1994) automatic lag rule."""
    t = len(h)
    n_pre = min(int(4.0 * (t / 100.0) ** (2.0 / 9.0)), t - 1)
    pre = _autocovariances(h, n_pre)
    s0 = pre[0] + 2.0 * pre[1:].sum()
    s1 = 2.0 * (np.arange(1, n_pre + 1) * pre[1:]).sum()
    lags = n_pre
    if s0 > 0:
        lags = int(1.1447 * abs(s1 / s0) ** (2.0 / 3.0) * t ** (1.0 / 3.0))
    lags = min(lags, t // 4)  # a bandwidth near T leaves a very noisy variance estimate

    acov = _autocovariances(h, lags)
    weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    return float(acov[0] + 2.0 * (weights * acov[1:]).sum()), lags


def ledoit_wolf_sharpe_test(
    r1: pd.Series, r2: pd.Series, risk_free: float = 0.0, confidence: float = 0.95
) -> HACResult:
    """HAC-based asymptotic test of Sharpe(r1) - Sharpe(r2), after Ledoit & Wolf (2008).

    The difference is a smooth function of the means and second moments of the two series.
    The long-run variance of the gradient-weighted moment deviations gives the standard error
    through the delta method, and the p-value is two-sided from the normal distribution.

    Differences from the paper's implementation: the long-run variance uses a plain Bartlett
    (Newey-West) kernel with the Newey-West (1994) automatic lag, and the standard error uses
    population (ddof=0) moments while the estimate uses `metrics.sharpe` (ddof=1). The gap is of
    order 1/T.
    """
    _check_confidence(confidence)
    a, b = _align(r1, r2)
    x1, x2 = _excess(a, risk_free), _excess(b, risk_free)
    estimate = _sharpe_gap(a, b, risk_free)
    if estimate == 0.0 and (np.array_equal(x1, x2) or np.allclose(x1 / x1.std(), x2 / x2.std())):
        # Identical or rescaled copies: the difference is exactly zero for every sample.
        return HACResult(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, confidence)

    moments = np.column_stack(
        [x1 - x1.mean(), x2 - x2.mean(), x1**2 - (x1**2).mean(), x2**2 - (x2**2).mean()]
    )
    h = moments @ _sharpe_gradient(x1, x2)
    long_run_var, lags = _bartlett_long_run_variance(h)
    std_error = math.sqrt(PERIODS_PER_YEAR * max(long_run_var, 0.0) / len(h))

    if std_error == 0.0:
        z_score = 0.0 if estimate == 0.0 else math.copysign(math.inf, estimate)
    else:
        z_score = estimate / std_error
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    z_crit = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    return HACResult(
        estimate=float(estimate),
        std_error=std_error,
        z_score=z_score,
        p_value=p_value,
        ci_low=float(estimate - z_crit * std_error),
        ci_high=float(estimate + z_crit * std_error),
        lags=lags,
        confidence=confidence,
    )


# --- multiple comparisons ---------------------------------------------------------------


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm (1979) step-down adjustment of the p-values of a family of tests.

    Testing m hypotheses at 5% each gives more than a 5% chance of at least one false positive.
    Holm controls that family-wise error rate: sort the p-values, multiply the smallest by m,
    the next by m - 1, ..., the largest by 1, then make the sequence non-decreasing and cap it
    at 1. A hypothesis is rejected at level alpha if its adjusted p-value is below alpha.

    It makes no assumption about the dependence between the tests and is never less powerful
    than Bonferroni. The result keeps the index and order of the input.
    """
    p = p_values.to_numpy(dtype=float)
    if p.size == 0 or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be a non-empty set of numbers between 0 and 1")

    order = np.argsort(p, kind="stable")
    multipliers = p.size - np.arange(p.size)
    stepped = np.minimum(1.0, np.maximum.accumulate(p[order] * multipliers))
    adjusted = np.empty_like(p)
    adjusted[order] = stepped
    return pd.Series(adjusted, index=p_values.index, name=p_values.name)
