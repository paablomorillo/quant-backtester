"""Pairs trading on cointegrated pairs (Engle & Granger, 1987; Gatev et al., 2006)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from backtester.strategy import Strategy

EXIT_TARGET = "target"
EXIT_STOP = "stop"
EXIT_TIME = "time"
EXIT_NOT_COINTEGRATED = "not_cointegrated"
STILL_OPEN = "open"


def hedge_ratio(log_y: np.ndarray, log_x: np.ndarray) -> tuple[float, float, float]:
    """OLS of log_y on log_x with an intercept: (alpha, beta, std of the residuals, ddof=1).

    A constant log_x has no slope; all three values are NaN in that case.
    """
    x_dev = log_x - log_x.mean()
    x_var = float(x_dev @ x_dev)
    if x_var == 0.0:
        return float("nan"), float("nan"), float("nan")
    beta = float(x_dev @ (log_y - log_y.mean())) / x_var
    alpha = float(log_y.mean() - beta * log_x.mean())
    residual = log_y - alpha - beta * log_x
    return alpha, beta, float(residual.std(ddof=1))


def walk_positions(
    z: np.ndarray,
    tradable: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_holding: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Position in the spread at each close (+1 long, -1 short, 0 flat) and why trades ended.

    Day by day, using only that day's z-score:
    - Flat: enter against the spread when entry_z < |z| < stop_z (short if z > 0, long if z < 0).
    - In a trade: close when the spread has come back within exit_z of zero or crossed it
      ("target"), when it has moved against the trade beyond stop_z ("stop"), or after
      max_holding days ("time"), checked in that order.
    - After a stop or a time-out there is no new entry until |z| < exit_z again.
    - A day on which the pair is not tradable, or has no z-score, closes any open trade
      ("not_cointegrated") and allows no entry.

    A closed trade is never reopened on the same day. The second array holds the exit reason
    on the day a trade is closed and "" elsewhere.

    The position depends on the whole path so far, so this is an explicit loop over days.
    """
    position = np.zeros(len(z), dtype=int)
    reason = np.full(len(z), "", dtype=object)
    current, held, locked = 0, 0, False
    for t, (z_t, ok) in enumerate(zip(z, tradable, strict=True)):
        if not ok or np.isnan(z_t):
            if current != 0:
                reason[t] = EXIT_NOT_COINTEGRATED
            current, locked = 0, False
        elif current != 0:
            held += 1
            against = -current * z_t  # how far the spread still is on the losing side
            if against < exit_z:
                current, reason[t] = 0, EXIT_TARGET
            elif against >= stop_z:
                current, locked, reason[t] = 0, True, EXIT_STOP
            elif held >= max_holding:
                current, locked, reason[t] = 0, True, EXIT_TIME
        else:
            if locked and abs(z_t) < exit_z:
                locked = False
            if not locked and entry_z < abs(z_t) < stop_z:
                current, held = (-1 if z_t > 0 else 1), 0
        position[t] = current
    return position, reason


class PairsTrading(Strategy):
    """Trades the spread of each pair (y, x) while the pair looks cointegrated.

    Every `reestimate` days, counted from the first row, the last `formation` closes give:
    - the hedge ratio: OLS of log(y) on log(x) with an intercept, so the spread is
      log(y) - alpha - beta * log(x);
    - the Engle-Granger cointegration test on that same window (statsmodels `coint`, which
      uses MacKinnon's critical values for residuals of an estimated regression);
    - the standard deviation of the spread in the window (its mean is zero by construction).

    These values are frozen until the next re-estimation, so the z-score of day t only uses
    prices up to t. A pair is tradable while its last test has a p-value below `p_value` and a
    positive beta. A window with a missing price in either leg is not tradable.

    The trading rules are in `walk_positions`. Long the spread means long y and short x, with
    the legs in proportion 1 : beta. The gross exposure is 1 whenever any pair has a trade
    open, shared equally among the pairs with an open trade, and 0 otherwise.

    Orders are only given on days when the target changes: a trade is opened or closed, or
    the hedge ratio of an open trade is re-estimated. The other rows are all-NaN ("no order"),
    so the engine lets the legs drift instead of trading back to the target every day. Run it
    with `rebalance=None`; a rebalancing calendar would delay entries and exits.

    The Engle-Granger test has low power on a one-year window and is repeated many times, so
    "tradable" is a noisy filter, not proof of cointegration.
    """

    name = "pairs_trading"

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        formation: int = 252,
        reestimate: int = 21,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_holding: int = 63,
        p_value: float = 0.05,
    ):
        pairs = [tuple(pair) for pair in pairs]
        if not pairs:
            raise ValueError("at least one pair is required")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
            raise ValueError("each pair must be two different tickers")
        if len(set(pairs)) != len(pairs):
            raise ValueError("pairs must not be repeated")
        if formation < 3:
            raise ValueError("formation must be at least 3")
        if reestimate < 1:
            raise ValueError("reestimate must be at least 1")
        if not 0.0 <= exit_z < entry_z < stop_z:
            raise ValueError("thresholds must satisfy 0 <= exit_z < entry_z < stop_z")
        if max_holding < 1:
            raise ValueError("max_holding must be at least 1")
        if not 0.0 < p_value < 1.0:
            raise ValueError("p_value must be between 0 and 1")
        self.pairs = pairs
        self.formation, self.reestimate = formation, reestimate
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.max_holding, self.p_value = max_holding, p_value

    # --- estimation ---------------------------------------------------------------------

    def _check_tickers(self, prices: pd.DataFrame) -> None:
        missing = sorted({t for pair in self.pairs for t in pair} - set(prices.columns))
        if missing:
            raise ValueError(f"tickers not found in prices: {', '.join(missing)}")

    def _formation(self, prices: pd.DataFrame, pair: tuple[str, str], test: bool) -> pd.DataFrame:
        """Daily alpha, beta, spread std and (if `test`) Engle-Granger p-value in force.

        Each formation row f uses rows f-formation+1..f and its values hold for rows
        f..f+reestimate-1. Rows before the first formation row are NaN.
        """
        log_y = np.log(prices[pair[0]].to_numpy(dtype=float))
        log_x = np.log(prices[pair[1]].to_numpy(dtype=float))
        values = np.full((len(prices), 4), np.nan)
        for f in range(self.formation - 1, len(prices), self.reestimate):
            y, x = log_y[f - self.formation + 1 : f + 1], log_x[f - self.formation + 1 : f + 1]
            if np.isnan(y).any() or np.isnan(x).any():
                continue
            alpha, beta, residual_std = hedge_ratio(y, x)
            if not residual_std > 0:  # also NaN: no spread to trade
                continue
            p = coint(y, x, trend="c")[1] if test else np.nan
            values[f : f + self.reestimate] = (alpha, beta, residual_std, p)
        table = pd.DataFrame(
            values, index=prices.index, columns=["alpha", "beta", "spread_std", "p_value"]
        )
        table["z"] = (log_y - table["alpha"] - table["beta"] * log_x) / table["spread_std"]
        return table

    def _state(self, prices: pd.DataFrame, pair: tuple[str, str]) -> pd.DataFrame:
        """Formation values plus whether the pair is tradable, its position and exit reasons."""
        state = self._formation(prices, pair, test=True)
        state["tradable"] = (state["p_value"] < self.p_value) & (state["beta"] > 0)
        state["position"], state["exit_reason"] = walk_positions(
            state["z"].to_numpy(),
            state["tradable"].to_numpy(),
            self.entry_z,
            self.exit_z,
            self.stop_z,
            self.max_holding,
        )
        return state

    def _per_pair(self, prices: pd.DataFrame, column: str, test: bool = True) -> pd.DataFrame:
        self._check_tickers(prices)
        build = self._state if test else (lambda p, pair: self._formation(p, pair, test=False))
        return pd.DataFrame({f"{y}/{x}": build(prices, (y, x))[column] for y, x in self.pairs})

    # --- diagnostics (one column per pair, named "Y/X") ---------------------------------

    def hedge_ratios(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Beta in force each day."""
        return self._per_pair(prices, "beta", test=False)

    def zscore(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Spread divided by its formation-window std, whether or not the pair is tradable."""
        return self._per_pair(prices, "z", test=False)

    def tradable(self, prices: pd.DataFrame) -> pd.DataFrame:
        """True on the days the pair passed its last cointegration test with beta > 0."""
        return self._per_pair(prices, "tradable")

    def positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Position in each spread decided at each close: +1 long, -1 short, 0 flat."""
        return self._per_pair(prices, "position")

    def trades(self, prices: pd.DataFrame) -> pd.DataFrame:
        """One row per trade: pair, side, entry and exit dates, days held and exit reason.

        Dates are those of the closes at which the decision is taken. A trade still open on
        the last day has no exit date and the reason "open".
        """
        self._check_tickers(prices)
        rows = []
        for y, x in self.pairs:
            state = self._state(prices, (y, x))
            position = state["position"].to_numpy()
            previous = np.concatenate([[0], position[:-1]])
            entries = np.flatnonzero((position != 0) & (previous == 0))
            exits = np.flatnonzero(state["exit_reason"].to_numpy() != "")
            for i, start in enumerate(entries):
                closed = i < len(exits)
                end = exits[i] if closed else len(position) - 1
                rows.append(
                    {
                        "pair": f"{y}/{x}",
                        "side": "long" if position[start] > 0 else "short",
                        "entry": prices.index[start],
                        "exit": prices.index[end] if closed else pd.NaT,
                        "days": int(end - start),
                        "reason": state["exit_reason"].iloc[end] if closed else STILL_OPEN,
                    }
                )
        columns = ["pair", "side", "entry", "exit", "days", "reason"]
        return pd.DataFrame(rows, columns=columns)

    # --- weights ------------------------------------------------------------------------

    def weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        self._check_tickers(prices)
        legs = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_active = pd.Series(0, index=prices.index)
        changed = pd.Series(False, index=prices.index)
        for y, x in self.pairs:
            state = self._state(prices, (y, x))
            position, beta = state["position"], state["beta"]
            is_open = position != 0
            # One unit of gross exposure per pair: 1 / (1 + beta) in y and beta / (1 + beta) in x.
            y_leg = (position / (1 + beta)).where(is_open, 0.0)
            legs[y] += y_leg
            legs[x] -= y_leg * beta.where(is_open, 0.0)
            n_active += is_open.astype(int)
            changed |= position.ne(position.shift(1, fill_value=0))
            changed |= is_open & beta.ne(beta.shift(1))
        target = legs.div(n_active.where(n_active > 0), axis=0).fillna(0.0)
        return target.where(changed, axis=0)
