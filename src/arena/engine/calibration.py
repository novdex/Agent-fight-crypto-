"""Per-agent probability calibration (U3).

LLM agents are notoriously overconfident: an agent that says "p=0.9" may hit
only 60% of the time. This module learns a per-agent mapping from *stated*
probability to *empirical* hit rate and rewrites signal probability vectors
accordingly, so the Brier-scored arena rewards honest, calibrated forecasts.

Components (see docs/INTERFACES2.md, U3):

- :class:`Calibrator` — "platt" (logistic recalibration, fitted by plain
  gradient descent — stdlib ``math`` only), "isotonic" (pool-adjacent-
  violators step function) or "none" (identity). With fewer than
  ``MIN_SAMPLES`` (10) training pairs every method behaves as identity.
- :func:`calibrate_signal` / :meth:`Calibrator.calibrate_signal` — apply the
  fitted map to a signal's (p_long, p_short, p_flat) vector, renormalize and
  refresh direction (argmax) and confidence (max prob).
- :func:`early_damping` — blend toward the uniform vector for an agent's
  first rounds: ``p' = 0.7 p + 0.3 * (1/3)``.
- :func:`calibration_table` / :func:`drift_flag` /
  :func:`overconfidence_flag` — reliability diagnostics for the leaderboard
  and weight haircuts.
- :func:`log_score` — logarithmic scoring rule, clamped at -6.
- :class:`CalibrationLog` — tiny sqlite-backed (p, hit) history per agent
  (own table ``calib_predictions``).
"""

from __future__ import annotations

import math
import sqlite3
from typing import Sequence, Union

from arena.models import Direction, Signal

__all__ = [
    "Calibrator",
    "CalibrationLog",
    "calibrate_signal",
    "calibration_table",
    "drift_flag",
    "early_damping",
    "log_score",
    "overconfidence_flag",
]

_DIRECTIONS = (Direction.LONG, Direction.SHORT, Direction.FLAT)
_UNIFORM = 1.0 / 3.0
_LOG_SCORE_FLOOR = -6.0


def _clamp01(p: float) -> float:
    return min(1.0, max(0.0, p))


def _sigmoid(z: float) -> float:
    # Numerically safe logistic.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _prob_vector(sig: Signal) -> dict[Direction, float]:
    """Signal's normalized probability vector (requires ``has_probs``)."""
    probs = {
        Direction.LONG: sig.p_long or 0.0,
        Direction.SHORT: sig.p_short or 0.0,
        Direction.FLAT: sig.p_flat or 0.0,
    }
    total = sum(probs.values())
    if total > 0:  # tolerate slightly unnormalized vectors
        return {d: p / total for d, p in probs.items()}
    return {d: _UNIFORM for d in probs}


def _with_probs(sig: Signal, probs: dict[Direction, float]) -> Signal:
    """Copy ``sig`` with a new prob vector; refresh direction & confidence.

    Direction becomes the argmax of the vector (the original direction wins
    ties so a pure rescaling never flips a call); confidence becomes the max
    probability.
    """
    total = sum(probs.values())
    if total > 0:
        probs = {d: p / total for d, p in probs.items()}
    else:
        probs = {d: _UNIFORM for d in _DIRECTIONS}
    best = max(probs.values())
    direction = sig.direction
    if probs.get(direction, 0.0) < best - 1e-12:
        direction = next(d for d in _DIRECTIONS if probs[d] >= best - 1e-12)
    return sig.model_copy(
        update={
            "p_long": probs[Direction.LONG],
            "p_short": probs[Direction.SHORT],
            "p_flat": probs[Direction.FLAT],
            "direction": direction,
            "confidence": _clamp01(best),
        }
    )


class Calibrator:
    """Maps stated probabilities to empirically calibrated ones.

    ``method`` is one of:

    - ``"platt"`` — fit ``sigmoid(a*p + b)`` to (p, hit) pairs by gradient
      descent on the logistic log-loss (closed-form-free, stdlib only).
    - ``"isotonic"`` — pool-adjacent-violators monotone step function.
    - ``"none"`` — identity.

    Guard: with fewer than :attr:`MIN_SAMPLES` training pairs, :meth:`apply`
    is the identity regardless of method (too little data to trust a fit).
    """

    MIN_SAMPLES = 10

    def __init__(self, method: str = "platt") -> None:
        method = method.lower().strip()
        if method not in ("platt", "isotonic", "none"):
            raise ValueError(f"unknown calibration method: {method!r}")
        self.method = method
        self._fitted = False
        # Platt parameters.
        self._a = 1.0
        self._b = 0.0
        # Isotonic step function: parallel lists of block upper-p bounds and
        # fitted values, both non-decreasing in value.
        self._iso_bounds: list[float] = []
        self._iso_values: list[float] = []

    # -- fitting -----------------------------------------------------------

    def fit(self, predictions: Sequence[tuple[float, bool]]) -> None:
        """Fit on (stated probability of the realized class, hit) pairs."""
        self._fitted = False
        if self.method == "none" or len(predictions) < self.MIN_SAMPLES:
            return
        pairs = [(_clamp01(float(p)), 1.0 if hit else 0.0) for p, hit in predictions]
        if self.method == "platt":
            self._fit_platt(pairs)
        else:
            self._fit_isotonic(pairs)
        self._fitted = True

    def _fit_platt(
        self, pairs: list[tuple[float, float]], *, lr: float = 0.5, iters: int = 2000
    ) -> None:
        """Logistic regression hit ~ sigmoid(a*p + b) by gradient descent."""
        a, b = 1.0, 0.0
        n = float(len(pairs))
        for _ in range(iters):
            grad_a = 0.0
            grad_b = 0.0
            for p, y in pairs:
                err = _sigmoid(a * p + b) - y
                grad_a += err * p
                grad_b += err
            a -= lr * grad_a / n
            b -= lr * grad_b / n
        self._a, self._b = a, b

    def _fit_isotonic(self, pairs: list[tuple[float, float]]) -> None:
        """Pool-adjacent-violators on hits sorted by stated probability."""
        ordered = sorted(pairs, key=lambda t: t[0])
        # Blocks: [value, weight, max_p]; merge while monotonicity violated.
        blocks: list[list[float]] = []
        for p, y in ordered:
            blocks.append([y, 1.0, p])
            while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
                v2, w2, p2 = blocks.pop()
                v1, w1, _ = blocks.pop()
                w = w1 + w2
                blocks.append([(v1 * w1 + v2 * w2) / w, w, p2])
        self._iso_bounds = [blk[2] for blk in blocks]
        self._iso_values = [blk[0] for blk in blocks]

    # -- application -------------------------------------------------------

    def apply(self, p: float) -> float:
        """Calibrated probability for stated probability ``p``."""
        p = _clamp01(p)
        if not self._fitted:
            return p
        if self.method == "platt":
            return _sigmoid(self._a * p + self._b)
        # Isotonic: value of the first block whose upper bound covers p.
        for bound, value in zip(self._iso_bounds, self._iso_values):
            if p <= bound:
                return _clamp01(value)
        return _clamp01(self._iso_values[-1]) if self._iso_values else p

    def calibrate_signal(self, sig: Signal) -> Signal:
        """Apply calibration to the signal's prob vector (renormalized)."""
        if not sig.has_probs or not self._fitted:
            return sig
        probs = {d: self.apply(p) for d, p in _prob_vector(sig).items()}
        return _with_probs(sig, probs)


def calibrate_signal(sig: Signal, calibrator: Calibrator) -> Signal:
    """Module-level convenience wrapper for :meth:`Calibrator.calibrate_signal`."""
    return calibrator.calibrate_signal(sig)


def early_damping(sig: Signal, rounds_seen: int, *, threshold: int) -> Signal:
    """Blend a young agent's forecast toward uniform.

    While ``rounds_seen < threshold`` every probability becomes
    ``p' = 0.7 * p + 0.3 * (1/3)``; direction/confidence are refreshed from
    the damped vector. Legacy signals (no prob vector) get the same blend on
    ``confidence``. At or past the threshold the signal is unchanged.
    """
    if rounds_seen >= threshold:
        return sig
    if not sig.has_probs:
        return sig.model_copy(
            update={"confidence": _clamp01(0.7 * sig.confidence + 0.3 * _UNIFORM)}
        )
    probs = {d: 0.7 * p + 0.3 * _UNIFORM for d, p in _prob_vector(sig).items()}
    return _with_probs(sig, probs)


def calibration_table(
    history: list[tuple[float, bool]], bins: int = 10
) -> list[dict]:
    """Reliability table: per non-empty bin ``{p_mean, hit_rate, n}``.

    Stated probabilities are bucketed into ``bins`` equal-width bins over
    [0, 1]; used for drift flags and leaderboard display.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    sums = [0.0] * bins
    hits = [0] * bins
    counts = [0] * bins
    for p, hit in history:
        p = _clamp01(float(p))
        idx = min(int(p * bins), bins - 1)
        sums[idx] += p
        hits[idx] += 1 if hit else 0
        counts[idx] += 1
    return [
        {"p_mean": sums[i] / counts[i], "hit_rate": hits[i] / counts[i], "n": counts[i]}
        for i in range(bins)
        if counts[i] > 0
    ]


def drift_flag(table: list[dict], gap: float = 0.15) -> bool:
    """True when any populated bin's |stated - realized| gap exceeds ``gap``."""
    return any(
        row["n"] > 0 and abs(row["p_mean"] - row["hit_rate"]) > gap for row in table
    )


def overconfidence_flag(history: list[tuple[float, bool]]) -> bool:
    """True when mean stated p > 0.6 but hit rate < 0.5 on the window."""
    if not history:
        return False
    n = len(history)
    mean_p = sum(_clamp01(float(p)) for p, _ in history) / n
    hit_rate = sum(1 for _, hit in history if hit) / n
    return mean_p > 0.6 and hit_rate < 0.5


def log_score(sig: Signal, outcome: Direction) -> float:
    """Logarithmic scoring rule: ``log P[outcome]``, clamped at -6.

    Signals without a prob vector are treated as putting ``confidence`` on
    their stated direction and splitting the remainder over the other two
    outcomes.
    """
    if sig.has_probs:
        p = _prob_vector(sig)[outcome]
    elif outcome == sig.direction:
        p = sig.confidence
    else:
        p = (1.0 - sig.confidence) / 2.0
    if p <= 0.0:
        return _LOG_SCORE_FLOOR
    return max(math.log(min(p, 1.0)), _LOG_SCORE_FLOOR)


class CalibrationLog:
    """Sqlite-backed per-agent (p, hit) prediction history.

    Owns the namespaced table ``calib_predictions(agent TEXT, p REAL,
    hit INTEGER, round_id INTEGER)`` in the shared arena DB (or any path /
    connection handed in).
    """

    def __init__(self, db: Union[str, sqlite3.Connection]) -> None:
        if isinstance(db, sqlite3.Connection):
            self._conn = db
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(db)
            self._owns_conn = True
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calib_predictions (
                agent TEXT NOT NULL,
                p REAL NOT NULL,
                hit INTEGER NOT NULL,
                round_id INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calib_agent_round "
            "ON calib_predictions(agent, round_id)"
        )
        self._conn.commit()

    def add(self, agent: str, p: float, hit: bool, round_id: int) -> None:
        """Record one resolved prediction."""
        self._conn.execute(
            "INSERT INTO calib_predictions (agent, p, hit, round_id) "
            "VALUES (?, ?, ?, ?)",
            (agent, _clamp01(float(p)), 1 if hit else 0, int(round_id)),
        )
        self._conn.commit()

    def history(self, agent: str, window: int) -> list[tuple[float, bool]]:
        """Most recent ``window`` (p, hit) pairs for ``agent``, oldest first."""
        if window <= 0:
            return []
        rows = self._conn.execute(
            "SELECT p, hit FROM calib_predictions WHERE agent = ? "
            "ORDER BY round_id DESC, rowid DESC LIMIT ?",
            (agent, int(window)),
        ).fetchall()
        return [(float(p), bool(hit)) for p, hit in reversed(rows)]

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    def __enter__(self) -> "CalibrationLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
