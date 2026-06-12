"""Signal scoring: deterministic, bounded to [-1, 1].

Formula (see docs/INTERFACES.md, Unit 3):

    r = 100 * (price_now - price_then) / price_then
    raw = r                              if direction == LONG
    raw = -r                             if direction == SHORT
    raw = flat_threshold_pct - abs(r)    if direction == FLAT
    score = confidence * tanh(raw / 5.0)
"""

from __future__ import annotations

import math

from arena.models import Direction, Signal

__all__ = ["realized_class", "realized_return_pct", "score_signal"]


def realized_return_pct(price_then: float, price_now: float) -> float:
    """Realized % move from ``price_then`` to ``price_now``."""
    return 100.0 * (price_now - price_then) / price_then


def realized_class(r: float, flat_threshold_pct: float = 1.0) -> Direction:
    """Classify a realized % move as the LONG/SHORT/FLAT outcome."""
    if r > flat_threshold_pct:
        return Direction.LONG
    if r < -flat_threshold_pct:
        return Direction.SHORT
    return Direction.FLAT


def _brier_score(sig: Signal, outcome: Direction) -> float:
    """Strictly proper Brier score mapped to [-1, 1].

    score = 1 - sum_i (p_i - o_i)^2 over {LONG, SHORT, FLAT}, where o is the
    one-hot realized outcome. A perfect confident call scores +1, a fully
    confident miss scores -1, the uniform forecast scores 1/3. Unlike the
    legacy confidence×tanh rule this is incentive-compatible: an agent
    maximizes expected score only by reporting its true probabilities
    (Gneiting & Raftery 2007).
    """
    probs = {
        Direction.LONG: sig.p_long or 0.0,
        Direction.SHORT: sig.p_short or 0.0,
        Direction.FLAT: sig.p_flat or 0.0,
    }
    total = sum(probs.values())
    if total > 0:  # tolerate slightly unnormalized vectors
        probs = {d: p / total for d, p in probs.items()}
    sq_err = sum((p - (1.0 if d == outcome else 0.0)) ** 2 for d, p in probs.items())
    return 1.0 - sq_err


def score_signal(
    sig: Signal, price_now: float, *, flat_threshold_pct: float = 1.0
) -> float:
    """Score one signal against the realized price. Result is in [-1, 1].

    Signals carrying a probability vector are scored with the strictly proper
    Brier rule against the realized LONG/SHORT/FLAT class. Legacy signals
    (direction + confidence only) keep the original confidence×tanh rule.
    """
    r = realized_return_pct(sig.price_at_signal, price_now)
    if sig.has_probs:
        return _brier_score(sig, realized_class(r, flat_threshold_pct))
    if sig.direction == Direction.LONG:
        raw = r
    elif sig.direction == Direction.SHORT:
        raw = -r
    else:  # FLAT
        raw = flat_threshold_pct - abs(r)
    return sig.confidence * math.tanh(raw / 5.0)
