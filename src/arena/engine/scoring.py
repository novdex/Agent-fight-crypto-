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

__all__ = ["realized_return_pct", "score_signal"]


def realized_return_pct(price_then: float, price_now: float) -> float:
    """Realized % move from ``price_then`` to ``price_now``."""
    return 100.0 * (price_now - price_then) / price_then


def score_signal(
    sig: Signal, price_now: float, *, flat_threshold_pct: float = 1.0
) -> float:
    """Score one signal against the realized price. Result is in [-1, 1]."""
    r = realized_return_pct(sig.price_at_signal, price_now)
    if sig.direction == Direction.LONG:
        raw = r
    elif sig.direction == Direction.SHORT:
        raw = -r
    else:  # FLAT
        raw = flat_threshold_pct - abs(r)
    return sig.confidence * math.tanh(raw / 5.0)
