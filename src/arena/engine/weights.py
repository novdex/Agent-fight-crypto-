"""Decision-power weights: multiplicative-weights update with clamping."""

from __future__ import annotations

import math

__all__ = ["equal_weights", "update_weights"]


def equal_weights(names: list[str]) -> dict[str, float]:
    """Equal weight ``1/len(names)`` per name. Empty list -> empty dict."""
    if not names:
        return {}
    share = 1.0 / len(names)
    return {name: share for name in names}


def update_weights(
    weights: dict[str, float],
    agent_scores: dict[str, float],
    *,
    eta: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    """Multiplicative-weights update.

    ``w[a] *= exp(eta * score[a])`` (missing score -> 0, i.e. unchanged),
    normalize to sum 1, clamp each weight to [min_weight, max_weight],
    then renormalize once.
    """
    if not weights:
        return {}

    updated = {
        name: w * math.exp(eta * agent_scores.get(name, 0.0))
        for name, w in weights.items()
    }

    total = sum(updated.values())
    if total <= 0.0:
        # Degenerate (all-zero or invalid weights): reset to equal weights.
        updated = equal_weights(list(weights))
    else:
        updated = {name: w / total for name, w in updated.items()}

    clamped = {
        name: min(max(w, min_weight), max_weight) for name, w in updated.items()
    }

    clamped_total = sum(clamped.values())
    if clamped_total <= 0.0:
        return equal_weights(list(weights))
    return {name: w / clamped_total for name, w in clamped.items()}
