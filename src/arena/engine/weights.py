"""Decision-power weights: multiplicative-weights update with clamping."""

from __future__ import annotations

import math

__all__ = ["adaptive_eta", "equal_weights", "update_weights"]


def adaptive_eta(round_index: int, n_agents: int) -> float:
    """Anytime-optimal Hedge learning rate: eta(t) = sqrt(ln N / t).

    Large early updates while evidence is scarce, fine-grained adjustments as
    the sample grows — the schedule that achieves the O(sqrt(T ln N)) regret
    bound (Cesa-Bianchi & Lugosi 2006). ``round_index`` is 1-based.
    """
    t = max(round_index, 1)
    n = max(n_agents, 2)
    return math.sqrt(math.log(n) / t)


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
    normalize to sum 1, then project onto the box [min_weight, max_weight]
    so the bounds hold exactly AND the weights still sum to 1. (A naive
    clamp-then-renormalize can push the top weight back above max_weight;
    the iterative projection cannot.) If the bounds are infeasible for the
    agent count (n*min > 1 or n*max < 1), normalized weights are returned
    unclamped.
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

    n = len(updated)
    if n * min_weight > 1.0 or n * max_weight < 1.0:
        return updated  # bounds infeasible for this agent count

    # Iterative box projection: pin violators to their bound, rescale the
    # remaining mass over the free agents, repeat until nothing violates.
    pinned: dict[str, float] = {}
    free = dict(updated)
    while free:
        free_mass = 1.0 - sum(pinned.values())
        free_total = sum(free.values())
        if free_total <= 0.0:
            scaled = {name: free_mass / len(free) for name in free}
        else:
            scaled = {name: w / free_total * free_mass for name, w in free.items()}
        violations = {
            name: (min_weight if w < min_weight else max_weight)
            for name, w in scaled.items()
            if w < min_weight or w > max_weight
        }
        if not violations:
            return {**pinned, **scaled}
        pinned.update(violations)
        for name in violations:
            del free[name]
    s = sum(pinned.values())
    return {name: w / s for name, w in pinned.items()} if s > 0 else equal_weights(list(weights))
