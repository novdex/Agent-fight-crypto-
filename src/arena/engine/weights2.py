"""Per-asset / per-regime weight machinery on top of the global Hedge update.

This module extends :mod:`arena.engine.weights` (whose update/projection
semantics are REUSED, never reimplemented) with:

* per-round multiplicative-factor clipping,
* effective-sample-size diagnostics for weight concentration,
* significance damping for young weight books,
* a decorrelation / diversity adjustment that penalizes near-clone agents
  and mildly rewards principled dissent,
* an EXP4-style per-coin weight-matrix update,
* regime-conditional weight-book keys,
* a deterministic sign-permutation skill test.

All functions are pure (no I/O, no global state) and stdlib-only.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable

from arena.engine.weights import equal_weights, update_weights
from arena.models import Direction

__all__ = [
    "bootstrap_skill_pvalue",
    "clip_update_factor",
    "decorrelation_adjust",
    "effective_sample_size",
    "regime_key",
    "significance_damping",
    "update_weight_matrix",
]

_KNOWN_REGIMES = ("bull", "bear", "chop")

# Numeric encoding of a direction for agreement bookkeeping.
_DIR_VALUE = {Direction.LONG: 1.0, Direction.SHORT: -1.0, Direction.FLAT: 0.0}


def clip_update_factor(factor: float, low: float, high: float) -> float:
    """Clamp a per-round multiplicative update factor into ``[low, high]``.

    Bounds how much decision power one round can move, no matter how extreme
    the round score is (cf. ``WeightsSettings.clip_factor_low/high``). If the
    caller passes inverted bounds they are swapped rather than raising.
    """
    if low > high:
        low, high = high, low
    return min(max(factor, low), high)


def effective_sample_size(weights: dict[str, float]) -> float:
    """Kish effective sample size: ``(sum w)^2 / sum(w^2)``.

    For weights normalized to sum 1 this is ``1 / sum(w^2)``: four equal
    weights give 4.0, one dominant weight gives a value near 1. Used to
    detect weight-book collapse (compare against ``WeightsSettings.ess_floor``).
    Returns 0.0 for an empty dict or non-positive total mass.
    """
    total = sum(weights.values())
    sq = sum(w * w for w in weights.values())
    if total <= 0.0 or sq <= 0.0:
        return 0.0
    return (total * total) / sq


def significance_damping(
    new_w: dict[str, float],
    old_w: dict[str, float],
    rounds_seen: int,
    threshold: int,
) -> dict[str, float]:
    """Blend a fresh weight update 50/50 with the previous weights while young.

    While ``rounds_seen < threshold`` the evidence is statistically weak, so
    each weight becomes ``0.5 * new + 0.5 * old`` (an agent missing from
    ``old_w`` falls back to the uniform prior ``1/n``), renormalized to sum 1.
    Once ``rounds_seen >= threshold`` the update is returned undamped.
    """
    if not new_w:
        return {}
    if rounds_seen >= threshold:
        return dict(new_w)
    prior = 1.0 / len(new_w)
    blended = {
        name: 0.5 * w + 0.5 * old_w.get(name, prior) for name, w in new_w.items()
    }
    total = sum(blended.values())
    if total <= 0.0:
        return equal_weights(list(new_w))
    return {name: w / total for name, w in blended.items()}


def _paired_directions(
    hist_a: Iterable[tuple[str, Direction]],
    hist_b: Iterable[tuple[str, Direction]],
) -> list[tuple[Direction, Direction]]:
    """Align two prediction histories into comparable (dir_a, dir_b) pairs.

    Each history is a chronological list of ``(symbol, direction)``. For every
    symbol both agents predicted, the k-th prediction of agent A on that
    symbol is paired with the k-th prediction of agent B on the same symbol
    (zip up to the shorter sequence) — i.e. same coin, same round position.
    """
    by_sym_a: dict[str, list[Direction]] = defaultdict(list)
    for sym, d in hist_a:
        by_sym_a[sym].append(d)
    by_sym_b: dict[str, list[Direction]] = defaultdict(list)
    for sym, d in hist_b:
        by_sym_b[sym].append(d)
    pairs: list[tuple[Direction, Direction]] = []
    for sym in sorted(by_sym_a.keys() & by_sym_b.keys()):
        pairs.extend(zip(by_sym_a[sym], by_sym_b[sym]))
    return pairs


def _agreement_correlation(pairs: list[tuple[Direction, Direction]]) -> float:
    """Direction-agreement correlation of two agents, in [-1, 1].

    Definition (concrete): for each paired prediction the agreement value is

    * ``+1`` when both agents call the same direction (LONG/LONG,
      SHORT/SHORT or FLAT/FLAT),
    * ``-1`` when they call strictly opposite directions (LONG vs SHORT),
    * ``0``  when exactly one of them is FLAT (partial disagreement).

    The agreement correlation is the mean of these values. Two clones score
    exactly +1, perfect contrarians score -1, and unrelated agents hover near
    0. Unlike Pearson correlation this is well defined even when an agent
    always emits the same direction (zero variance).
    """
    if not pairs:
        return 0.0
    total = 0.0
    for da, db in pairs:
        if da == db:
            total += 1.0
        elif _DIR_VALUE[da] * _DIR_VALUE[db] < 0.0:  # LONG vs SHORT
            total -= 1.0
        # one side FLAT, the other not -> 0
    return total / len(pairs)


def _direction_distribution(
    history: Iterable[tuple[str, Direction]],
) -> dict[Direction, float]:
    """Laplace-smoothed empirical distribution over LONG/SHORT/FLAT."""
    counts = {Direction.LONG: 1.0, Direction.SHORT: 1.0, Direction.FLAT: 1.0}
    for _, d in history:
        counts[d] += 1.0
    total = sum(counts.values())
    return {d: c / total for d, c in counts.items()}


def decorrelation_adjust(
    weights: dict[str, float],
    prediction_history: dict[str, list[tuple[str, Direction]]],
    *,
    corr_threshold: float = 0.8,
    penalty: float = 0.3,
    diversity_bonus: float = 0.05,
    min_overlap: int = 3,
) -> dict[str, float]:
    """Penalize near-clone agent pairs and mildly reward principled dissent.

    For every agent pair with at least ``min_overlap`` aligned predictions
    (see :func:`_paired_directions`) the direction-agreement correlation is
    computed (see :func:`_agreement_correlation`). When ``|corr| >
    corr_threshold`` the *weaker* agent of the pair — the one with the lower
    current weight, ties broken by sorted name — receives a multiplicative
    haircut of ``1 - penalty * (|corr| - corr_threshold) / (1 -
    corr_threshold)`` (up to ``1 - penalty`` for perfect clones/contrarians).
    Only the weaker agent pays: redundant information should cost the copy,
    not the source.

    Diversity bonus: each agent's smoothed direction distribution is compared
    to the pool average via KL divergence ``KL(agent || pool)``; the agent's
    weight is multiplied by ``1 + diversity_bonus * min(KL, 1)``. Agents that
    consistently dissent from the herd (in a measured, distribution-level
    sense, not single lucky calls) earn a small premium because they carry
    non-redundant information.

    The result is renormalized to sum 1. Agents without history are left
    untouched (multiplier 1) apart from renormalization.
    """
    if not weights:
        return {}

    mult = {name: 1.0 for name in weights}
    names = sorted(weights)

    # Pairwise clone penalty (haircut the weaker of each over-correlated pair).
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            hist_a = prediction_history.get(a, [])
            hist_b = prediction_history.get(b, [])
            pairs = _paired_directions(hist_a, hist_b)
            if len(pairs) < min_overlap:
                continue
            corr = _agreement_correlation(pairs)
            if abs(corr) <= corr_threshold:
                continue
            excess = (abs(corr) - corr_threshold) / max(1.0 - corr_threshold, 1e-9)
            haircut = 1.0 - penalty * min(excess, 1.0)
            weaker = min((a, b), key=lambda n: (weights[n], n))
            mult[weaker] *= haircut

    # KL-dissent diversity bonus.
    with_history = [n for n in names if prediction_history.get(n)]
    if diversity_bonus > 0.0 and len(with_history) >= 2:
        dists = {n: _direction_distribution(prediction_history[n]) for n in with_history}
        pool = {
            d: sum(dist[d] for dist in dists.values()) / len(dists)
            for d in (Direction.LONG, Direction.SHORT, Direction.FLAT)
        }
        for name in with_history:
            kl = sum(
                p * math.log(p / pool[d]) for d, p in dists[name].items() if p > 0.0
            )
            mult[name] *= 1.0 + diversity_bonus * min(max(kl, 0.0), 1.0)

    adjusted = {name: w * mult[name] for name, w in weights.items()}
    total = sum(adjusted.values())
    if total <= 0.0:
        return equal_weights(list(weights))
    return {name: w / total for name, w in adjusted.items()}


def update_weight_matrix(
    matrix: dict[str, dict[str, float]],
    scores: dict[str, dict[str, float]],
    *,
    eta: float,
    min_weight: float,
    max_weight: float,
    clip_low: float,
    clip_high: float,
) -> dict[str, dict[str, float]]:
    """Per-asset EXP4-style update: one multiplicative-weights step per coin.

    ``matrix`` is ``{coin: {agent: weight}}`` and ``scores`` is
    ``{coin: {agent: round_score}}``. For each coin the raw multiplicative
    factor ``exp(eta * score)`` is clipped into ``[clip_low, clip_high]`` via
    :func:`clip_update_factor`, then converted back to an equivalent score
    ``log(factor) / eta`` so that the actual update, normalization and
    min/max box projection are delegated verbatim to
    :func:`arena.engine.weights.update_weights` — the per-coin semantics are
    exactly the global Hedge semantics, just clipped. Agents without a score
    on a coin are unchanged (factor 1). The input matrix is not mutated;
    coins absent from ``scores`` are returned re-projected but unmoved.
    """
    low = max(clip_low, 1e-12)
    out: dict[str, dict[str, float]] = {}
    for coin, coin_weights in matrix.items():
        coin_scores = scores.get(coin, {})
        if eta > 0.0:
            adjusted: dict[str, float] = {}
            for agent, s in coin_scores.items():
                if agent not in coin_weights:
                    continue
                exponent = min(max(eta * s, -700.0), 700.0)
                factor = clip_update_factor(math.exp(exponent), low, clip_high)
                adjusted[agent] = math.log(factor) / eta
        else:
            # eta == 0 -> every factor is exp(0) == 1; scores are irrelevant.
            adjusted = {}
        out[coin] = update_weights(
            dict(coin_weights),
            adjusted,
            eta=eta,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    return out


def regime_key(regime: str) -> str:
    """Map a regime label to its weight-book key (``"regime:<name>"``).

    Case/whitespace-insensitive. Unknown labels fall back to the neutral
    ``"chop"`` book so an unrecognized regime never crashes a round.
    """
    r = regime.strip().lower()
    if r not in _KNOWN_REGIMES:
        r = "chop"
    return f"regime:{r}"


def bootstrap_skill_pvalue(
    scores: list[float], n_perm: int = 1000, seed: int = 0
) -> float:
    """One-sided sign-permutation test that the mean round score exceeds 0.

    Null hypothesis: scores are symmetric around zero (no skill). Each
    permutation flips the sign of every score independently with probability
    1/2 (seeded ``random.Random`` -> fully deterministic) and the p-value is
    the add-one-smoothed fraction of permuted means that reach the observed
    mean: ``(1 + #{perm_mean >= observed}) / (n_perm + 1)``. Clearly positive
    score series yield small p-values; symmetric noise yields roughly uniform
    p-values. Empty input returns 1.0 (no evidence of skill).
    """
    if not scores or n_perm <= 0:
        return 1.0
    observed = sum(scores) / len(scores)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        perm_sum = 0.0
        for s in scores:
            perm_sum += s if rng.random() < 0.5 else -s
        if perm_sum / len(scores) >= observed:
            hits += 1
    return (1 + hits) / (n_perm + 1)
