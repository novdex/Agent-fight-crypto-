"""Consensus v2: probability pooling with extremizing, trimming, herding
haircut, an optional LMSR market mechanism, a no-trade gate, and crude
consensus intervals.

Per symbol the engine pools the agents' probability vectors over the three
outcomes (LONG, SHORT, FLAT) into one consensus vector, then post-processes:

- agents that report ``p_long``/``p_short``/``p_flat`` contribute their
  normalized vector; agents without probs contribute a one-hot-ish vector
  built from their stated direction with ``confidence`` mass on it and the
  remainder split evenly across the other two outcomes;
- weights come from the global ``weights`` book, overridden per agent when a
  ``per_asset_weights`` matrix (``{coin: {agent: w}}``) covers the coin;
- ``trim_fraction > 0`` drops the top/bottom fraction of contributing agents
  by weight before pooling (trimmed mean over forecasters);
- the pooled vector's max component is extremized with
  ``settings.extremize_lambda`` and the remaining components renormalized
  proportionally (Satopaa et al. style sharpening of the crowd forecast);
- when every contributing agent has the same argmax direction the final
  confidence is multiplied by ``settings.herding_haircut`` (unanimity is
  evidence of herding, not of independent agreement);
- ``mechanism="lmsr"`` replaces mean pooling with a 3-outcome LMSR market
  (cost C(q) = b * ln(sum_i exp(q_i / b))): each agent sequentially buys
  ``weight * confidence`` units of its argmax outcome and the final market
  prices are used as the pooled probabilities.
"""

from __future__ import annotations

import math

from arena.models import ConsensusSettings, Direction, Signal

__all__ = [
    "extremize",
    "consensus_signals_v2",
    "no_trade_gate",
    "consensus_interval",
]

CONSENSUS_AGENT = "consensus"

# Fixed outcome order for all probability vectors in this module.
_DIRS: tuple[Direction, Direction, Direction] = (
    Direction.LONG,
    Direction.SHORT,
    Direction.FLAT,
)

_UNIFORM: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def extremize(p: float, lam: float) -> float:
    """Extremize a probability: ``p^lam / (p^lam + (1-p)^lam)``.

    ``lam = 1`` is the identity; ``lam > 1`` pushes probabilities away from
    0.5 toward the nearer extreme. 0, 0.5 and 1 are fixed points.
    """
    p = min(1.0, max(0.0, p))
    if p in (0.0, 1.0):
        return p
    num = p**lam
    den = num + (1.0 - p) ** lam
    return num / den


def _agent_vector(sig: Signal) -> tuple[float, float, float]:
    """Normalized (p_long, p_short, p_flat) for one signal.

    Signals without a probability vector contribute a one-hot-ish vector:
    ``confidence`` mass on the stated direction, the rest split evenly.
    """
    if sig.has_probs:
        raw = (float(sig.p_long or 0.0), float(sig.p_short or 0.0), float(sig.p_flat or 0.0))
        total = sum(raw)
        if total <= 0.0:
            return _UNIFORM
        return (raw[0] / total, raw[1] / total, raw[2] / total)
    conf = min(1.0, max(0.0, sig.confidence))
    rest = (1.0 - conf) / 2.0
    return tuple(conf if d is sig.direction else rest for d in _DIRS)  # type: ignore[return-value]


def _argmax_direction(vec: tuple[float, float, float]) -> Direction:
    """Argmax outcome of a prob vector; a tie for the top -> FLAT."""
    top = max(vec)
    leaders = [d for d, p in zip(_DIRS, vec) if p == top]
    return leaders[0] if len(leaders) == 1 else Direction.FLAT


def _agent_direction(sig: Signal) -> Direction:
    """The direction an agent is effectively backing (for herding/LMSR)."""
    if sig.has_probs:
        return _argmax_direction(_agent_vector(sig))
    return sig.direction


def _trim(
    contributions: list[tuple[str, Signal, float]], trim_fraction: float
) -> list[tuple[str, Signal, float]]:
    """Drop the top/bottom ``trim_fraction`` of contributors by weight."""
    n = len(contributions)
    k = int(n * trim_fraction)
    if k <= 0:
        return contributions
    k = min(k, (n - 1) // 2)  # always keep at least one contributor
    if k <= 0:
        return contributions
    ordered = sorted(contributions, key=lambda c: (c[2], c[0]))
    return ordered[k : n - k]


def _pool_mean(
    contributions: list[tuple[str, Signal, float]],
) -> tuple[float, float, float]:
    """Weighted mean of the contributors' normalized prob vectors."""
    total_w = sum(w for _, _, w in contributions)
    if total_w <= 0.0:
        return _UNIFORM
    acc = [0.0, 0.0, 0.0]
    for _, sig, w in contributions:
        vec = _agent_vector(sig)
        for i in range(3):
            acc[i] += w * vec[i]
    return (acc[0] / total_w, acc[1] / total_w, acc[2] / total_w)


def _pool_lmsr(
    contributions: list[tuple[str, Signal, float]], b: float
) -> tuple[float, float, float]:
    """3-outcome LMSR market: each agent buys weight*confidence of its argmax
    outcome sequentially; final prices = softmax(q / b)."""
    b = max(b, 1e-9)
    q = [0.0, 0.0, 0.0]
    for _, sig, w in contributions:
        outcome = _agent_direction(sig)
        qty = w * min(1.0, max(0.0, sig.confidence))
        q[_DIRS.index(outcome)] += qty
    top = max(q)
    exps = [math.exp((qi - top) / b) for qi in q]
    z = sum(exps)
    return (exps[0] / z, exps[1] / z, exps[2] / z)


def _extremize_vector(
    vec: tuple[float, float, float], lam: float
) -> tuple[float, float, float]:
    """Extremize the max component, renormalize the rest proportionally.

    A tie for the top component is left untouched (the tie means FLAT and
    sharpening an arbitrary side would silently break it).
    """
    top = max(vec)
    if vec.count(top) > 1:
        return vec
    idx = vec.index(top)
    new_top = extremize(top, lam)
    rest = 1.0 - top
    out = [0.0, 0.0, 0.0]
    out[idx] = new_top
    if rest > 0.0:
        scale = (1.0 - new_top) / rest
        for i in range(3):
            if i != idx:
                out[i] = vec[i] * scale
    return (out[0], out[1], out[2])


def consensus_signals_v2(
    signals_by_agent: dict[str, list[Signal]],
    weights: dict[str, float],
    *,
    settings: ConsensusSettings,
    per_asset_weights: dict[str, dict[str, float]] | None = None,
) -> list[Signal]:
    """Pool agents' probability vectors into one consensus Signal per symbol.

    See the module docstring for the pooling pipeline. The output signals
    carry ``agent="consensus"``, the pooled vector on ``p_long``/``p_short``/
    ``p_flat``, ``direction = argmax`` of the pooled vector (tie -> FLAT),
    ``confidence = max prob`` (after the herding haircut, when unanimous) and
    a compact vote/price summary in ``rationale``.
    """
    # Union of symbols, preserving first-seen order (matches v1 conventions).
    signals_for_symbol: dict[str, list[tuple[str, Signal]]] = {}
    for agent, signals in signals_by_agent.items():
        for sig in signals:
            signals_for_symbol.setdefault(sig.symbol, []).append((agent, sig))

    result: list[Signal] = []
    for symbol, pairs in signals_for_symbol.items():
        overrides = (per_asset_weights or {}).get(symbol)
        contributions: list[tuple[str, Signal, float]] = []
        for agent, sig in pairs:
            w = weights.get(agent, 0.0)
            if overrides is not None and agent in overrides:
                w = overrides[agent]
            contributions.append((agent, sig, w))

        if settings.trim_fraction > 0.0:
            contributions = _trim(contributions, settings.trim_fraction)

        if settings.mechanism == "lmsr":
            pooled = _pool_lmsr(contributions, settings.lmsr_b)
        else:
            pooled = _pool_mean(contributions)

        pooled = _extremize_vector(pooled, settings.extremize_lambda)

        direction = _argmax_direction(pooled)
        confidence = max(pooled)

        agent_dirs = {_agent_direction(sig) for _, sig, _ in contributions}
        unanimous = len(agent_dirs) == 1
        if unanimous:
            confidence *= settings.herding_haircut

        label = "lmsr" if settings.mechanism == "lmsr" else "pool"
        rationale = (
            f"{label}[n={len(contributions)}]"
            f" L={pooled[0]:.2f}/S={pooled[1]:.2f}/F={pooled[2]:.2f}"
        )
        if unanimous:
            rationale += f" unanimous x{settings.herding_haircut:.2f}"

        result.append(
            Signal(
                agent=CONSENSUS_AGENT,
                symbol=symbol,
                direction=direction,
                confidence=min(1.0, max(0.0, confidence)),
                rationale=rationale,
                price_at_signal=pairs[0][1].price_at_signal,
                p_long=pooled[0],
                p_short=pooled[1],
                p_flat=pooled[2],
            )
        )
    return result


def no_trade_gate(
    consensus: list[Signal], min_confidence: float
) -> tuple[list[Signal], bool]:
    """Force the whole round FLAT when the consensus is collectively unsure.

    When the mean over symbols of the top pooled probability is below
    ``min_confidence``, every signal is replaced by a FLAT signal with
    confidence 0 and a uniform probability vector and ``gated=True`` is
    returned. ``min_confidence <= 0`` disables the gate.
    """
    if min_confidence <= 0.0 or not consensus:
        return consensus, False

    tops = [
        max(s.p_long, s.p_short, s.p_flat) if s.has_probs else s.confidence
        for s in consensus
    ]
    mean_top = sum(tops) / len(tops)
    if mean_top >= min_confidence:
        return consensus, False

    gated = [
        Signal(
            agent=CONSENSUS_AGENT,
            symbol=s.symbol,
            direction=Direction.FLAT,
            confidence=0.0,
            rationale=(
                f"no-trade gate: mean top-prob {mean_top:.2f}"
                f" < {min_confidence:.2f}"
            ),
            price_at_signal=s.price_at_signal,
            p_long=_UNIFORM[0],
            p_short=_UNIFORM[1],
            p_flat=_UNIFORM[2],
        )
        for s in consensus
    ]
    return gated, True


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile of pre-sorted values, q in [0, 1]."""
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def consensus_interval(
    signals_by_agent: dict[str, list[Signal]], symbol: str
) -> tuple[float, float]:
    """Crude (5th, 95th) percentile band of the contributing agents'
    probability for the winning direction on ``symbol``.

    The winning direction is the argmax of the unweighted mean of the agents'
    normalized prob vectors (tie -> FLAT). With fewer than two contributing
    agents the band collapses to ``(p, p)`` (and ``(0.0, 0.0)`` with none).
    """
    sym = symbol.upper()
    vectors: list[tuple[float, float, float]] = []
    for signals in signals_by_agent.values():
        for sig in signals:
            if sig.symbol.upper() == sym:
                vectors.append(_agent_vector(sig))

    if not vectors:
        return (0.0, 0.0)

    mean_vec = tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(3))
    winner_idx = _DIRS.index(_argmax_direction(mean_vec))  # type: ignore[arg-type]
    probs = sorted(v[winner_idx] for v in vectors)

    if len(probs) < 2:
        return (probs[0], probs[0])
    return (_percentile(probs, 0.05), _percentile(probs, 0.95))
