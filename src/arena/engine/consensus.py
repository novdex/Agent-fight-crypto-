"""Weighted-confidence consensus across agents' signals."""

from __future__ import annotations

from arena.models import Direction, Signal

__all__ = ["consensus_signals"]

CONSENSUS_AGENT = "consensus"


def consensus_signals(
    signals_by_agent: dict[str, list[Signal]],
    weights: dict[str, float],
) -> list[Signal]:
    """Build one consensus signal per symbol (union across agents).

    Per symbol: ``mass(direction) = sum(weights[a] * confidence)`` over agents
    voting that direction (agents absent from ``weights`` contribute 0).
    Winning direction = argmax mass; a tie for the top mass -> FLAT.
    Confidence = winning_mass / total_mass (0 if total is 0).
    """
    # Union of symbols, preserving first-seen order; remember signals per symbol.
    signals_for_symbol: dict[str, list[tuple[str, Signal]]] = {}
    for agent, signals in signals_by_agent.items():
        for sig in signals:
            signals_for_symbol.setdefault(sig.symbol, []).append((agent, sig))

    result: list[Signal] = []
    for symbol, contributions in signals_for_symbol.items():
        mass: dict[Direction, float] = {d: 0.0 for d in Direction}
        for agent, sig in contributions:
            mass[sig.direction] += weights.get(agent, 0.0) * sig.confidence

        total_mass = sum(mass.values())
        top_mass = max(mass.values())
        leaders = [d for d, m in mass.items() if m == top_mass]
        direction = leaders[0] if len(leaders) == 1 else Direction.FLAT

        winning_mass = mass[direction]
        confidence = winning_mass / total_mass if total_mass > 0.0 else 0.0

        rationale = (
            f"LONG {mass[Direction.LONG]:.2f}"
            f" vs SHORT {mass[Direction.SHORT]:.2f}"
            f" vs FLAT {mass[Direction.FLAT]:.2f}"
        )

        result.append(
            Signal(
                agent=CONSENSUS_AGENT,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                rationale=rationale,
                price_at_signal=contributions[0][1].price_at_signal,
            )
        )
    return result
