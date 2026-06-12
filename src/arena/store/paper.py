"""Paper-trading portfolio math.

A round's scored signals are turned into a simple equity update: an
equal-weight stake of ``equity * stake_fraction`` is split across the agent's
non-FLAT signals; FLAT signals hold cash.
"""

from __future__ import annotations

from arena.models import Direction, ScoredSignal


def apply_round_to_equity(
    equity: float,
    scored: list[ScoredSignal],
    *,
    stake_fraction: float = 0.5,
    fee_rate: float = 0.0,
) -> float:
    """Return the new equity after applying one round's scored signals.

    ``equity * stake_fraction`` is divided equally across the non-FLAT
    signals. Each position's PnL is::

        stake_per_position * (r / 100) * dir_mult - 2 * fee_rate * stake_per_position

    where ``r`` is the realized % move from ``price_at_signal`` to
    ``price_at_eval``, ``dir_mult`` is +1 for LONG, -1 for SHORT, and
    ``fee_rate`` is the per-side taker fee (e.g. 0.0005 = 5 bps; charged on
    entry and exit). FLAT signals hold cash, so an all-FLAT round leaves
    equity unchanged and pays no fees. Signals with a non-positive
    ``price_at_signal`` (invalid market data) also hold cash rather than
    dividing by zero.
    """
    positions = [
        s for s in scored if s.direction != Direction.FLAT and s.price_at_signal > 0
    ]
    if not positions:
        return equity
    stake_per_position = equity * stake_fraction / len(positions)
    pnl = 0.0
    for sig in positions:
        r = 100.0 * (sig.price_at_eval - sig.price_at_signal) / sig.price_at_signal
        dir_mult = 1.0 if sig.direction == Direction.LONG else -1.0
        pnl += stake_per_position * (r / 100.0) * dir_mult
        pnl -= 2.0 * fee_rate * stake_per_position  # entry + exit taker fees
    return equity + pnl
