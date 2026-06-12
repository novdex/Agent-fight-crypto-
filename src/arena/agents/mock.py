"""Deterministic offline mock agent — no network, no random module state."""

from __future__ import annotations

import hashlib

from arena.agents.base import BaseAgent
from arena.models import Direction, MarketSnapshot, Signal

_DIRECTIONS = (Direction.LONG, Direction.SHORT, Direction.FLAT)


class MockAgent(BaseAgent):
    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        seed = self.spec.seed
        signals: list[Signal] = []
        for coin in snapshot.coins:
            digest = hashlib.sha256(
                f"{seed}:{coin.symbol}:{snapshot.as_of.isoformat()}".encode()
            ).digest()
            direction = _DIRECTIONS[digest[0] % 3]
            # Map two digest bytes onto [0.3, 0.95].
            frac = int.from_bytes(digest[1:3], "big") / 0xFFFF
            confidence = round(0.3 + frac * 0.65, 4)
            signals.append(
                Signal(
                    agent=self.name,
                    symbol=coin.symbol,
                    direction=direction,
                    confidence=confidence,
                    rationale=f"mock deterministic signal (seed={seed})",
                    price_at_signal=coin.price_usd,
                )
            )
        return signals
