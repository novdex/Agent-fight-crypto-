"""Agent base class and error type. FROZEN INTERFACE — see docs/INTERFACES.md."""

from __future__ import annotations

from arena.models import AgentSpec, MarketSnapshot, Signal


class AgentError(Exception):
    """Raised when an agent fails to produce signals (API failure, etc.).

    Callers catch this so a single agent's failure never crashes a round.
    """


class BaseAgent:
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        raise NotImplementedError
