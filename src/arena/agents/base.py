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
        #: Text prepended to the round prompt by providers — the CLI composes
        #: persona, memory lessons, debate brief etc. into it per round.
        self.prompt_preamble: str = ""

    @property
    def name(self) -> str:
        return self.spec.name

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        raise NotImplementedError

    def ask(self, prompt: str) -> str:
        """Free-form single-turn completion (used by the debate phase)."""
        raise NotImplementedError
