"""LLM agent factory. FROZEN INTERFACE — see docs/INTERFACES.md (Unit 2)."""

from __future__ import annotations

import hashlib
import sys

from arena.config import api_key_for
from arena.agents.base import AgentError, BaseAgent
from arena.agents.mock import MockAgent
from arena.models import AgentSpec

__all__ = ["AgentError", "BaseAgent", "MockAgent", "available_agents", "create_agent"]


def _stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def create_agent(spec: AgentSpec) -> BaseAgent:
    """Build an agent for the given spec, dispatching on spec.provider."""
    if spec.provider == "anthropic":
        from arena.agents.anthropic_agent import AnthropicAgent

        return AnthropicAgent(spec)
    if spec.provider == "openai_compat":
        from arena.agents.openai_compat import OpenAICompatAgent

        return OpenAICompatAgent(spec)
    if spec.provider == "mock":
        return MockAgent(spec)
    raise ValueError(f"Unknown agent provider: {spec.provider!r}")


def available_agents(specs: list[AgentSpec], *, mock: bool = False) -> list[BaseAgent]:
    """Build one agent per spec.

    With mock=True every spec is forced to a MockAgent (same name, seed derived
    from a stable hash of the name). Otherwise, specs whose API key env var is
    unset are skipped with a warning on stderr instead of failing.
    """
    agents: list[BaseAgent] = []
    for spec in specs:
        if mock:
            mock_spec = spec.model_copy(
                update={"provider": "mock", "seed": _stable_seed(spec.name)}
            )
            agents.append(MockAgent(mock_spec))
            continue
        if spec.provider != "mock" and not api_key_for(spec.api_key_env):
            print(
                f"warning: skipping agent {spec.name!r} — API key env var "
                f"{spec.api_key_env or '<unset>'} is not set",
                file=sys.stderr,
            )
            continue
        agents.append(create_agent(spec))
    return agents
