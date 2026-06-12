"""OpenAI-compatible chat-completions agent (OpenAI, Gemini, xAI, ...)."""

from __future__ import annotations

import httpx

from arena.config import api_key_for
from arena.agents.base import AgentError, BaseAgent
from arena.agents.prompts import build_prompt, parse_signals
from arena.models import AgentSpec, MarketSnapshot, Signal


class OpenAICompatAgent(BaseAgent):
    def __init__(self, spec: AgentSpec):
        super().__init__(spec)
        self._api_key = api_key_for(spec.api_key_env)

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        prompt = build_prompt(snapshot)
        base_url = self.spec.base_url.rstrip("/")
        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.spec.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120.0,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:
            raise AgentError(f"HTTP request failed for agent {self.name!r}: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentError(
                f"Malformed chat-completions response for agent {self.name!r}: {exc}"
            ) from exc

        if not isinstance(text, str):
            raise AgentError(f"Non-text completion content for agent {self.name!r}")
        try:
            return parse_signals(text, self.name, snapshot)
        except Exception as exc:  # one agent's failure must never crash a round
            raise AgentError(f"Failed to parse reply from agent {self.name!r}: {exc}") from exc
