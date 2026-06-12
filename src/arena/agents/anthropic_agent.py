"""Anthropic (Claude) agent using the official anthropic SDK."""

from __future__ import annotations

import anthropic

from arena.config import api_key_for
from arena.agents.base import AgentError, BaseAgent
from arena.agents.prompts import build_prompt, parse_signals
from arena.models import AgentSpec, MarketSnapshot, Signal


class AnthropicAgent(BaseAgent):
    def __init__(self, spec: AgentSpec):
        super().__init__(spec)
        # The SDK retries 429/5xx with backoff on its own; cap request time so
        # a hung connection can't eat the round's whole agent budget.
        self._client = anthropic.Anthropic(
            api_key=api_key_for(spec.api_key_env), timeout=240.0, max_retries=3
        )

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        prompt = build_prompt(snapshot)
        try:
            response = self._client.messages.create(
                model=self.spec.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AnthropicError as exc:
            raise AgentError(f"Anthropic API call failed for agent {self.name!r}: {exc}") from exc

        try:
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            return parse_signals(text, self.name, snapshot)
        except Exception as exc:  # one agent's failure must never crash a round
            raise AgentError(f"Failed to parse reply from agent {self.name!r}: {exc}") from exc
