"""Anthropic (Claude) agent using the official anthropic SDK."""

from __future__ import annotations

import anthropic

from arena.config import api_key_for
from arena.agents.base import AgentError, BaseAgent
from arena.agents.prompts import build_prompt, parse_signals
from arena.models import AgentSpec, MarketSnapshot, Signal

#: JSON schema enforced via structured outputs (improvement #92): malformed
#: replies become impossible instead of degrading to all-FLAT.
SIGNALS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "p_long": {"type": "number"},
                    "p_short": {"type": "number"},
                    "p_flat": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["symbol", "p_long", "p_short", "p_flat", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["signals"],
    "additionalProperties": False,
}


class AnthropicAgent(BaseAgent):
    def __init__(self, spec: AgentSpec):
        super().__init__(spec)
        # The SDK retries 429/5xx with backoff on its own; cap request time so
        # a hung connection can't eat the round's whole agent budget.
        self._client = anthropic.Anthropic(
            api_key=api_key_for(spec.api_key_env), timeout=240.0, max_retries=3
        )
        self.last_raw: str = ""
        self.last_usage: dict = {}

    def _create(self, prompt: str, *, structured: bool):
        kwargs: dict = dict(
            model=self.spec.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        if structured:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": SIGNALS_SCHEMA}
            }
        return self._client.messages.create(**kwargs)

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        prompt = build_prompt(snapshot)
        try:
            try:
                response = self._create(prompt, structured=True)
            except anthropic.BadRequestError as exc:
                # Older models/endpoints may not support output_config —
                # retry once without structured outputs.
                if "output_config" not in str(exc):
                    raise
                response = self._create(prompt, structured=False)
        except anthropic.AnthropicError as exc:
            raise AgentError(f"Anthropic API call failed for agent {self.name!r}: {exc}") from exc

        try:
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            self.last_raw = text
            usage = getattr(response, "usage", None)
            self.last_usage = (
                {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                }
                if usage is not None
                else {}
            )
            return parse_signals(text, self.name, snapshot)
        except Exception as exc:  # one agent's failure must never crash a round
            raise AgentError(f"Failed to parse reply from agent {self.name!r}: {exc}") from exc
