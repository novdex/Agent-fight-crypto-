"""OpenAI-compatible chat-completions agent (OpenAI, Gemini, xAI, ...)."""

from __future__ import annotations

import time

import httpx

from arena.config import api_key_for
from arena.agents.base import AgentError, BaseAgent
from arena.agents.prompts import build_prompt, parse_signals
from arena.models import AgentSpec, Direction, MarketSnapshot, Signal

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


class OpenAICompatAgent(BaseAgent):
    def __init__(self, spec: AgentSpec):
        super().__init__(spec)
        self._api_key = api_key_for(spec.api_key_env)
        self.last_raw: str = ""
        self.last_usage: dict = {}

    def _post_with_retry(self, url: str, body: dict) -> httpx.Response:
        """POST with exponential backoff on transient failures (httpx has no
        built-in retries, unlike the Anthropic SDK)."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                    timeout=120.0,
                )
                if response.status_code in _RETRYABLE_STATUS:
                    last_exc = AgentError(
                        f"HTTP {response.status_code} from {url} for agent {self.name!r}"
                    )
                else:
                    response.raise_for_status()
                    return response
            except httpx.TransportError as exc:  # connect/read/write failures
                last_exc = exc
            except httpx.HTTPStatusError as exc:  # non-retryable 4xx
                raise AgentError(
                    f"HTTP request failed for agent {self.name!r}: {exc}"
                ) from exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2.0 ** (attempt + 1))  # 2s, 4s
        raise AgentError(
            f"HTTP request failed for agent {self.name!r} after "
            f"{_MAX_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    def ask(self, prompt: str) -> str:
        """Free-form single-turn completion (debate phase)."""
        base_url = self.spec.base_url.rstrip("/")
        try:
            response = self._post_with_retry(
                f"{base_url}/chat/completions",
                {"model": self.spec.model, "messages": [{"role": "user", "content": prompt}]},
            )
            text = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentError(f"ask failed for agent {self.name!r}: {exc}") from exc
        return text if isinstance(text, str) else ""

    def _logprob_refine(self, sig: Signal, snapshot: MarketSnapshot) -> Signal:
        """Replace verbalized probabilities with logprob-derived ones (#22).

        One single-token classification call per coin: softmax over the
        LONG/SHORT/FLAT top-logprobs (Kadavath et al. 2022 — model-internal
        probabilities are better calibrated than verbalized ones). Any
        failure keeps the verbalized vector.
        """
        import math

        coin = snapshot.coin(sig.symbol)
        if coin is None:
            return sig
        base_url = self.spec.base_url.rstrip("/")
        prompt = (
            f"{sig.symbol} at ${coin.price_usd:,.4f}: 24h change "
            f"{coin.change_24h_pct if coin.change_24h_pct is not None else 'n/a'}%, "
            f"RSI {coin.rsi_14 if coin.rsi_14 is not None else 'n/a'}. Over the "
            "next 24h, will it move >+1% (LONG), <-1% (SHORT), or in between "
            "(FLAT)? Answer with exactly one word: LONG, SHORT, or FLAT."
        )
        try:
            response = self._post_with_retry(
                f"{base_url}/chat/completions",
                {
                    "model": self.spec.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2,
                    "logprobs": True,
                    "top_logprobs": 10,
                },
            )
            content = response.json()["choices"][0]["logprobs"]["content"]
            top = content[0]["top_logprobs"]
            logits = {"LONG": None, "SHORT": None, "FLAT": None}
            for entry in top:
                token = str(entry.get("token", "")).strip().upper()
                if token in logits and logits[token] is None:
                    logits[token] = float(entry["logprob"])
            floor = min(v for v in logits.values() if v is not None) - 2.0
            exps = {k: math.exp(v if v is not None else floor) for k, v in logits.items()}
            total = sum(exps.values())
            p = {k: v / total for k, v in exps.items()}
        except Exception:
            return sig  # keep the verbalized vector
        direction = max(p, key=p.get)
        return sig.model_copy(
            update={
                "p_long": p["LONG"],
                "p_short": p["SHORT"],
                "p_flat": p["FLAT"],
                "direction": Direction(direction),
                "confidence": p[direction],
            }
        )

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        prompt = self.prompt_preamble + build_prompt(snapshot)
        base_url = self.spec.base_url.rstrip("/")
        body: dict = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt}],
            # Structured outputs (improvement #92); some compat endpoints
            # reject the parameter — we retry without it below.
            "response_format": {"type": "json_object"},
        }
        try:
            try:
                response = self._post_with_retry(f"{base_url}/chat/completions", body)
            except AgentError as exc:
                # Immediate 4xx rejection usually means the endpoint doesn't
                # support response_format — drop it and try once more.
                # Exhausted-transient-retries errors ("after N attempts")
                # are real outages: re-raise rather than doubling traffic.
                if "attempts" in str(exc) or "response_format" not in body:
                    raise
                body.pop("response_format", None)
                response = self._post_with_retry(f"{base_url}/chat/completions", body)
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            self.last_usage = {
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
        except httpx.HTTPError as exc:
            raise AgentError(f"HTTP request failed for agent {self.name!r}: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentError(
                f"Malformed chat-completions response for agent {self.name!r}: {exc}"
            ) from exc

        if not isinstance(text, str):
            raise AgentError(f"Non-text completion content for agent {self.name!r}")
        self.last_raw = text
        try:
            signals = parse_signals(text, self.name, snapshot)
            if self.spec.logprob_probs:
                signals = [self._logprob_refine(s, snapshot) for s in signals]
            return signals
        except Exception as exc:  # one agent's failure must never crash a round
            raise AgentError(f"Failed to parse reply from agent {self.name!r}: {exc}") from exc
