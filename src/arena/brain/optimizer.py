"""OPRO-lite champion/challenger prompt-preamble store (single JSON file).

Three built-in preamble variants ship with the book (neutral, risk-emphasis,
contrarian-emphasis). Per agent, the book tracks each variant's running mean
round score; `preamble_for` mostly serves the champion (best mean) with a
periodic challenger probe, and `record_result` attributes the score to the
last-served variant and re-promotes the champion. Fully deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.brain.memory import atomic_write_json

VARIANTS: dict[str, str] = {
    "neutral": (
        "Analyze the market data objectively. Weigh momentum, mean-reversion "
        "and positioning evidence equally, and report honest probabilities."
    ),
    "risk_emphasis": (
        "Prioritize capital preservation. Demand strong confirming evidence "
        "before LONG/SHORT probabilities above 0.5; when signals conflict or "
        "volatility is elevated, lean toward FLAT and modest probabilities."
    ),
    "contrarian_emphasis": (
        "Hunt for crowded trades. When funding, long/short ratios or sentiment "
        "are at extremes, favor the contrarian side of the crowd; fade euphoric "
        "rallies and panic dumps rather than chasing them."
    ),
}

_VARIANT_ORDER: tuple[str, ...] = tuple(VARIANTS)
_DEFAULT_CHAMPION = "neutral"
_PROBE_ROUNDS = 1  # serve every variant at least this many times first
_CHALLENGER_EVERY = 4  # after probing, every Nth serve goes to the challenger


class PromptVariantBook:
    """Champion/challenger prompt-preamble book persisted as one JSON file."""

    def __init__(self, root: str):
        self.path = Path(root) / "prompt_variants.json"

    # -- persistence ---------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            state = {}
        return state if isinstance(state, dict) else {}

    def _save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.path, state)

    @staticmethod
    def _agent_state(state: dict[str, Any], agent: str) -> dict[str, Any]:
        entry = state.setdefault(agent, {})
        if not isinstance(entry, dict):
            entry = state[agent] = {}
        entry.setdefault("champion", _DEFAULT_CHAMPION)
        entry.setdefault("serves", 0)
        entry.setdefault("last_served", None)
        stats = entry.setdefault("stats", {})
        for name in _VARIANT_ORDER:
            stats.setdefault(name, {"n": 0, "mean": 0.0})
        return entry

    # -- public API ------------------------------------------------------------

    def preamble_for(self, agent: str) -> str:
        """Return the preamble to use this round, recording which was served."""
        state = self._load()
        entry = self._agent_state(state, agent)
        variant = self._choose(entry)
        entry["last_served"] = variant
        entry["serves"] = int(entry["serves"]) + 1
        self._save(state)
        return VARIANTS[variant]

    def record_result(self, agent: str, round_score: float) -> None:
        """Attribute `round_score` to the last-served variant; promote champion."""
        state = self._load()
        entry = self._agent_state(state, agent)
        variant = entry.get("last_served") or entry["champion"]
        if variant not in VARIANTS:
            variant = _DEFAULT_CHAMPION
        stat = entry["stats"][variant]
        n = int(stat["n"]) + 1
        stat["mean"] = float(stat["mean"]) + (round_score - float(stat["mean"])) / n
        stat["n"] = n
        entry["last_served"] = None
        entry["champion"] = self._best_variant(entry)
        self._save(state)

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _best_variant(entry: dict[str, Any]) -> str:
        """Champion = highest mean among sampled variants (stable on ties)."""
        stats = entry["stats"]
        sampled = [v for v in _VARIANT_ORDER if int(stats[v]["n"]) > 0]
        if not sampled:
            return str(entry["champion"])
        current = str(entry["champion"])
        best = max(sampled, key=lambda v: float(stats[v]["mean"]))
        if current in sampled and float(stats[current]["mean"]) >= float(
            stats[best]["mean"]
        ):
            return current  # ties never dethrone the incumbent
        return best

    @staticmethod
    def _choose(entry: dict[str, Any]) -> str:
        stats = entry["stats"]
        # Probe phase: every variant gets _PROBE_ROUNDS samples first
        # (count pending serves so back-to-back calls keep rotating).
        pending = entry.get("last_served")
        unexplored = [
            v
            for v in _VARIANT_ORDER
            if int(stats[v]["n"]) + (1 if pending == v else 0) < _PROBE_ROUNDS
        ]
        if unexplored:
            return unexplored[0]
        champion = str(entry["champion"])
        if int(entry["serves"]) % _CHALLENGER_EVERY == _CHALLENGER_EVERY - 1:
            challengers = [v for v in _VARIANT_ORDER if v != champion]
            if challengers:
                return max(challengers, key=lambda v: float(stats[v]["mean"]))
        return champion
