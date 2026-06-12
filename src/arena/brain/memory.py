"""Per-agent layered JSON-file memory: recent reflections + distilled lessons.

Layout: one ``<root>/<agent>.json`` file per agent holding a bounded list of
round records (short-term layer) and a small set of distilled long-term
notes recomputed on every write. All writes are atomic (tmp file + rename).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from arena.models import ScoredSignal

_SHORT_ROUNDS = 5  # recent reflections surfaced to prompts
_PATTERN_ROUNDS = 50  # rounds kept on disk for long-term distillation
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to `path` atomically: temp file in the same dir + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _safe_filename(agent: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", agent.strip()) or "agent"
    return f"{cleaned}.json"


class AgentMemory:
    """JSON-file layered memory for one agent under a shared root dir."""

    def __init__(self, root: str, agent: str):
        self.agent = agent
        self.path = Path(root) / _safe_filename(agent)

    # -- persistence -------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        state.setdefault("agent", self.agent)
        rounds = state.get("rounds")
        state["rounds"] = rounds if isinstance(rounds, list) else []
        notes = state.get("long_term")
        state["long_term"] = notes if isinstance(notes, list) else []
        return state

    def _save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.path, state)

    # -- public API ----------------------------------------------------------

    def lessons(self, max_items: int = 8) -> list[str]:
        """Distilled long-term notes + recent reflections, for prompt injection."""
        state = self._load()
        items: list[str] = [str(n) for n in state["long_term"]]
        for rec in state["rounds"][-_SHORT_ROUNDS:]:
            text = str(rec.get("reflection", "")).strip()
            if text:
                items.append(text)
        seen: set[str] = set()
        unique = [i for i in items if not (i in seen or seen.add(i))]
        return unique[:max_items] if max_items > 0 else []

    def record_round(self, round_id: int, round_score: float, reflection: str) -> None:
        """Append one round record, re-distill long-term notes, write atomically."""
        state = self._load()
        state["rounds"] = [
            r for r in state["rounds"] if r.get("round_id") != round_id
        ]
        state["rounds"].append(
            {
                "round_id": round_id,
                "round_score": round_score,
                "reflection": reflection,
            }
        )
        state["rounds"] = state["rounds"][-_PATTERN_ROUNDS:]
        state["long_term"] = self._distill(state["rounds"])
        self._save(state)

    def reflect(self, round_score: float, scored: list[ScoredSignal]) -> str:
        """Deterministic template reflection: what worked / failed / regime."""
        parts = [f"Round score {round_score:+.3f}."]
        wins = sorted((s for s in scored if s.score > 0), key=lambda s: -s.score)
        losses = sorted((s for s in scored if s.score < 0), key=lambda s: s.score)
        if wins:
            parts.append(
                "Worked: "
                + ", ".join(
                    f"{s.symbol} {s.direction.value} ({s.score:+.2f})" for s in wins[:3]
                )
                + "."
            )
        if losses:
            parts.append(
                "Failed: "
                + ", ".join(
                    f"{s.symbol} {s.direction.value} ({s.score:+.2f})"
                    for s in losses[:3]
                )
                + "."
            )
        if not wins and not losses:
            parts.append("No scoring calls this round.")
        parts.append(self._regime_note(scored))
        return " ".join(parts)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _regime_note(scored: list[ScoredSignal]) -> str:
        moves = [
            (s.price_at_eval - s.price_at_signal) / s.price_at_signal * 100.0
            for s in scored
            if s.price_at_signal > 0
        ]
        if not moves:
            return "Regime: unknown (no price data)."
        mean_move = sum(moves) / len(moves)
        if mean_move > 1.0:
            return f"Regime: risk-on drift (mean move {mean_move:+.2f}%)."
        if mean_move < -1.0:
            return f"Regime: risk-off drift (mean move {mean_move:+.2f}%)."
        return f"Regime: choppy/flat tape (mean move {mean_move:+.2f}%); FLAT was cheap."

    @staticmethod
    def _distill(rounds: list[dict[str, Any]]) -> list[str]:
        """Recompute long-term pattern notes from the stored round records."""
        scores = [float(r.get("round_score", 0.0)) for r in rounds]
        if not scores:
            return []
        mean = sum(scores) / len(scores)
        notes = [f"Lifetime: {len(scores)} rounds, mean score {mean:+.3f}."]
        recent = scores[-3:]
        if len(recent) == 3 and all(s < 0 for s in recent):
            notes.append(
                "Last 3 rounds negative — cut confidence and prefer FLAT in unclear setups."
            )
        elif len(recent) == 3 and all(s > 0 for s in recent):
            notes.append("Last 3 rounds positive — current approach is working; stay consistent.")
        return notes
