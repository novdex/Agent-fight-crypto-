"""Round journal: exact snapshots, raw LLM replies, per-agent cost.

The journal is the audit/replay record (improvement #95): every round's
exact market snapshot (for backtest replays and funding-aware evaluation),
every agent's raw reply text, and token/cost accounting.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

from arena.models import MarketSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_rounds (
    round_id INTEGER PRIMARY KEY,
    snapshot_json TEXT,
    config_hash TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS journal_replies (
    round_id INTEGER,
    agent TEXT,
    raw_text TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL
);
"""


class Journal:
    def __init__(self, path: str = "arena.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def record_snapshot(
        self, round_id: int, snapshot: MarketSnapshot, config_hash: str = ""
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO journal_rounds "
                "(round_id, snapshot_json, config_hash, created_at) VALUES (?, ?, ?, ?)",
                (
                    round_id,
                    snapshot.model_dump_json(),
                    config_hash,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def record_reply(
        self,
        round_id: int,
        agent: str,
        raw_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO journal_replies "
                "(round_id, agent, raw_text, input_tokens, output_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (round_id, agent, raw_text, input_tokens, output_tokens, cost_usd),
            )

    def snapshot_for_round(self, round_id: int) -> Optional[MarketSnapshot]:
        row = self._conn.execute(
            "SELECT snapshot_json FROM journal_rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if row is None:
            return None
        return MarketSnapshot.model_validate_json(row["snapshot_json"])

    def snapshots(self) -> list[MarketSnapshot]:
        """All journaled snapshots in round order (replay-backtest input)."""
        rows = self._conn.execute(
            "SELECT snapshot_json FROM journal_rounds ORDER BY round_id"
        ).fetchall()
        return [MarketSnapshot.model_validate_json(r["snapshot_json"]) for r in rows]

    def costs_by_agent(self) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT agent, SUM(cost_usd) AS total FROM journal_replies GROUP BY agent"
        ).fetchall()
        return {row["agent"]: float(row["total"] or 0.0) for row in rows}
