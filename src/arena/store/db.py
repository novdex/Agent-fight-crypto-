"""SQLite store for rounds, signals, scores, weights and paper equity.

Uses only the stdlib ``sqlite3`` module. All datetimes are stored as ISO-8601
strings in UTC; naive datetimes are treated as UTC on both write and read so
naive/aware comparisons never raise.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from arena.models import Direction, RoundStatus, ScoredSignal, Signal

_CONSENSUS = "consensus"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of TEXT,
    horizon_hours REAL,
    status TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    round_id INTEGER,
    agent TEXT,
    symbol TEXT,
    direction TEXT,
    confidence REAL,
    rationale TEXT,
    price_at_signal REAL,
    price_at_eval REAL NULL,
    score REAL NULL,
    p_long REAL NULL,
    p_short REAL NULL,
    p_flat REAL NULL
);
CREATE TABLE IF NOT EXISTS round_scores (
    round_id INTEGER,
    agent TEXT,
    score REAL
);
CREATE TABLE IF NOT EXISTS weights (
    agent TEXT PRIMARY KEY,
    weight REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    agent TEXT PRIMARY KEY,
    equity REAL,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_round_agent_symbol
    ON signals (round_id, agent, symbol);
"""


def _to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as an aware UTC datetime (naive values are assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_dt(value: str) -> datetime:
    """Parse a stored ISO string into an aware UTC datetime."""
    return _to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """SQLite-backed persistence for the arena (rounds, signals, weights, equity)."""

    def __init__(self, path: str = "arena.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the original schema to older DBs."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(signals)").fetchall()
        }
        with self._conn:
            for col in ("p_long", "p_short", "p_flat"):
                if col not in existing:
                    self._conn.execute(f"ALTER TABLE signals ADD COLUMN {col} REAL NULL")

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # rounds & signals
    # ------------------------------------------------------------------

    def create_round(self, as_of: datetime, horizon_hours: float) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO rounds (as_of, horizon_hours, status) VALUES (?, ?, ?)",
                (_to_utc(as_of).isoformat(), float(horizon_hours), RoundStatus.PENDING.value),
            )
        round_id = cur.lastrowid
        if round_id is None:  # pragma: no cover - sqlite always sets it on INSERT
            raise RuntimeError("sqlite did not return a row id for the new round")
        return round_id

    def add_signals(self, round_id: int, signals: list[Signal]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO signals "
                "(round_id, agent, symbol, direction, confidence, rationale, "
                " price_at_signal, p_long, p_short, p_flat) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        round_id,
                        s.agent,
                        s.symbol,
                        s.direction.value,
                        s.confidence,
                        s.rationale,
                        s.price_at_signal,
                        s.p_long,
                        s.p_short,
                        s.p_flat,
                    )
                    for s in signals
                ],
            )

    def pending_rounds(self, now: datetime, *, force: bool = False) -> list[dict]:
        """Return PENDING rounds whose ``as_of + horizon_hours <= now``.

        With ``force=True`` every PENDING round is returned regardless of due
        time. Each dict has keys ``id``, ``as_of`` (aware UTC datetime) and
        ``horizon_hours``.
        """
        now_utc = _to_utc(now)
        rows = self._conn.execute(
            "SELECT id, as_of, horizon_hours FROM rounds WHERE status = ? ORDER BY id",
            (RoundStatus.PENDING.value,),
        ).fetchall()
        out: list[dict] = []
        for row in rows:
            as_of = _parse_dt(row["as_of"])
            horizon = float(row["horizon_hours"])
            if force or as_of + timedelta(hours=horizon) <= now_utc:
                out.append({"id": row["id"], "as_of": as_of, "horizon_hours": horizon})
        return out

    def signals_for_round(self, round_id: int) -> list[Signal]:
        rows = self._conn.execute(
            "SELECT agent, symbol, direction, confidence, rationale, price_at_signal, "
            "p_long, p_short, p_flat "
            "FROM signals WHERE round_id = ? ORDER BY rowid",
            (round_id,),
        ).fetchall()
        return [
            Signal(
                agent=row["agent"],
                symbol=row["symbol"],
                direction=Direction(row["direction"]),
                confidence=row["confidence"],
                rationale=row["rationale"],
                price_at_signal=row["price_at_signal"],
                p_long=row["p_long"],
                p_short=row["p_short"],
                p_flat=row["p_flat"],
            )
            for row in rows
        ]

    def evaluated_rounds_count(self) -> int:
        """Number of rounds already EVALUATED (drives the adaptive eta schedule)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM rounds WHERE status = ?",
            (RoundStatus.EVALUATED.value,),
        ).fetchone()
        return int(row["n"])

    def record_scores(
        self,
        round_id: int,
        scored: list[ScoredSignal],
        round_scores: dict[str, float],
    ) -> None:
        """Persist evaluation results and mark the round EVALUATED.

        Writes ``price_at_eval`` and ``score`` onto the matching signals rows,
        inserts one ``round_scores`` row per agent, and flips the round status.
        Signals within a round are keyed by ``(agent, symbol)`` — the agent
        contract guarantees exactly one signal per agent per coin.
        """
        with self._conn:
            self._conn.executemany(
                "UPDATE signals SET price_at_eval = ?, score = ? "
                "WHERE round_id = ? AND agent = ? AND symbol = ?",
                [(s.price_at_eval, s.score, round_id, s.agent, s.symbol) for s in scored],
            )
            self._conn.executemany(
                "INSERT INTO round_scores (round_id, agent, score) VALUES (?, ?, ?)",
                [(round_id, agent, score) for agent, score in round_scores.items()],
            )
            self._conn.execute(
                "UPDATE rounds SET status = ? WHERE id = ?",
                (RoundStatus.EVALUATED.value, round_id),
            )

    # ------------------------------------------------------------------
    # weights ("decision power")
    # ------------------------------------------------------------------

    def get_weights(self, agent_names: list[str]) -> dict[str, float]:
        """Return weights for ``agent_names``, initializing unseen names.

        Stored weights are read from the ``weights`` table. Any name without a
        stored weight is initialized to ``1 / len(agent_names)``, then the full
        returned dict is renormalized to sum to 1 and persisted (the synthetic
        ``"consensus"`` agent is never stored).
        """
        if not agent_names:
            return {}
        placeholders = ",".join("?" for _ in agent_names)
        rows = self._conn.execute(
            f"SELECT agent, weight FROM weights WHERE agent IN ({placeholders})",
            agent_names,
        ).fetchall()
        stored = {row["agent"]: float(row["weight"]) for row in rows}
        weights = {
            name: stored.get(name, 1.0 / len(agent_names)) for name in agent_names
        }
        total = sum(weights.values())
        if total > 0:
            weights = {name: w / total for name, w in weights.items()}
        self.set_weights(weights)
        return weights

    def set_weights(self, weights: dict[str, float]) -> None:
        now = _now_iso()
        with self._conn:
            self._conn.executemany(
                "INSERT INTO weights (agent, weight, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(agent) DO UPDATE SET weight = excluded.weight, "
                "updated_at = excluded.updated_at",
                [
                    (agent, float(w), now)
                    for agent, w in weights.items()
                    if agent != _CONSENSUS
                ],
            )

    # ------------------------------------------------------------------
    # paper portfolios
    # ------------------------------------------------------------------

    def get_equity(self, agent: str, *, start_equity: float = 10_000.0) -> float:
        row = self._conn.execute(
            "SELECT equity FROM equity WHERE agent = ?", (agent,)
        ).fetchone()
        return float(row["equity"]) if row is not None else start_equity

    def set_equity(self, agent: str, equity: float) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO equity (agent, equity, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(agent) DO UPDATE SET equity = excluded.equity, "
                "updated_at = excluded.updated_at",
                (agent, float(equity), _now_iso()),
            )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def leaderboard(self) -> list[dict]:
        """One dict per agent (incl. ``"consensus"``), sorted by weight desc.

        Keys: ``agent``, ``weight``, ``rounds``, ``avg_score``, ``wins``,
        ``equity``.

        Win rule: for each evaluated round the benchmark is the maximum round
        score among *non-consensus* agents. A non-consensus agent "wins" a
        round when its round score is strictly greater than every other
        non-consensus agent's score (a unique maximum; ties produce no
        winner). ``"consensus"`` wins a round when its score is strictly
        greater than that non-consensus maximum.
        """
        score_rows = self._conn.execute(
            "SELECT rs.round_id, rs.agent, rs.score FROM round_scores rs "
            "JOIN rounds r ON r.id = rs.round_id WHERE r.status = ?",
            (RoundStatus.EVALUATED.value,),
        ).fetchall()

        by_round: dict[int, dict[str, float]] = {}
        for row in score_rows:
            by_round.setdefault(row["round_id"], {})[row["agent"]] = float(row["score"])

        agents: set[str] = {row["agent"] for row in score_rows}
        weight_rows = self._conn.execute("SELECT agent, weight FROM weights").fetchall()
        weights = {row["agent"]: float(row["weight"]) for row in weight_rows}
        agents.update(weights)
        equity_rows = self._conn.execute("SELECT agent, equity FROM equity").fetchall()
        equities = {row["agent"]: float(row["equity"]) for row in equity_rows}
        agents.update(equities)

        rounds_count: dict[str, int] = {a: 0 for a in agents}
        score_sum: dict[str, float] = {a: 0.0 for a in agents}
        wins: dict[str, int] = {a: 0 for a in agents}
        for scores in by_round.values():
            non_consensus = [s for a, s in scores.items() if a != _CONSENSUS]
            best = max(non_consensus) if non_consensus else None
            best_is_unique = best is not None and non_consensus.count(best) == 1
            for agent, score in scores.items():
                rounds_count[agent] += 1
                score_sum[agent] += score
                if best is None:
                    continue
                if agent == _CONSENSUS:
                    if score > best:
                        wins[agent] += 1
                elif best_is_unique and score == best:
                    wins[agent] += 1

        board = [
            {
                "agent": agent,
                "weight": weights.get(agent, 0.0),
                "rounds": rounds_count[agent],
                "avg_score": (
                    score_sum[agent] / rounds_count[agent] if rounds_count[agent] else 0.0
                ),
                "wins": wins[agent],
                "equity": equities.get(agent, 10_000.0),
            }
            for agent in agents
        ]
        board.sort(key=lambda d: (-d["weight"], d["agent"]))
        return board

    def history(self, limit: int = 20) -> list[dict]:
        """Per-agent scores for the most recent ``limit`` evaluated rounds.

        Returns one dict per (round, agent) pair, newest round first:
        ``{"round_id", "as_of": aware UTC datetime, "agent", "round_score"}``.
        ``limit`` bounds the number of rounds, not the number of rows.
        """
        rows = self._conn.execute(
            "SELECT rs.round_id, r.as_of, rs.agent, rs.score "
            "FROM round_scores rs JOIN rounds r ON r.id = rs.round_id "
            "WHERE rs.round_id IN ("
            "    SELECT id FROM rounds WHERE status = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY rs.round_id DESC, rs.agent",
            (RoundStatus.EVALUATED.value, limit),
        ).fetchall()
        return [
            {
                "round_id": row["round_id"],
                "as_of": _parse_dt(row["as_of"]),
                "agent": row["agent"],
                "round_score": float(row["score"]),
            }
            for row in rows
        ]
