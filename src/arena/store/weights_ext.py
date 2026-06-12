"""SQLite persistence for per-asset / per-regime weight books.

Owns exactly one namespaced table::

    pa_weights(book TEXT, coin TEXT, agent TEXT, weight REAL,
               PRIMARY KEY(book, coin, agent))

A "book" is a named weight matrix — e.g. ``"global"``, ``"regime:bull"``,
``"regime:bear"``, ``"regime:chop"`` (see ``arena.engine.weights2.regime_key``).
Stdlib only; shares the arena DB file without touching any other table.
"""

from __future__ import annotations

import sqlite3

__all__ = ["WeightMatrixStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pa_weights (
    book TEXT,
    coin TEXT,
    agent TEXT,
    weight REAL,
    PRIMARY KEY (book, coin, agent)
);
"""


class WeightMatrixStore:
    """Load/save ``{coin: {agent: weight}}`` matrices, keyed by book name."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def get(
        self, book: str, coins: list[str], agents: list[str]
    ) -> dict[str, dict[str, float]]:
        """Return the book's matrix for ``coins`` x ``agents``.

        Missing ``(coin, agent)`` cells are initialized to the equal weight
        ``1 / len(agents)``, then every coin row is renormalized to sum to 1
        (degenerate non-positive rows reset to equal weights). The resulting
        matrix is persisted so initialization happens exactly once.
        """
        if not agents:
            return {coin: {} for coin in coins}
        rows = self._conn.execute(
            "SELECT coin, agent, weight FROM pa_weights WHERE book = ?", (book,)
        ).fetchall()
        stored = {(r["coin"], r["agent"]): float(r["weight"]) for r in rows}

        share = 1.0 / len(agents)
        matrix: dict[str, dict[str, float]] = {}
        for coin in coins:
            row_w = {agent: stored.get((coin, agent), share) for agent in agents}
            total = sum(row_w.values())
            if total > 0.0:
                row_w = {agent: w / total for agent, w in row_w.items()}
            else:
                row_w = {agent: share for agent in agents}
            matrix[coin] = row_w
        self.set(book, matrix)
        return matrix

    def set(self, book: str, matrix: dict[str, dict[str, float]]) -> None:
        """Upsert every ``(coin, agent)`` cell of ``matrix`` into ``book``."""
        with self._conn:
            self._conn.executemany(
                "INSERT INTO pa_weights (book, coin, agent, weight) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(book, coin, agent) DO UPDATE SET "
                "weight = excluded.weight",
                [
                    (book, coin, agent, float(w))
                    for coin, row in matrix.items()
                    for agent, w in row.items()
                ],
            )

    def close(self) -> None:
        self._conn.close()
