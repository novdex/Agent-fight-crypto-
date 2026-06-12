"""Tests for Unit 4 — SQLite store & paper trading.

Imports only ``arena.store`` and ``arena.models`` per the module contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from arena.models import Direction, ScoredSignal, Signal
from arena.store import Store, apply_round_to_equity

AS_OF = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_signal(
    agent: str = "alpha",
    symbol: str = "BTC",
    direction: Direction = Direction.LONG,
    confidence: float = 0.8,
    rationale: str = "momentum",
    price: float = 100.0,
) -> Signal:
    return Signal(
        agent=agent,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        rationale=rationale,
        price_at_signal=price,
    )


def make_store(tmp_path: Path) -> Store:
    return Store(str(tmp_path / "arena.db"))


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------


def test_schema_creation_idempotent(tmp_path: Path) -> None:
    path = str(tmp_path / "arena.db")
    store1 = Store(path)
    round_id = store1.create_round(AS_OF, 24.0)
    store1.close()

    store2 = Store(path)  # re-opening must not fail or wipe data
    assert store2.pending_rounds(AS_OF, force=True)[0]["id"] == round_id
    store2.close()


# ----------------------------------------------------------------------
# round lifecycle
# ----------------------------------------------------------------------


def test_round_lifecycle(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    round_id = store.create_round(AS_OF, 24.0)
    signals = [
        make_signal("alpha", "BTC", Direction.LONG, 0.9, "bullish", 100.0),
        make_signal("alpha", "ETH", Direction.SHORT, 0.4, "bearish", 2000.0),
    ]
    store.add_signals(round_id, signals)

    # before horizon: not due
    assert store.pending_rounds(AS_OF + timedelta(hours=1)) == []
    # force returns it anyway
    forced = store.pending_rounds(AS_OF + timedelta(hours=1), force=True)
    assert [p["id"] for p in forced] == [round_id]
    # after horizon: due, with the documented dict shape
    due = store.pending_rounds(AS_OF + timedelta(hours=25))
    assert len(due) == 1
    assert due[0]["id"] == round_id
    assert due[0]["as_of"] == AS_OF
    assert due[0]["horizon_hours"] == 24.0
    # exactly at horizon counts as due (<=)
    assert store.pending_rounds(AS_OF + timedelta(hours=24)) != []

    # signals round-trip every Signal field
    got = store.signals_for_round(round_id)
    assert sorted(got, key=lambda s: s.symbol) == sorted(
        signals, key=lambda s: s.symbol
    )

    scored = [
        ScoredSignal(**s.model_dump(), price_at_eval=p, score=sc)
        for s, p, sc in zip(signals, [110.0, 1900.0], [0.5, 0.3])
    ]
    store.record_scores(round_id, scored, {"alpha": 0.4})

    # evaluated rounds are no longer pending, even with force
    assert store.pending_rounds(AS_OF + timedelta(hours=48)) == []
    assert store.pending_rounds(AS_OF + timedelta(hours=48), force=True) == []
    store.close()


def test_pending_rounds_handles_naive_and_aware(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    naive_as_of = datetime(2026, 6, 1, 12, 0, 0)  # naive, treated as UTC
    round_id = store.create_round(naive_as_of, 2.0)

    aware_now = datetime(2026, 6, 1, 15, 0, 0, tzinfo=timezone.utc)
    assert [p["id"] for p in store.pending_rounds(aware_now)] == [round_id]
    naive_now = datetime(2026, 6, 1, 15, 0, 0)
    assert [p["id"] for p in store.pending_rounds(naive_now)] == [round_id]
    # returned as_of is aware UTC and comparable to aware datetimes
    as_of = store.pending_rounds(aware_now)[0]["as_of"]
    assert as_of.tzinfo is not None
    assert as_of == datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    store.close()


# ----------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------


def test_weights_init_persist_renormalize(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    # unseen names initialize to equal weights
    w = store.get_weights(["alpha", "beta"])
    assert w == {"alpha": 0.5, "beta": 0.5}

    store.set_weights({"alpha": 0.7, "beta": 0.3})
    assert store.get_weights(["alpha", "beta"]) == pytest.approx(
        {"alpha": 0.7, "beta": 0.3}
    )

    # new name joins: gets 1/len(names), full dict renormalized to sum 1
    w3 = store.get_weights(["alpha", "beta", "gamma"])
    total = 0.7 + 0.3 + 1.0 / 3.0
    assert w3["alpha"] == pytest.approx(0.7 / total)
    assert w3["beta"] == pytest.approx(0.3 / total)
    assert w3["gamma"] == pytest.approx((1.0 / 3.0) / total)
    assert sum(w3.values()) == pytest.approx(1.0)

    # consensus is never written to the weights table
    store.set_weights({"consensus": 0.9, "alpha": 0.5})
    board = {row["agent"]: row for row in store.leaderboard()}
    assert board["alpha"]["weight"] == pytest.approx(0.5)
    assert "consensus" not in {
        row["agent"] for row in store.leaderboard() if row["weight"] > 0
    }
    store.close()


def test_get_weights_empty(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.get_weights([]) == {}
    store.close()


# ----------------------------------------------------------------------
# equity
# ----------------------------------------------------------------------


def test_equity_default_and_set_get(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.get_equity("alpha") == 10_000.0
    assert store.get_equity("alpha", start_equity=5_000.0) == 5_000.0
    store.set_equity("alpha", 12_345.6)
    assert store.get_equity("alpha") == pytest.approx(12_345.6)
    store.set_equity("alpha", 9_000.0)  # upsert overwrites
    assert store.get_equity("alpha") == pytest.approx(9_000.0)
    store.close()


# ----------------------------------------------------------------------
# leaderboard & history
# ----------------------------------------------------------------------


def _evaluated_round(
    store: Store, as_of: datetime, scores: dict[str, float]
) -> int:
    round_id = store.create_round(as_of, 24.0)
    signals = [make_signal(agent, "BTC") for agent in scores]
    store.add_signals(round_id, signals)
    scored = [
        ScoredSignal(**s.model_dump(), price_at_eval=110.0, score=scores[s.agent])
        for s in signals
    ]
    store.record_scores(round_id, scored, scores)
    return round_id


def test_leaderboard_shape_wins_and_ordering(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.set_weights({"alpha": 0.3, "beta": 0.7})
    store.set_equity("alpha", 11_000.0)
    store.set_equity("consensus", 10_500.0)

    # round 1: alpha strictly best agent; consensus beats it too
    _evaluated_round(store, AS_OF, {"alpha": 0.5, "beta": 0.2, "consensus": 0.6})
    # round 2: tie between agents -> no winner; consensus below max -> no win
    _evaluated_round(
        store, AS_OF + timedelta(days=1), {"alpha": 0.1, "beta": 0.1, "consensus": 0.05}
    )
    # round 3: beta strictly best; consensus equal to max -> not strict, no win
    _evaluated_round(
        store, AS_OF + timedelta(days=2), {"alpha": 0.0, "beta": 0.4, "consensus": 0.4}
    )

    board = store.leaderboard()
    assert {row["agent"] for row in board} == {"alpha", "beta", "consensus"}
    for row in board:
        assert set(row) == {"agent", "weight", "rounds", "avg_score", "wins", "equity"}

    by_agent = {row["agent"]: row for row in board}
    assert by_agent["alpha"]["wins"] == 1
    assert by_agent["beta"]["wins"] == 1
    assert by_agent["consensus"]["wins"] == 1
    assert by_agent["alpha"]["rounds"] == 3
    assert by_agent["alpha"]["avg_score"] == pytest.approx((0.5 + 0.1 + 0.0) / 3)
    assert by_agent["consensus"]["avg_score"] == pytest.approx((0.6 + 0.05 + 0.4) / 3)
    assert by_agent["alpha"]["equity"] == pytest.approx(11_000.0)
    assert by_agent["beta"]["equity"] == pytest.approx(10_000.0)  # default
    assert by_agent["consensus"]["equity"] == pytest.approx(10_500.0)
    assert by_agent["consensus"]["weight"] == 0.0  # never stored

    # sorted by weight descending
    weights = [row["weight"] for row in board]
    assert weights == sorted(weights, reverse=True)
    assert board[0]["agent"] == "beta"
    store.close()


def test_history_limit_and_order(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    ids = [
        _evaluated_round(store, AS_OF + timedelta(days=i), {"alpha": 0.1 * i, "beta": 0.0})
        for i in range(4)
    ]
    # a still-pending round must not appear in history
    store.create_round(AS_OF + timedelta(days=9), 24.0)

    rows = store.history(limit=2)
    assert {r["round_id"] for r in rows} == set(ids[-2:])  # last 2 rounds only
    assert len(rows) == 4  # 2 rounds x 2 agents
    assert [r["round_id"] for r in rows] == sorted(
        [r["round_id"] for r in rows], reverse=True
    )
    for r in rows:
        assert set(r) == {"round_id", "as_of", "agent", "round_score"}
        assert r["as_of"].tzinfo is not None
    newest = [r for r in rows if r["round_id"] == ids[-1]]
    assert {r["agent"]: r["round_score"] for r in newest} == pytest.approx(
        {"alpha": 0.3, "beta": 0.0}
    )
    store.close()


# ----------------------------------------------------------------------
# paper trading
# ----------------------------------------------------------------------


def _scored(
    agent: str,
    symbol: str,
    direction: Direction,
    price_then: float,
    price_now: float,
) -> ScoredSignal:
    return ScoredSignal(
        agent=agent,
        symbol=symbol,
        direction=direction,
        confidence=0.5,
        rationale="",
        price_at_signal=price_then,
        price_at_eval=price_now,
        score=0.0,
    )


def test_apply_round_to_equity_hand_computed() -> None:
    # equity 10000, stake_fraction 0.5 -> 5000 staked across 2 positions: 2500 each
    scored = [
        _scored("a", "BTC", Direction.LONG, 100.0, 110.0),  # +10% -> +250
        _scored("a", "ETH", Direction.SHORT, 200.0, 190.0),  # -5%, short -> +125
        _scored("a", "SOL", Direction.FLAT, 50.0, 80.0),  # FLAT holds cash
    ]
    new_equity = apply_round_to_equity(10_000.0, scored)
    assert new_equity == pytest.approx(10_375.0)


def test_apply_round_to_equity_losses_and_stake_fraction() -> None:
    scored = [_scored("a", "BTC", Direction.LONG, 100.0, 90.0)]  # -10%
    # full stake_fraction=1.0 -> entire equity in one position
    assert apply_round_to_equity(10_000.0, scored, stake_fraction=1.0) == pytest.approx(
        9_000.0
    )


def test_apply_round_to_equity_all_flat_unchanged() -> None:
    scored = [
        _scored("a", "BTC", Direction.FLAT, 100.0, 130.0),
        _scored("a", "ETH", Direction.FLAT, 200.0, 100.0),
    ]
    assert apply_round_to_equity(10_000.0, scored) == 10_000.0
    assert apply_round_to_equity(10_000.0, []) == 10_000.0


def test_signals_probability_vector_round_trip(tmp_path) -> None:
    store = Store(str(tmp_path / "probs.db"))
    rid = store.create_round(datetime(2026, 6, 12, tzinfo=timezone.utc), 24.0)
    sig = Signal(
        agent="a", symbol="BTC", direction=Direction.LONG, confidence=0.7,
        price_at_signal=100.0, p_long=0.7, p_short=0.2, p_flat=0.1,
    )
    legacy = Signal(
        agent="b", symbol="BTC", direction=Direction.SHORT, confidence=0.4,
        price_at_signal=100.0,
    )
    store.add_signals(rid, [sig, legacy])
    loaded = {s.agent: s for s in store.signals_for_round(rid)}
    assert loaded["a"].p_long == 0.7 and loaded["a"].has_probs
    assert loaded["b"].p_long is None and not loaded["b"].has_probs
    store.close()


def test_evaluated_rounds_count(tmp_path) -> None:
    store = Store(str(tmp_path / "count.db"))
    assert store.evaluated_rounds_count() == 0
    rid = store.create_round(datetime(2026, 6, 12, tzinfo=timezone.utc), 24.0)
    sig = Signal(
        agent="a", symbol="BTC", direction=Direction.LONG, confidence=0.5,
        price_at_signal=100.0,
    )
    store.add_signals(rid, [sig])
    assert store.evaluated_rounds_count() == 0  # still pending
    scored = ScoredSignal(**sig.model_dump(), price_at_eval=101.0, score=0.1)
    store.record_scores(rid, [scored], {"a": 0.1})
    assert store.evaluated_rounds_count() == 1
    store.close()


def test_paper_trading_fees_deducted(tmp_path) -> None:
    # One LONG position, +10% move, 50% stake, 5 bps/side fees:
    # pnl = 5000 * 0.10 - 2 * 0.0005 * 5000 = 500 - 5 = 495
    sig = ScoredSignal(
        agent="a", symbol="BTC", direction=Direction.LONG, confidence=0.9,
        price_at_signal=100.0, price_at_eval=110.0, score=0.5,
    )
    assert apply_round_to_equity(10_000.0, [sig], fee_rate=0.0005) == pytest.approx(10_495.0)
    # fee-free default keeps the original behavior
    assert apply_round_to_equity(10_000.0, [sig]) == pytest.approx(10_500.0)
