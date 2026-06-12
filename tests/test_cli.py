"""Tests for arena.cli.

The other arena modules (data, agents, engine, store) are built in parallel
and may not exist here, so every test injects deterministic fake modules into
``sys.modules`` *before* invoking ``arena.cli.main``. Only ``arena.cli`` and
``arena.models`` / ``arena.config`` are imported for real.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from arena.models import CoinSnapshot, Direction, MarketSnapshot, Signal

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)

SNAPSHOT = MarketSnapshot(
    as_of=NOW,
    coins=[
        CoinSnapshot(symbol="BTC", name="Bitcoin", price_usd=100_000.0),
        CoinSnapshot(symbol="ETH", name="Ethereum", price_usd=4_000.0),
    ],
)


def _signals_for(agent: str, snapshot: MarketSnapshot) -> list[Signal]:
    return [
        Signal(
            agent=agent,
            symbol=coin.symbol,
            direction=Direction.LONG,
            confidence=0.8,
            rationale=f"{agent} likes {coin.symbol}",
            price_at_signal=coin.price_usd,
        )
        for coin in snapshot.coins
    ]


class Fakes:
    """Mutable holder shared between the injected modules and the test."""

    def __init__(self) -> None:
        self.agent_error: type[Exception] = type("AgentError", (Exception,), {})
        self.agents: list[Any] = []
        self.pending: list[dict[str, Any]] = []
        self.round_signals: dict[int, list[Signal]] = {}
        self.leaderboard_rows: list[dict[str, Any]] = []
        self.history_rows: list[dict[str, Any]] = []
        self.stores: list[Any] = []
        self.update_weights_calls: list[dict[str, Any]] = []
        self.score_calls: list[tuple[Signal, float]] = []
        self.adaptive_eta_calls: list[tuple[int, int]] = []


class FakeAgent:
    def __init__(self, name: str, *, fail: Optional[Exception] = None) -> None:
        self.name = name
        self._fail = fail

    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        if self._fail is not None:
            raise self._fail
        return _signals_for(self.name, snapshot)


def _make_fake_store_cls(fakes: Fakes) -> type:
    class FakeStore:
        def __init__(self, path: str = "arena.db") -> None:
            self.path = path
            self.created: list[tuple[datetime, float]] = []
            self.added_signals: dict[int, list[Signal]] = {}
            self.recorded: list[tuple[int, list[Any], dict[str, float]]] = []
            self.weights_set: list[dict[str, float]] = []
            self.equity: dict[str, float] = {}
            self.closed = False
            fakes.stores.append(self)

        def close(self) -> None:
            self.closed = True

        def create_round(self, as_of: datetime, horizon_hours: float) -> int:
            self.created.append((as_of, horizon_hours))
            return 7

        def add_signals(self, round_id: int, signals: list[Signal]) -> None:
            self.added_signals[round_id] = list(signals)

        def evaluated_rounds_count(self) -> int:
            return len(self.recorded)

        def pending_rounds(self, now: datetime, *, force: bool = False) -> list[dict[str, Any]]:
            assert now.tzinfo is not None  # CLI must pass aware UTC datetimes
            return list(fakes.pending)

        def signals_for_round(self, round_id: int) -> list[Signal]:
            return list(fakes.round_signals.get(round_id, []))

        def record_scores(
            self, round_id: int, scored: list[Any], round_scores: dict[str, float]
        ) -> None:
            self.recorded.append((round_id, scored, round_scores))

        def get_weights(self, agent_names: list[str]) -> dict[str, float]:
            n = max(len(agent_names), 1)
            return {name: 1.0 / n for name in agent_names}

        def set_weights(self, weights: dict[str, float]) -> None:
            self.weights_set.append(dict(weights))

        def get_equity(self, agent: str, *, start_equity: float = 10_000.0) -> float:
            return self.equity.get(agent, start_equity)

        def set_equity(self, agent: str, equity: float) -> None:
            self.equity[agent] = equity

        def leaderboard(self) -> list[dict[str, Any]]:
            return list(fakes.leaderboard_rows)

        def history(self, limit: int = 20) -> list[dict[str, Any]]:
            return list(fakes.history_rows)

    return FakeStore


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Inject fake arena.data / arena.agents / arena.engine / arena.store."""
    holder = Fakes()
    holder.agents = [FakeAgent("claude"), FakeAgent("gpt")]

    # --- arena.data -------------------------------------------------------
    data_mod = types.ModuleType("arena.data")

    def fetch_top_coins(n: int = 20, *, offline: bool = False) -> MarketSnapshot:
        return SNAPSHOT

    def fetch_prices(symbols: list[str], *, offline: bool = False) -> dict[str, float]:
        base = {"BTC": 105_000.0, "ETH": 3_900.0}
        return {s: base.get(s, 1.0) for s in symbols}

    data_mod.fetch_top_coins = fetch_top_coins  # type: ignore[attr-defined]
    data_mod.fetch_prices = fetch_prices  # type: ignore[attr-defined]

    # --- arena.agents (+ .base) -------------------------------------------
    agents_mod = types.ModuleType("arena.agents")
    base_mod = types.ModuleType("arena.agents.base")
    base_mod.AgentError = holder.agent_error  # type: ignore[attr-defined]

    def available_agents(specs: list[Any], *, mock: bool = False) -> list[Any]:
        return list(holder.agents)

    agents_mod.available_agents = available_agents  # type: ignore[attr-defined]
    agents_mod.base = base_mod  # type: ignore[attr-defined]

    # --- arena.engine ------------------------------------------------------
    engine_mod = types.ModuleType("arena.engine")

    def score_signal(sig: Signal, price_now: float, *, flat_threshold_pct: float = 1.0) -> float:
        holder.score_calls.append((sig, price_now))
        return 0.5 if sig.direction is Direction.LONG else -0.25

    def update_weights(
        weights: dict[str, float],
        agent_scores: dict[str, float],
        *,
        eta: float,
        min_weight: float,
        max_weight: float,
    ) -> dict[str, float]:
        holder.update_weights_calls.append(
            {
                "weights": dict(weights),
                "agent_scores": dict(agent_scores),
                "eta": eta,
                "min_weight": min_weight,
                "max_weight": max_weight,
            }
        )
        return {name: 0.123 for name in weights}

    def consensus_signals(
        signals_by_agent: dict[str, list[Signal]], weights: dict[str, float]
    ) -> list[Signal]:
        symbols = {s.symbol: s.price_at_signal for sigs in signals_by_agent.values() for s in sigs}
        return [
            Signal(
                agent="consensus",
                symbol=sym,
                direction=Direction.LONG,
                confidence=0.6,
                rationale="majority vote",
                price_at_signal=price,
            )
            for sym, price in sorted(symbols.items())
        ]

    def adaptive_eta(round_index: int, n_agents: int) -> float:
        holder.adaptive_eta_calls.append((round_index, n_agents))
        return 0.777

    engine_mod.score_signal = score_signal  # type: ignore[attr-defined]
    engine_mod.update_weights = update_weights  # type: ignore[attr-defined]
    engine_mod.consensus_signals = consensus_signals  # type: ignore[attr-defined]
    engine_mod.adaptive_eta = adaptive_eta  # type: ignore[attr-defined]

    # --- arena.store --------------------------------------------------------
    store_mod = types.ModuleType("arena.store")
    store_mod.Store = _make_fake_store_cls(holder)  # type: ignore[attr-defined]

    def apply_round_to_equity(
        equity: float,
        scored: list[Any],
        *,
        stake_fraction: float = 0.5,
        fee_rate: float = 0.0,
    ) -> float:
        return equity + 100.0

    store_mod.apply_round_to_equity = apply_round_to_equity  # type: ignore[attr-defined]

    for name, mod in {
        "arena.data": data_mod,
        "arena.agents": agents_mod,
        "arena.agents.base": base_mod,
        "arena.engine": engine_mod,
        "arena.store": store_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return holder


def _argv(*tail: str, tmp_path: Any) -> list[str]:
    """Global flags pointing at a nonexistent config (-> pure defaults) and a tmp DB."""
    return ["--config", str(tmp_path / "missing.yaml"), "--db", str(tmp_path / "test.db"), *tail]


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def test_parser_accepts_global_flags_and_subcommands() -> None:
    from arena.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(["--config", "c.yaml", "--db", "x.db", "--mock", "--offline", "run-round"])
    assert args.command == "run-round"
    assert args.config == "c.yaml"
    assert args.db == "x.db"
    assert args.mock is True
    assert args.offline is True

    args = parser.parse_args(["evaluate", "--force"])
    assert args.command == "evaluate"
    assert args.force is True
    assert args.config == "config.yaml"  # default
    assert args.db is None
    assert args.mock is False and args.offline is False

    # global flags are also accepted after the subcommand
    args = parser.parse_args(["evaluate", "--offline"])
    assert args.offline is True

    args = parser.parse_args(["leaderboard"])
    assert args.command == "leaderboard"

    args = parser.parse_args(["history"])
    assert args.command == "history"

    args = parser.parse_args(["loop", "--interval-mins", "5"])
    assert args.command == "loop"
    assert args.interval_mins == 5.0

    args = parser.parse_args(["loop"])
    assert args.interval_mins == 60.0


def test_main_without_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    from arena.cli import main

    rc = main([])
    assert rc == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_module_imports_without_parallel_modules() -> None:
    # arena.cli must import lazily: no arena.data/agents/engine/store at module level.
    import importlib

    import arena.cli

    importlib.reload(arena.cli)


# ---------------------------------------------------------------------------
# run-round
# ---------------------------------------------------------------------------


def test_run_round_happy_path(fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    from arena.cli import main

    rc = main(_argv("run-round", "--mock", "--offline", tmp_path=tmp_path))
    assert rc == 0

    assert len(fakes.stores) == 1
    store = fakes.stores[0]
    assert store.path == str(tmp_path / "test.db")
    assert store.created == [(NOW, 24.0)]
    assert store.closed is True

    stored = store.added_signals[7]
    by_agent = {s.agent for s in stored}
    assert by_agent == {"claude", "gpt", "consensus"}
    # 2 agents x 2 coins + 2 consensus signals
    assert len(stored) == 6

    out = capsys.readouterr().out
    assert "consensus" in out
    assert "BTC" in out and "ETH" in out
    assert "claude" in out and "gpt" in out


def test_run_round_agent_error_does_not_kill_round(
    fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena.cli import main

    fakes.agents = [
        FakeAgent("claude"),
        FakeAgent("gpt", fail=fakes.agent_error("rate limited")),
        FakeAgent("gemini"),
    ]
    rc = main(_argv("run-round", tmp_path=tmp_path))
    assert rc == 0

    stored = fakes.stores[0].added_signals[7]
    assert {s.agent for s in stored} == {"claude", "gemini", "consensus"}
    err = capsys.readouterr().err
    assert "gpt" in err and "rate limited" in err


def test_run_round_fewer_than_two_agents_exits_nonzero(
    fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena.cli import main

    fakes.agents = [FakeAgent("claude")]
    rc = main(_argv("run-round", tmp_path=tmp_path))
    assert rc != 0
    assert fakes.stores == []  # nothing persisted
    assert "fewer than 2" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_scores_round_and_updates_weights(
    fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena.cli import main

    fakes.pending = [{"id": 7, "as_of": NOW, "horizon_hours": 24.0}]
    fakes.round_signals = {
        7: (
            _signals_for("claude", SNAPSHOT)
            + _signals_for("gpt", SNAPSHOT)
            + [
                Signal(
                    agent="consensus",
                    symbol="BTC",
                    direction=Direction.SHORT,
                    confidence=0.6,
                    price_at_signal=100_000.0,
                )
            ]
        )
    }

    rc = main(_argv("evaluate", "--force", "--offline", tmp_path=tmp_path))
    assert rc == 0

    store = fakes.stores[0]
    # scores recorded for all signals, incl. consensus
    (round_id, scored, round_scores) = store.recorded[0]
    assert round_id == 7
    assert len(scored) == 5
    assert round_scores["claude"] == pytest.approx(0.5)
    assert round_scores["gpt"] == pytest.approx(0.5)
    assert round_scores["consensus"] == pytest.approx(-0.25)

    # weight update excludes consensus, uses settings' eta/min/max
    assert len(fakes.update_weights_calls) == 1
    call = fakes.update_weights_calls[0]
    assert set(call["agent_scores"]) == {"claude", "gpt"}
    # adaptive eta (default on): cli asked the fake schedule and used its value
    assert fakes.adaptive_eta_calls == [(1, 2)]  # 1st evaluated round, 2 competitors
    assert call["eta"] == pytest.approx(0.777)
    assert call["min_weight"] == pytest.approx(0.05)
    assert call["max_weight"] == pytest.approx(0.60)
    # One weight write; the raw fake update (0.123 each) passes through real
    # significance damping (rounds_seen=1 < threshold -> 50/50 blend with the
    # old equal weights, renormalized), so assert structure not exact values.
    assert len(store.weights_set) == 1
    written = store.weights_set[0]
    assert set(written) == {"claude", "gpt"}
    # With the real weights2 importable the values are significance-damped and
    # renormalized; under the fake (non-package) engine module they pass
    # through raw — either way both fighters carry positive weight.
    assert all(v > 0 for v in written.values())

    # Paper equity recorded for agents AND consensus via the real risk
    # pipeline (Kelly sizing, fees, slippage) — values move off the start.
    assert set(store.equity) == {"claude", "gpt", "consensus"}
    assert all(v > 0 and v != 10_000.0 for v in store.equity.values())

    out = capsys.readouterr().out
    assert "claude" in out and "consensus" in out


def test_evaluate_with_no_due_rounds(
    fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena.cli import main

    fakes.pending = []
    rc = main(_argv("evaluate", tmp_path=tmp_path))
    assert rc == 0
    assert "No rounds due" in capsys.readouterr().out
    assert fakes.update_weights_calls == []


# ---------------------------------------------------------------------------
# leaderboard / history
# ---------------------------------------------------------------------------


def test_leaderboard_renders_store_rows(
    fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena.cli import main

    fakes.leaderboard_rows = [
        {"agent": "claude", "weight": 0.4, "rounds": 3, "avg_score": 0.21, "wins": 2, "equity": 11_500.0},
        {"agent": "grok", "weight": 0.1, "rounds": 3, "avg_score": -0.05, "wins": 0, "equity": 9_400.0},
    ]
    rc = main(_argv("leaderboard", tmp_path=tmp_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "claude" in out and "grok" in out
    assert "40.0%" in out and "10.0%" in out
    assert "11,500.00" in out


def test_history_renders_store_rows(
    fakes: Fakes, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from arena.cli import main

    fakes.history_rows = [
        {"round_id": 7, "as_of": NOW, "agent": "claude", "round_score": 0.31},
        {"round_id": 7, "as_of": NOW, "agent": "gpt", "round_score": -0.10},
    ]
    rc = main(_argv("history", tmp_path=tmp_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "claude" in out and "gpt" in out and "+0.3100" in out


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------


def test_loop_exits_cleanly_on_keyboard_interrupt(
    fakes: Fakes, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import arena.cli
    from arena.cli import main

    def fake_sleep(seconds: float) -> None:
        assert seconds == pytest.approx(120.0)  # 2 minutes
        raise KeyboardInterrupt

    monkeypatch.setattr(arena.cli.time, "sleep", fake_sleep)
    rc = main(_argv("loop", "--interval-mins", "2", tmp_path=tmp_path))
    assert rc == 0
    # one full cycle ran before the interrupt: a round was stored
    assert any(store.created for store in fakes.stores)


def test_pool_signal_samples_averages_probs() -> None:
    from arena.cli import _pool_signal_samples

    def sig(p: tuple[float, float, float]) -> Signal:
        return Signal(
            agent="a", symbol="BTC", direction=Direction.LONG, confidence=max(p),
            price_at_signal=100.0, p_long=p[0], p_short=p[1], p_flat=p[2],
        )

    pooled = _pool_signal_samples([[sig((0.6, 0.3, 0.1))], [sig((0.2, 0.7, 0.1))],
                                   [sig((0.4, 0.5, 0.1))]])
    assert len(pooled) == 1
    out = pooled[0]
    assert out.p_long == pytest.approx(0.4)
    assert out.p_short == pytest.approx(0.5)
    assert out.direction is Direction.SHORT  # argmax of the pooled vector
    assert out.confidence == pytest.approx(0.5)
