"""Offline tests for arena.backtest (metrics, replay, report)."""

from __future__ import annotations

import math

import pytest

from arena.agents.mock import MockAgent
from arena.backtest import (
    ReplayBacktester,
    calmar,
    deflated_sharpe,
    hit_rate,
    max_drawdown_pct,
    profit_factor,
    regime_split,
    round_report_markdown,
    sharpe,
    sortino,
    synthetic_snapshots,
)
from arena.models import AgentSpec, Direction, ScoredSignal


def _mock_agents(n: int = 3) -> list[MockAgent]:
    return [
        MockAgent(AgentSpec(name=f"agent{i}", provider="mock", seed=i + 1))
        for i in range(n)
    ]


# ---------------------------------------------------------------- metrics


def test_sharpe_sign_and_undefined() -> None:
    up = [0.01, 0.02, 0.015, 0.005, 0.02]
    down = [-r for r in up]
    assert sharpe(up) is not None and sharpe(up) > 0
    assert sharpe(down) < 0
    assert sharpe([]) is None
    assert sharpe([0.01]) is None  # < 2 observations
    assert sharpe([0.01, 0.01, 0.01]) is None  # zero std


def test_sortino_sign_and_undefined() -> None:
    mixed = [0.02, -0.01, 0.03, -0.02, 0.01]
    assert sortino(mixed) > 0
    assert sortino([0.01]) is None
    assert sortino([0.01, 0.02]) is None  # no downside -> undefined


def test_max_drawdown_exact() -> None:
    # Peak 120 -> trough 90: exactly 25%.
    assert max_drawdown_pct([100.0, 120.0, 90.0, 110.0]) == 25.0
    assert max_drawdown_pct([100.0, 110.0, 120.0]) == 0.0
    assert max_drawdown_pct([]) == 0.0


def test_calmar_defined_and_undefined() -> None:
    curve = [100.0, 120.0, 90.0, 130.0]
    c = calmar(curve)
    assert c is not None and c > 0
    assert calmar([100.0]) is None
    assert calmar([100.0, 110.0, 120.0]) is None  # zero drawdown


def test_hit_rate() -> None:
    assert hit_rate([1.0, -1.0, 0.5, 0.0]) == 0.5
    assert hit_rate([]) is None


def test_profit_factor_exact() -> None:
    # gains 0.05, losses 0.05 -> exactly 1.0
    assert profit_factor([0.02, -0.01, 0.03, -0.04]) == 1.0
    assert profit_factor([0.04, -0.02]) == 2.0
    assert profit_factor([0.01, 0.02]) is None  # no losses
    assert profit_factor([]) is None


def test_deflated_sharpe_below_raw() -> None:
    raw = 1.5
    deflated = deflated_sharpe(raw, n_obs=50, n_trials=4)
    assert deflated is not None
    assert 0 < deflated < raw
    # Exact simplified-Bailey factor.
    assert math.isclose(deflated, raw * math.sqrt(1 - 4 / 49))
    # Heavy multiple-testing relative to obs floors at 0.
    assert deflated_sharpe(raw, n_obs=3, n_trials=10) == 0.0
    assert deflated_sharpe(raw, n_obs=1) is None


def test_regime_split() -> None:
    rounds = [
        {"regime": "bull", "agent": "a", "round_score": 0.2},
        {"regime": "bull", "agent": "a", "round_score": 0.4},
        {"regime": "bear", "agent": "a", "round_score": -0.1},
        {"regime": "bull", "agent": "b", "round_score": 0.1},
    ]
    split = regime_split(rounds)
    assert math.isclose(split["bull"]["a"]["mean_score"], 0.3)
    assert split["bull"]["a"]["count"] == 2
    assert split["bear"]["a"] == {"mean_score": -0.1, "count": 1}
    assert split["bull"]["b"]["count"] == 1


# ----------------------------------------------------------------- replay


def test_synthetic_snapshots_deterministic() -> None:
    a = synthetic_snapshots(6, 3, seed=7)
    b = synthetic_snapshots(6, 3, seed=7)
    assert len(a) == 6
    assert a[0].symbols == ["C0", "C1", "C2"]
    assert [c.price_usd for s in a for c in s.coins] == [
        c.price_usd for s in b for c in s.coins
    ]
    # Different seeds diverge.
    c = synthetic_snapshots(6, 3, seed=8)
    assert [x.price_usd for x in a[0].coins] != [x.price_usd for x in c[0].coins]
    # Prices actually move and changes are populated.
    assert a[0].coins[0].price_usd != a[5].coins[0].price_usd
    assert a[1].coins[0].change_1h_pct is not None


def test_replay_deterministic_and_shapes() -> None:
    snaps = synthetic_snapshots(12, 4, seed=1)
    result1 = ReplayBacktester(snaps, _mock_agents()).run()
    result2 = ReplayBacktester(synthetic_snapshots(12, 4, seed=1), _mock_agents()).run()

    assert result1["leaderboard"] == result2["leaderboard"]
    assert {"leaderboard", "equity_curves", "regret", "benchmark"} <= set(result1)

    # 12 snapshots -> 11 walk-forward rounds.
    for curve in result1["equity_curves"].values():
        assert len(curve) == len(snaps) - 1

    rows = result1["leaderboard"]
    assert len(rows) == 3
    assert {r["agent"] for r in rows} == {"agent0", "agent1", "agent2"}
    for row in rows:
        assert set(row) == {"agent", "weight", "avg_score", "equity", "sharpe"}
        assert -1.0 <= row["avg_score"] <= 1.0
        assert row["equity"] > 0
    assert math.isclose(sum(r["weight"] for r in rows), 1.0)
    # Leaderboard is sorted by weight descending.
    assert [r["weight"] for r in rows] == sorted(
        (r["weight"] for r in rows), reverse=True
    )


def test_replay_regret_of_best_agent_is_zero() -> None:
    snaps = synthetic_snapshots(12, 4, seed=1)
    result = ReplayBacktester(snaps, _mock_agents()).run()
    regret = result["regret"]
    assert min(regret.values()) == 0.0
    assert all(v >= 0.0 for v in regret.values())
    # The zero-regret agent is the one with the highest avg_score.
    best = max(result["leaderboard"], key=lambda r: r["avg_score"])
    assert regret[best["agent"]] == 0.0


def test_replay_fixed_eta_runs() -> None:
    snaps = synthetic_snapshots(5, 2, seed=3)
    result = ReplayBacktester(snaps, _mock_agents(2)).run(
        eta_adaptive=False, fee_rate=0.0
    )
    assert len(result["leaderboard"]) == 2
    assert all(len(c) == 4 for c in result["equity_curves"].values())


# ----------------------------------------------------------------- report


def test_round_report_markdown_contents() -> None:
    scored = [
        ScoredSignal(
            agent="alpha",
            symbol="BTC",
            direction=Direction.LONG,
            confidence=0.8,
            price_at_signal=100.0,
            price_at_eval=105.0,
            score=0.55,
        ),
        ScoredSignal(
            agent="beta",
            symbol="ETH",
            direction=Direction.SHORT,
            confidence=0.6,
            price_at_signal=50.0,
            price_at_eval=49.0,
            score=0.21,
        ),
    ]
    md = round_report_markdown(
        7,
        scored,
        {"alpha": 0.55, "beta": 0.21},
        {"alpha": 0.5, "beta": 0.5},
        {"alpha": 0.58, "beta": 0.42},
    )
    assert "## Round 7" in md
    for name in ("alpha", "beta"):
        assert name in md
    assert "BTC" in md and "ETH" in md
    assert "0.5000" in md  # weight before
    assert "0.5800" in md and "0.4200" in md  # weights after
    assert "LONG" in md and "SHORT" in md


def test_round_report_empty_round() -> None:
    md = round_report_markdown(1, [], {}, {"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5})
    assert "No signals" in md
    assert "| a |" in md and "| b |" in md


def test_replay_benchmark_and_tuner() -> None:
    from arena.agents.mock import MockAgent
    from arena.backtest.replay import ReplayBacktester, synthetic_snapshots
    from arena.backtest.tune import tune_parameters
    from arena.models import AgentSpec

    snaps = synthetic_snapshots(10, 3, seed=5)
    agents = [MockAgent(AgentSpec(name=f"m{i}", provider="mock", seed=i)) for i in range(3)]
    result = ReplayBacktester(snaps, agents).run()
    bench = result["benchmark"]
    assert bench and bench["symbol"] == snaps[0].coins[0].symbol
    assert len(bench["equity_curve"]) == len(snaps) - 1
    # buy & hold equity must track the price ratio exactly
    sym = bench["symbol"]
    ratio = snaps[-1].coin(sym).price_usd / snaps[0].coin(sym).price_usd
    assert bench["equity"] == pytest.approx(10_000.0 * ratio)
    assert result["survivorship_safe"] is True

    tuned = tune_parameters(snaps, agents, etas=(0.2, 0.5), max_weights=(0.6,))
    assert tuned["best"] in tuned["trials"]
    assert len(tuned["trials"]) == 2
    # determinism
    tuned2 = tune_parameters(snaps, agents, etas=(0.2, 0.5), max_weights=(0.6,))
    assert tuned["trials"] == tuned2["trials"]
