"""Offline tests for the per-asset/regime weight engine (U4)."""

from __future__ import annotations

import math

import pytest

from arena.engine.weights2 import (
    bootstrap_skill_pvalue,
    clip_update_factor,
    decorrelation_adjust,
    effective_sample_size,
    regime_key,
    significance_damping,
    update_weight_matrix,
)
from arena.models import Direction
from arena.store.weights_ext import WeightMatrixStore

AGENTS = ["alpha", "beta", "gamma", "delta"]


def equal_matrix(coins: list[str]) -> dict[str, dict[str, float]]:
    return {coin: {a: 1.0 / len(AGENTS) for a in AGENTS} for coin in coins}


# ---------------------------------------------------------------------------
# clip_update_factor
# ---------------------------------------------------------------------------


def test_clip_update_factor_bounds() -> None:
    assert clip_update_factor(5.0, 0.5, 2.0) == 2.0
    assert clip_update_factor(0.1, 0.5, 2.0) == 0.5
    assert clip_update_factor(1.3, 0.5, 2.0) == 1.3
    # inverted bounds are swapped, not fatal
    assert clip_update_factor(5.0, 2.0, 0.5) == 2.0


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------


def test_ess_equal_weights_of_four_is_four() -> None:
    weights = {a: 0.25 for a in AGENTS}
    assert effective_sample_size(weights) == pytest.approx(4.0)


def test_ess_dominant_weight_near_one() -> None:
    weights = {"alpha": 0.97, "beta": 0.01, "gamma": 0.01, "delta": 0.01}
    ess = effective_sample_size(weights)
    assert 1.0 <= ess < 1.2


def test_ess_scale_invariant_and_empty() -> None:
    w = {"a": 0.5, "b": 0.5}
    assert effective_sample_size(w) == pytest.approx(
        effective_sample_size({k: 10 * v for k, v in w.items()})
    )
    assert effective_sample_size({}) == 0.0


# ---------------------------------------------------------------------------
# significance_damping
# ---------------------------------------------------------------------------


def test_significance_damping_blends_50_50_when_young() -> None:
    new = {"alpha": 0.7, "beta": 0.3}
    old = {"alpha": 0.5, "beta": 0.5}
    damped = significance_damping(new, old, rounds_seen=10, threshold=30)
    assert damped["alpha"] == pytest.approx(0.6)
    assert damped["beta"] == pytest.approx(0.4)
    assert sum(damped.values()) == pytest.approx(1.0)


def test_significance_damping_passthrough_when_mature() -> None:
    new = {"alpha": 0.7, "beta": 0.3}
    old = {"alpha": 0.5, "beta": 0.5}
    assert significance_damping(new, old, rounds_seen=30, threshold=30) == new


def test_significance_damping_missing_old_uses_uniform_prior() -> None:
    new = {"alpha": 0.8, "beta": 0.2}
    damped = significance_damping(new, {}, rounds_seen=0, threshold=10)
    # blend with the 1/2 uniform prior: 0.65 / 0.35
    assert damped["alpha"] == pytest.approx(0.65)
    assert damped["beta"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# update_weight_matrix
# ---------------------------------------------------------------------------


def test_matrix_update_moves_winner_only_on_its_coin() -> None:
    matrix = equal_matrix(["BTC", "ETH"])
    scores = {"BTC": {"alpha": 1.0, "beta": -0.5, "gamma": -0.5, "delta": -0.5}}
    out = update_weight_matrix(
        matrix,
        scores,
        eta=0.5,
        min_weight=0.0,
        max_weight=1.0,
        clip_low=0.5,
        clip_high=2.0,
    )
    # winner up on BTC, losers down
    assert out["BTC"]["alpha"] > 0.25
    for loser in ("beta", "gamma", "delta"):
        assert out["BTC"][loser] < 0.25
    assert sum(out["BTC"].values()) == pytest.approx(1.0)
    # ETH (no scores) stays equal
    for a in AGENTS:
        assert out["ETH"][a] == pytest.approx(0.25)
    # input matrix is not mutated
    assert matrix["BTC"]["alpha"] == pytest.approx(0.25)


def test_matrix_update_clip_respected() -> None:
    matrix = equal_matrix(["BTC"])
    # eta * score = 50 -> raw factor e^50, clipped to 2.0; others factor 1
    scores = {"BTC": {"alpha": 50.0}}
    out = update_weight_matrix(
        matrix,
        scores,
        eta=1.0,
        min_weight=0.0,
        max_weight=1.0,
        clip_low=0.5,
        clip_high=2.0,
    )
    # 2 / (2 + 1 + 1 + 1) = 0.4 exactly when the clip holds
    assert out["BTC"]["alpha"] == pytest.approx(0.4)
    # symmetric: huge negative score clips at 0.5 -> 0.5 / 3.5
    out_low = update_weight_matrix(
        matrix,
        {"BTC": {"alpha": -50.0}},
        eta=1.0,
        min_weight=0.0,
        max_weight=1.0,
        clip_low=0.5,
        clip_high=2.0,
    )
    assert out_low["BTC"]["alpha"] == pytest.approx(0.5 / 3.5)


def test_matrix_update_min_max_projection() -> None:
    matrix = equal_matrix(["BTC"])
    scores = {"BTC": {"alpha": 1.0, "beta": -1.0, "gamma": -1.0, "delta": -1.0}}
    out = update_weight_matrix(
        matrix,
        scores,
        eta=2.0,
        min_weight=0.05,
        max_weight=0.60,
        clip_low=0.01,
        clip_high=100.0,
    )
    for w in out["BTC"].values():
        assert 0.05 - 1e-9 <= w <= 0.60 + 1e-9
    assert sum(out["BTC"].values()) == pytest.approx(1.0)
    assert out["BTC"]["alpha"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# regime_key
# ---------------------------------------------------------------------------


def test_regime_key() -> None:
    assert regime_key("bull") == "regime:bull"
    assert regime_key(" Bear ") == "regime:bear"
    assert regime_key("chop") == "regime:chop"
    assert regime_key("sideways-nonsense") == "regime:chop"


# ---------------------------------------------------------------------------
# bootstrap_skill_pvalue
# ---------------------------------------------------------------------------


def test_bootstrap_pvalue_low_for_clearly_positive_scores() -> None:
    scores = [0.3] * 30
    p = bootstrap_skill_pvalue(scores, n_perm=1000, seed=0)
    assert p < 0.01


def test_bootstrap_pvalue_high_for_symmetric_noise() -> None:
    scores = [0.5, -0.5] * 15  # mean exactly 0, perfectly symmetric
    p = bootstrap_skill_pvalue(scores, n_perm=1000, seed=0)
    assert 0.3 < p <= 1.0


def test_bootstrap_pvalue_deterministic_and_safe_on_empty() -> None:
    scores = [0.1, -0.2, 0.3, 0.05, -0.1]
    assert bootstrap_skill_pvalue(scores, seed=7) == bootstrap_skill_pvalue(
        scores, seed=7
    )
    assert bootstrap_skill_pvalue([]) == 1.0


# ---------------------------------------------------------------------------
# decorrelation_adjust
# ---------------------------------------------------------------------------


def _clone_history() -> dict[str, list[tuple[str, Direction]]]:
    symbols = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT"]
    longs = [(s, Direction.LONG) for s in symbols]
    # gamma dissents but is NOT anti-correlated: agreement values vs the
    # clones are [0, 0, -1, 0, +1, 0] -> correlation 0.
    dissent = [
        ("BTC", Direction.FLAT),
        ("ETH", Direction.FLAT),
        ("SOL", Direction.SHORT),
        ("ADA", Direction.FLAT),
        ("XRP", Direction.LONG),
        ("DOT", Direction.FLAT),
    ]
    return {"alpha": list(longs), "beta": list(longs), "gamma": dissent}


def test_decorrelation_penalizes_clones_and_rewards_dissent() -> None:
    weights = {"alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3}
    history = _clone_history()
    adjusted = decorrelation_adjust(weights, history)
    assert sum(adjusted.values()) == pytest.approx(1.0)
    # one of the clones (the "weaker"; equal weights -> name tiebreak) pays
    assert adjusted["alpha"] < 1 / 3
    assert adjusted["alpha"] < adjusted["beta"]
    # the principled dissenter ends above both clones
    assert adjusted["gamma"] > adjusted["alpha"]
    assert adjusted["gamma"] > adjusted["beta"]


def test_decorrelation_haircuts_the_weaker_clone() -> None:
    weights = {"alpha": 0.5, "beta": 0.2, "gamma": 0.3}
    adjusted = decorrelation_adjust(weights, _clone_history())
    # beta is the weaker clone: it pays, alpha keeps (relative) rank
    assert adjusted["beta"] / weights["beta"] < adjusted["alpha"] / weights["alpha"]


def test_decorrelation_noop_without_overlap_or_correlation() -> None:
    weights = {"alpha": 0.6, "beta": 0.4}
    # too little overlap (fewer than 3 shared predictions)
    history = {
        "alpha": [("BTC", Direction.LONG)],
        "beta": [("BTC", Direction.LONG)],
    }
    adjusted = decorrelation_adjust(weights, history, diversity_bonus=0.0)
    assert adjusted["alpha"] == pytest.approx(0.6)
    assert adjusted["beta"] == pytest.approx(0.4)
    # empty history is also a no-op
    adjusted = decorrelation_adjust(weights, {}, diversity_bonus=0.0)
    assert adjusted == pytest.approx({"alpha": 0.6, "beta": 0.4})


# ---------------------------------------------------------------------------
# WeightMatrixStore
# ---------------------------------------------------------------------------


def test_store_initializes_equal_weights_per_coin(tmp_path) -> None:
    store = WeightMatrixStore(str(tmp_path / "arena.db"))
    try:
        matrix = store.get("global", ["BTC", "ETH"], AGENTS)
        assert set(matrix) == {"BTC", "ETH"}
        for coin in ("BTC", "ETH"):
            for a in AGENTS:
                assert matrix[coin][a] == pytest.approx(0.25)
            assert sum(matrix[coin].values()) == pytest.approx(1.0)
    finally:
        store.close()


def test_store_round_trip_and_persistence(tmp_path) -> None:
    path = str(tmp_path / "arena.db")
    store = WeightMatrixStore(path)
    store.set("regime:bull", {"BTC": {"alpha": 0.7, "beta": 0.3}})
    store.close()

    reopened = WeightMatrixStore(path)
    try:
        matrix = reopened.get("regime:bull", ["BTC", "ETH"], ["alpha", "beta"])
        assert matrix["BTC"]["alpha"] == pytest.approx(0.7)
        assert matrix["BTC"]["beta"] == pytest.approx(0.3)
        # ETH was missing -> initialized to equal weights and persisted
        assert matrix["ETH"]["alpha"] == pytest.approx(0.5)
        assert matrix["ETH"]["beta"] == pytest.approx(0.5)
    finally:
        reopened.close()


def test_store_get_renormalizes_and_books_are_isolated(tmp_path) -> None:
    path = str(tmp_path / "arena.db")
    store = WeightMatrixStore(path)
    try:
        store.set("global", {"BTC": {"alpha": 2.0, "beta": 2.0}})
        matrix = store.get("global", ["BTC"], ["alpha", "beta"])
        assert matrix["BTC"]["alpha"] == pytest.approx(0.5)
        assert matrix["BTC"]["beta"] == pytest.approx(0.5)
        # a different book is untouched by "global" writes
        other = store.get("regime:bear", ["BTC"], ["alpha", "beta"])
        assert other["BTC"]["alpha"] == pytest.approx(0.5)
    finally:
        store.close()


def test_store_get_with_no_agents(tmp_path) -> None:
    store = WeightMatrixStore(str(tmp_path / "arena.db"))
    try:
        assert store.get("global", ["BTC"], []) == {"BTC": {}}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# integration: repeated matrix updates through the store
# ---------------------------------------------------------------------------


def test_matrix_updates_through_store_converge_to_per_coin_winner(tmp_path) -> None:
    store = WeightMatrixStore(str(tmp_path / "arena.db"))
    try:
        matrix = store.get("global", ["BTC", "ETH"], AGENTS)
        scores = {
            "BTC": {"alpha": 0.8, "beta": -0.2, "gamma": -0.2, "delta": -0.2},
            "ETH": {"alpha": -0.2, "beta": 0.8, "gamma": -0.2, "delta": -0.2},
        }
        for _ in range(5):
            matrix = update_weight_matrix(
                matrix,
                scores,
                eta=0.35,
                min_weight=0.05,
                max_weight=0.60,
                clip_low=0.5,
                clip_high=2.0,
            )
        store.set("global", matrix)
        reloaded = store.get("global", ["BTC", "ETH"], AGENTS)
        assert max(reloaded["BTC"], key=reloaded["BTC"].get) == "alpha"
        assert max(reloaded["ETH"], key=reloaded["ETH"].get) == "beta"
        for coin in ("BTC", "ETH"):
            assert sum(reloaded[coin].values()) == pytest.approx(1.0)
            for w in reloaded[coin].values():
                assert 0.05 - 1e-9 <= w <= 0.60 + 1e-9
    finally:
        store.close()
