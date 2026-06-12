"""Tests for arena.engine (scoring, weights, consensus).

Imports only arena.engine and arena.models, per docs/INTERFACES.md.
"""

from __future__ import annotations

import math

import pytest

from arena.engine import (
    consensus_signals,
    equal_weights,
    realized_return_pct,
    score_signal,
    update_weights,
)
from arena.models import Direction, Signal


def make_signal(
    agent: str = "a",
    symbol: str = "BTC",
    direction: Direction = Direction.LONG,
    confidence: float = 1.0,
    price: float = 100.0,
) -> Signal:
    return Signal(
        agent=agent,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        price_at_signal=price,
    )


# ---------------------------------------------------------------------------
# realized_return_pct
# ---------------------------------------------------------------------------


class TestRealizedReturnPct:
    def test_up_move(self) -> None:
        assert realized_return_pct(100.0, 105.0) == 5.0

    def test_down_move(self) -> None:
        assert realized_return_pct(100.0, 90.0) == -10.0

    def test_no_move(self) -> None:
        assert realized_return_pct(123.45, 123.45) == 0.0


# ---------------------------------------------------------------------------
# score_signal
# ---------------------------------------------------------------------------


class TestScoreSignal:
    def test_correct_long_on_up_move_positive(self) -> None:
        sig = make_signal(direction=Direction.LONG, confidence=0.8, price=100.0)
        assert score_signal(sig, 105.0) > 0.0

    def test_wrong_long_on_down_move_negative(self) -> None:
        sig = make_signal(direction=Direction.LONG, confidence=0.8, price=100.0)
        assert score_signal(sig, 95.0) < 0.0

    def test_correct_short_on_down_move_positive(self) -> None:
        sig = make_signal(direction=Direction.SHORT, confidence=0.7, price=100.0)
        assert score_signal(sig, 92.0) > 0.0

    def test_wrong_short_on_up_move_negative(self) -> None:
        sig = make_signal(direction=Direction.SHORT, confidence=0.7, price=100.0)
        assert score_signal(sig, 108.0) < 0.0

    def test_flat_rewarded_on_small_move(self) -> None:
        # |r| = 0.5% below the 1% flat threshold -> positive score.
        sig = make_signal(direction=Direction.FLAT, confidence=0.9, price=100.0)
        assert score_signal(sig, 100.5) > 0.0

    def test_flat_penalized_on_big_move(self) -> None:
        # |r| = 10% way above the 1% threshold -> negative score.
        sig = make_signal(direction=Direction.FLAT, confidence=0.9, price=100.0)
        assert score_signal(sig, 110.0) < 0.0
        # Big move down penalizes FLAT just the same.
        assert score_signal(sig, 90.0) < 0.0

    def test_flat_threshold_is_configurable(self) -> None:
        sig = make_signal(direction=Direction.FLAT, confidence=1.0, price=100.0)
        # 3% move: penalized with default 1% threshold, rewarded with 5%.
        assert score_signal(sig, 103.0) < 0.0
        assert score_signal(sig, 103.0, flat_threshold_pct=5.0) > 0.0

    def test_exact_formula(self) -> None:
        sig = make_signal(direction=Direction.LONG, confidence=0.8, price=100.0)
        expected = 0.8 * math.tanh(5.0 / 5.0)
        assert math.isclose(score_signal(sig, 105.0), expected, rel_tol=1e-12)

        sig_short = make_signal(direction=Direction.SHORT, confidence=0.5, price=200.0)
        expected_short = 0.5 * math.tanh(-(-3.0) / 5.0)  # r = -3%, raw = +3
        assert math.isclose(
            score_signal(sig_short, 194.0), expected_short, rel_tol=1e-12
        )

        sig_flat = make_signal(direction=Direction.FLAT, confidence=0.6, price=100.0)
        expected_flat = 0.6 * math.tanh((1.0 - 2.0) / 5.0)
        assert math.isclose(score_signal(sig_flat, 102.0), expected_flat, rel_tol=1e-12)

    def test_score_bounded_even_on_huge_moves(self) -> None:
        for direction in (Direction.LONG, Direction.SHORT, Direction.FLAT):
            sig = make_signal(direction=direction, confidence=1.0, price=100.0)
            for price_now in (1.0, 50.0, 100.0, 1_000.0, 1_000_000.0):
                score = score_signal(sig, price_now)
                assert -1.0 <= score <= 1.0

    def test_zero_confidence_scores_zero(self) -> None:
        sig = make_signal(direction=Direction.LONG, confidence=0.0, price=100.0)
        assert score_signal(sig, 200.0) == 0.0

    def test_score_scales_with_confidence(self) -> None:
        lo = make_signal(direction=Direction.LONG, confidence=0.2, price=100.0)
        hi = make_signal(direction=Direction.LONG, confidence=0.9, price=100.0)
        assert score_signal(hi, 110.0) > score_signal(lo, 110.0) > 0.0

    def test_monotonic_in_move_for_long(self) -> None:
        sig = make_signal(direction=Direction.LONG, confidence=1.0, price=100.0)
        scores = [score_signal(sig, p) for p in (95.0, 99.0, 101.0, 105.0, 120.0)]
        assert scores == sorted(scores)
        assert scores[0] < 0.0 < scores[-1]

    def test_monotonic_decreasing_in_move_for_short(self) -> None:
        sig = make_signal(direction=Direction.SHORT, confidence=1.0, price=100.0)
        scores = [score_signal(sig, p) for p in (95.0, 99.0, 101.0, 105.0, 120.0)]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# equal_weights / update_weights
# ---------------------------------------------------------------------------


class TestEqualWeights:
    def test_split_evenly(self) -> None:
        w = equal_weights(["a", "b", "c", "d"])
        assert w == {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

    def test_single_name(self) -> None:
        assert equal_weights(["solo"]) == {"solo": 1.0}

    def test_empty(self) -> None:
        assert equal_weights([]) == {}


class TestUpdateWeights:
    def test_winner_gains_loser_loses(self) -> None:
        start = equal_weights(["a", "b", "c"])
        out = update_weights(
            start,
            {"a": 0.8, "b": -0.8, "c": 0.0},
            eta=0.35,
            min_weight=0.05,
            max_weight=0.6,
        )
        assert out["a"] > start["a"]
        assert out["b"] < start["b"]
        assert out["a"] > out["c"] > out["b"]

    def test_sum_is_one(self) -> None:
        out = update_weights(
            {"a": 0.5, "b": 0.3, "c": 0.2},
            {"a": 1.0, "b": -1.0, "c": 0.3},
            eta=0.35,
            min_weight=0.05,
            max_weight=0.6,
        )
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_min_clamp_respected_before_final_renormalize(self) -> None:
        # Hammer one agent with repeated losses; min_weight keeps it alive.
        w = equal_weights(["a", "b"])
        for _ in range(50):
            w = update_weights(
                w, {"a": 1.0, "b": -1.0}, eta=0.5, min_weight=0.05, max_weight=0.95
            )
        assert abs(sum(w.values()) - 1.0) < 1e-9
        # After clamping to [0.05, 0.95] the renormalized floor stays >= 0.05.
        assert w["b"] >= 0.05 - 1e-9
        assert w["a"] <= 0.95 + 1e-9

    def test_max_clamp_respected(self) -> None:
        w = {"a": 0.5, "b": 0.25, "c": 0.25}
        for _ in range(50):
            w = update_weights(
                w,
                {"a": 1.0, "b": -1.0, "c": -1.0},
                eta=0.5,
                min_weight=0.05,
                max_weight=0.6,
            )
        # Clamp at 0.6, floors at 0.05 -> renormalized sum 0.7 -> max ~0.857.
        # The key invariant: "a" never absorbs everything.
        assert w["a"] < 0.9
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_missing_score_treated_as_zero(self) -> None:
        start = equal_weights(["a", "b", "c"])
        out = update_weights(
            start, {"a": 0.5}, eta=0.35, min_weight=0.0, max_weight=1.0
        )
        # b and c had no score: their raw weights pass through the exp step
        # unchanged, so they remain equal to each other and below a.
        assert out["b"] == out["c"]
        assert out["a"] > out["b"]
        assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_all_missing_scores_is_noop_after_normalization(self) -> None:
        start = {"a": 0.7, "b": 0.3}
        out = update_weights(start, {}, eta=0.35, min_weight=0.0, max_weight=1.0)
        assert math.isclose(out["a"], 0.7, rel_tol=1e-12)
        assert math.isclose(out["b"], 0.3, rel_tol=1e-12)

    def test_single_agent(self) -> None:
        out = update_weights(
            {"only": 1.0}, {"only": -1.0}, eta=0.35, min_weight=0.05, max_weight=0.6
        )
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert list(out) == ["only"]

    def test_empty_weights(self) -> None:
        assert (
            update_weights({}, {"a": 1.0}, eta=0.35, min_weight=0.05, max_weight=0.6)
            == {}
        )

    def test_zero_sum_weights_guarded(self) -> None:
        out = update_weights(
            {"a": 0.0, "b": 0.0}, {"a": 1.0}, eta=0.35, min_weight=0.05, max_weight=0.6
        )
        assert abs(sum(out.values()) - 1.0) < 1e-9
        assert all(v > 0.0 for v in out.values())

    def test_scores_for_unknown_agents_ignored(self) -> None:
        start = equal_weights(["a", "b"])
        out = update_weights(
            start,
            {"a": 0.2, "b": 0.2, "ghost": 5.0},
            eta=0.35,
            min_weight=0.05,
            max_weight=0.6,
        )
        assert set(out) == {"a", "b"}
        assert abs(sum(out.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# consensus_signals
# ---------------------------------------------------------------------------


class TestConsensusSignals:
    def test_empty_input(self) -> None:
        assert consensus_signals({}, {}) == []
        assert consensus_signals({"a": []}, {"a": 1.0}) == []

    def test_clear_majority_wins(self) -> None:
        signals = {
            "a": [make_signal("a", "BTC", Direction.LONG, 0.9)],
            "b": [make_signal("b", "BTC", Direction.LONG, 0.8)],
            "c": [make_signal("c", "BTC", Direction.SHORT, 0.5)],
        }
        weights = equal_weights(["a", "b", "c"])
        out = consensus_signals(signals, weights)
        assert len(out) == 1
        cons = out[0]
        assert cons.agent == "consensus"
        assert cons.symbol == "BTC"
        assert cons.direction == Direction.LONG
        # winning mass = (0.9 + 0.8)/3; total = (0.9 + 0.8 + 0.5)/3
        assert math.isclose(cons.confidence, 1.7 / 2.2, rel_tol=1e-9)
        assert cons.price_at_signal == 100.0

    def test_weighted_minority_with_huge_confidence_wins(self) -> None:
        signals = {
            "heavy": [make_signal("heavy", "ETH", Direction.SHORT, 1.0, price=50.0)],
            "light1": [make_signal("light1", "ETH", Direction.LONG, 0.3, price=50.0)],
            "light2": [make_signal("light2", "ETH", Direction.LONG, 0.3, price=50.0)],
        }
        weights = {"heavy": 0.6, "light1": 0.2, "light2": 0.2}
        out = consensus_signals(signals, weights)
        assert len(out) == 1
        cons = out[0]
        # SHORT mass = 0.6 * 1.0 = 0.6 > LONG mass = 0.2*0.3 + 0.2*0.3 = 0.12
        assert cons.direction == Direction.SHORT
        assert math.isclose(cons.confidence, 0.6 / 0.72, rel_tol=1e-9)
        assert cons.price_at_signal == 50.0

    def test_tie_is_flat(self) -> None:
        signals = {
            "a": [make_signal("a", "BTC", Direction.LONG, 0.5)],
            "b": [make_signal("b", "BTC", Direction.SHORT, 0.5)],
        }
        out = consensus_signals(signals, equal_weights(["a", "b"]))
        assert out[0].direction == Direction.FLAT

    def test_zero_total_mass_is_flat_with_zero_confidence(self) -> None:
        signals = {"a": [make_signal("a", "BTC", Direction.LONG, 0.0)]}
        out = consensus_signals(signals, {"a": 1.0})
        assert out[0].direction == Direction.FLAT
        assert out[0].confidence == 0.0

    def test_agent_missing_from_weights_contributes_zero(self) -> None:
        signals = {
            "known": [make_signal("known", "BTC", Direction.SHORT, 0.4)],
            "unknown": [make_signal("unknown", "BTC", Direction.LONG, 1.0)],
        }
        out = consensus_signals(signals, {"known": 1.0})
        assert out[0].direction == Direction.SHORT
        assert math.isclose(out[0].confidence, 1.0, rel_tol=1e-9)

    def test_union_of_symbols_across_agents(self) -> None:
        signals = {
            "a": [
                make_signal("a", "BTC", Direction.LONG, 0.5),
                make_signal("a", "ETH", Direction.SHORT, 0.5),
            ],
            "b": [make_signal("b", "SOL", Direction.LONG, 0.5, price=20.0)],
        }
        out = consensus_signals(signals, equal_weights(["a", "b"]))
        assert {s.symbol for s in out} == {"BTC", "ETH", "SOL"}
        sol = next(s for s in out if s.symbol == "SOL")
        assert sol.direction == Direction.LONG
        assert sol.price_at_signal == 20.0

    def test_rationale_summarizes_vote_masses(self) -> None:
        signals = {
            "a": [make_signal("a", "BTC", Direction.LONG, 0.84)],
            "b": [make_signal("b", "BTC", Direction.SHORT, 0.36)],
            "c": [make_signal("c", "BTC", Direction.FLAT, 0.10)],
        }
        weights = {"a": 0.5, "b": 0.5, "c": 0.5}
        out = consensus_signals(signals, weights)
        assert out[0].rationale == "LONG 0.42 vs SHORT 0.18 vs FLAT 0.05"

    def test_confidence_within_bounds(self) -> None:
        signals = {
            "a": [make_signal("a", "BTC", Direction.LONG, 1.0)],
            "b": [make_signal("b", "BTC", Direction.LONG, 1.0)],
        }
        out = consensus_signals(signals, equal_weights(["a", "b"]))
        assert 0.0 <= out[0].confidence <= 1.0
        assert math.isclose(out[0].confidence, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Brier scoring (probability-vector signals) & adaptive eta
# ---------------------------------------------------------------------------


def _prob_signal(p_long: float, p_short: float, p_flat: float) -> Signal:
    from arena.agents import __name__ as _  # noqa: F401 - no agents import needed
    return Signal(
        agent="a",
        symbol="BTC",
        direction=Direction.LONG,
        confidence=max(p_long, p_short, p_flat),
        price_at_signal=100.0,
        p_long=p_long,
        p_short=p_short,
        p_flat=p_flat,
    )


def test_realized_class_thresholds() -> None:
    from arena.engine import realized_class

    assert realized_class(2.0, 1.0) == Direction.LONG
    assert realized_class(-2.0, 1.0) == Direction.SHORT
    assert realized_class(0.5, 1.0) == Direction.FLAT
    assert realized_class(-0.5, 1.0) == Direction.FLAT


def test_brier_perfect_confident_call_scores_one() -> None:
    sig = _prob_signal(1.0, 0.0, 0.0)
    assert score_signal(sig, 105.0) == pytest.approx(1.0)  # +5% -> LONG outcome


def test_brier_confident_miss_scores_minus_one() -> None:
    sig = _prob_signal(1.0, 0.0, 0.0)
    assert score_signal(sig, 95.0) == pytest.approx(-1.0)  # -5% -> SHORT outcome


def test_brier_uniform_forecast_scores_one_third() -> None:
    third = 1.0 / 3.0
    sig = _prob_signal(third, third, third)
    assert score_signal(sig, 105.0) == pytest.approx(1.0 / 3.0)


def test_brier_honest_beats_overconfident_in_expectation() -> None:
    # True distribution: P(LONG)=0.6, P(SHORT)=0.4. Honest report must have
    # higher expected score than an overconfident 0.99/0.01 report.
    honest = _prob_signal(0.6, 0.4, 0.0)
    overconfident = _prob_signal(0.99, 0.01, 0.0)

    def expected(sig: Signal) -> float:
        up = score_signal(sig, 105.0)  # LONG outcome
        down = score_signal(sig, 95.0)  # SHORT outcome
        return 0.6 * up + 0.4 * down

    assert expected(honest) > expected(overconfident)


def test_brier_normalizes_sloppy_vectors() -> None:
    sig = _prob_signal(0.8, 0.4, 0.0)  # sums to 1.2 -> normalized internally
    assert -1.0 <= score_signal(sig, 105.0) <= 1.0


def test_signals_without_probs_keep_legacy_tanh_score() -> None:
    legacy = Signal(
        agent="a", symbol="BTC", direction=Direction.LONG,
        confidence=0.8, price_at_signal=100.0,
    )
    import math

    expected = 0.8 * math.tanh(5.0 / 5.0)
    assert score_signal(legacy, 105.0) == pytest.approx(expected)


def test_adaptive_eta_schedule() -> None:
    import math

    from arena.engine import adaptive_eta

    assert adaptive_eta(1, 4) == pytest.approx(math.sqrt(math.log(4)))
    assert adaptive_eta(100, 4) == pytest.approx(math.sqrt(math.log(4) / 100))
    assert adaptive_eta(4, 4) < adaptive_eta(1, 4)  # decays over time
    assert adaptive_eta(0, 1) > 0  # degenerate inputs stay positive
