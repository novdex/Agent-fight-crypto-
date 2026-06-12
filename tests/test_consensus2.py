"""Tests for arena.engine.consensus2 (U5: prob pooling consensus v2).

Offline only: imports arena.engine.consensus2 and arena.models.
"""

from __future__ import annotations

import math

import pytest

from arena.engine.consensus2 import (
    consensus_interval,
    consensus_signals_v2,
    extremize,
    no_trade_gate,
)
from arena.models import ConsensusSettings, Direction, Signal


def prob_signal(
    agent: str,
    symbol: str = "BTC",
    p_long: float = 1 / 3,
    p_short: float = 1 / 3,
    p_flat: float = 1 / 3,
    confidence: float | None = None,
    price: float = 100.0,
) -> Signal:
    probs = {Direction.LONG: p_long, Direction.SHORT: p_short, Direction.FLAT: p_flat}
    direction = max(probs, key=lambda d: probs[d])
    return Signal(
        agent=agent,
        symbol=symbol,
        direction=direction,
        confidence=max(probs.values()) if confidence is None else confidence,
        price_at_signal=price,
        p_long=p_long,
        p_short=p_short,
        p_flat=p_flat,
    )


def dir_signal(
    agent: str,
    symbol: str = "BTC",
    direction: Direction = Direction.LONG,
    confidence: float = 0.8,
    price: float = 100.0,
) -> Signal:
    return Signal(
        agent=agent,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        price_at_signal=price,
    )


def settings(**kwargs: object) -> ConsensusSettings:
    base: dict[str, object] = {
        "extremize_lambda": 1.0,
        "trim_fraction": 0.0,
        "herding_haircut": 1.0,
        "no_trade_min_confidence": 0.0,
        "mechanism": "vote",
    }
    base.update(kwargs)
    return ConsensusSettings(**base)


def by_symbol(signals: list[Signal]) -> dict[str, Signal]:
    return {s.symbol: s for s in signals}


# ---------------------------------------------------------------------------
# extremize
# ---------------------------------------------------------------------------


class TestExtremize:
    @pytest.mark.parametrize("lam", [0.5, 1.0, 1.3, 3.0])
    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_fixed_points(self, p: float, lam: float) -> None:
        assert extremize(p, lam) == pytest.approx(p)

    @pytest.mark.parametrize("p", [0.05, 0.2, 0.4, 0.5, 0.77, 0.99])
    def test_lambda_one_is_identity(self, p: float) -> None:
        assert extremize(p, 1.0) == pytest.approx(p)

    def test_lambda_gt_one_pushes_away_from_half(self) -> None:
        assert extremize(0.7, 1.3) > 0.7
        assert extremize(0.3, 1.3) < 0.3

    def test_formula(self) -> None:
        p, lam = 0.7, 1.3
        expected = p**lam / (p**lam + (1 - p) ** lam)
        assert extremize(p, lam) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# consensus_signals_v2 — pooling
# ---------------------------------------------------------------------------


class TestPooling:
    def test_pooled_vector_is_weighted_mean(self) -> None:
        sba = {
            "a": [prob_signal("a", p_long=0.8, p_short=0.1, p_flat=0.1)],
            "b": [prob_signal("b", p_long=0.2, p_short=0.6, p_flat=0.2)],
        }
        out = consensus_signals_v2(sba, {"a": 0.75, "b": 0.25}, settings=settings())
        assert len(out) == 1
        sig = out[0]
        assert sig.agent == "consensus"
        assert sig.p_long == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)
        assert sig.p_short == pytest.approx(0.75 * 0.1 + 0.25 * 0.6)
        assert sig.p_flat == pytest.approx(0.75 * 0.1 + 0.25 * 0.2)
        assert sig.direction is Direction.LONG
        assert sig.confidence == pytest.approx(sig.p_long)

    def test_pooling_respects_weights(self) -> None:
        """The heavier agent's view must dominate the pooled outcome."""
        sba = {
            "long_agent": [prob_signal("long_agent", p_long=0.7, p_short=0.2, p_flat=0.1)],
            "short_agent": [prob_signal("short_agent", p_long=0.2, p_short=0.7, p_flat=0.1)],
        }
        heavy_long = consensus_signals_v2(
            sba, {"long_agent": 0.9, "short_agent": 0.1}, settings=settings()
        )[0]
        heavy_short = consensus_signals_v2(
            sba, {"long_agent": 0.1, "short_agent": 0.9}, settings=settings()
        )[0]
        assert heavy_long.direction is Direction.LONG
        assert heavy_short.direction is Direction.SHORT
        assert heavy_long.p_long > heavy_short.p_long

    def test_probless_agent_contributes_one_hot_ish(self) -> None:
        sba = {"a": [dir_signal("a", direction=Direction.SHORT, confidence=0.9)]}
        sig = consensus_signals_v2(sba, {"a": 1.0}, settings=settings())[0]
        assert sig.p_short == pytest.approx(0.9)
        assert sig.p_long == pytest.approx(0.05)
        assert sig.p_flat == pytest.approx(0.05)
        assert sig.direction is Direction.SHORT

    def test_prob_vectors_normalized_before_pooling(self) -> None:
        sba = {"a": [prob_signal("a", p_long=0.4, p_short=0.2, p_flat=0.2)]}  # sums to 0.8
        sig = consensus_signals_v2(sba, {"a": 1.0}, settings=settings())[0]
        assert sig.p_long == pytest.approx(0.5)
        assert sig.p_long + sig.p_short + sig.p_flat == pytest.approx(1.0)

    def test_zero_total_weight_gives_uniform_flat(self) -> None:
        sba = {"a": [prob_signal("a", p_long=0.8, p_short=0.1, p_flat=0.1)]}
        sig = consensus_signals_v2(sba, {}, settings=settings())[0]
        assert sig.direction is Direction.FLAT
        assert sig.p_long == pytest.approx(1 / 3)

    def test_one_signal_per_symbol_union(self) -> None:
        sba = {
            "a": [prob_signal("a", "BTC", 0.6, 0.2, 0.2), prob_signal("a", "ETH", 0.2, 0.6, 0.2)],
            "b": [prob_signal("b", "SOL", 0.2, 0.2, 0.6)],
        }
        out = by_symbol(consensus_signals_v2(sba, {"a": 1.0, "b": 1.0}, settings=settings()))
        assert set(out) == {"BTC", "ETH", "SOL"}
        assert out["BTC"].direction is Direction.LONG
        assert out["ETH"].direction is Direction.SHORT
        assert out["SOL"].direction is Direction.FLAT

    def test_extremize_sharpens_winner_and_renormalizes(self) -> None:
        sba = {"a": [prob_signal("a", p_long=0.6, p_short=0.3, p_flat=0.1)]}
        sig = consensus_signals_v2(
            sba, {"a": 1.0}, settings=settings(extremize_lambda=2.0)
        )[0]
        expected_top = extremize(0.6, 2.0)
        assert sig.p_long == pytest.approx(expected_top)
        assert sig.p_long > 0.6
        # rest renormalized proportionally (0.3 : 0.1 ratio preserved)
        assert sig.p_short / sig.p_flat == pytest.approx(3.0)
        assert sig.p_long + sig.p_short + sig.p_flat == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# per-asset weight override
# ---------------------------------------------------------------------------


class TestPerAssetWeights:
    def test_override_changes_one_coins_outcome(self) -> None:
        sba = {
            "bull": [
                prob_signal("bull", "BTC", 0.7, 0.2, 0.1),
                prob_signal("bull", "ETH", 0.7, 0.2, 0.1),
            ],
            "bear": [
                prob_signal("bear", "BTC", 0.2, 0.7, 0.1),
                prob_signal("bear", "ETH", 0.2, 0.7, 0.1),
            ],
        }
        weights = {"bull": 0.8, "bear": 0.2}
        plain = by_symbol(consensus_signals_v2(sba, weights, settings=settings()))
        assert plain["BTC"].direction is Direction.LONG
        assert plain["ETH"].direction is Direction.LONG

        per_asset = {"ETH": {"bull": 0.1, "bear": 0.9}}
        out = by_symbol(
            consensus_signals_v2(
                sba, weights, settings=settings(), per_asset_weights=per_asset
            )
        )
        assert out["BTC"].direction is Direction.LONG  # untouched coin
        assert out["ETH"].direction is Direction.SHORT  # flipped by override


# ---------------------------------------------------------------------------
# trimmed pooling
# ---------------------------------------------------------------------------


class TestTrimming:
    def test_trim_drops_extreme_weights(self) -> None:
        sba = {
            "tiny": [prob_signal("tiny", p_long=0.1, p_short=0.8, p_flat=0.1)],
            "mid": [prob_signal("mid", p_long=0.8, p_short=0.1, p_flat=0.1)],
            "huge": [prob_signal("huge", p_long=0.1, p_short=0.8, p_flat=0.1)],
        }
        weights = {"tiny": 0.01, "mid": 0.3, "huge": 5.0}
        untrimmed = consensus_signals_v2(sba, weights, settings=settings())[0]
        assert untrimmed.direction is Direction.SHORT
        # trim_fraction 1/3 drops tiny and huge, leaving only mid.
        trimmed = consensus_signals_v2(
            sba, weights, settings=settings(trim_fraction=0.34)
        )[0]
        assert trimmed.direction is Direction.LONG
        assert trimmed.p_long == pytest.approx(0.8)

    def test_trim_never_drops_everyone(self) -> None:
        sba = {"only": [prob_signal("only", p_long=0.6, p_short=0.2, p_flat=0.2)]}
        sig = consensus_signals_v2(
            sba, {"only": 1.0}, settings=settings(trim_fraction=0.5)
        )[0]
        assert sig.direction is Direction.LONG


# ---------------------------------------------------------------------------
# herding haircut
# ---------------------------------------------------------------------------


class TestHerdingHaircut:
    def test_haircut_applies_on_unanimity(self) -> None:
        sba = {
            "a": [prob_signal("a", p_long=0.7, p_short=0.2, p_flat=0.1)],
            "b": [prob_signal("b", p_long=0.6, p_short=0.3, p_flat=0.1)],
        }
        w = {"a": 0.5, "b": 0.5}
        plain = consensus_signals_v2(sba, w, settings=settings(herding_haircut=1.0))[0]
        cut = consensus_signals_v2(sba, w, settings=settings(herding_haircut=0.85))[0]
        assert cut.confidence == pytest.approx(plain.confidence * 0.85)
        # the prob vector itself is untouched
        assert cut.p_long == pytest.approx(plain.p_long)

    def test_no_haircut_on_disagreement(self) -> None:
        sba = {
            "a": [prob_signal("a", p_long=0.7, p_short=0.2, p_flat=0.1)],
            "b": [prob_signal("b", p_long=0.2, p_short=0.7, p_flat=0.1)],
        }
        w = {"a": 0.7, "b": 0.3}
        sig = consensus_signals_v2(sba, w, settings=settings(herding_haircut=0.5))[0]
        assert sig.confidence == pytest.approx(max(sig.p_long, sig.p_short, sig.p_flat))
        assert "unanimous" not in sig.rationale

    def test_haircut_uses_directionless_agents_stated_direction(self) -> None:
        sba = {
            "a": [dir_signal("a", direction=Direction.LONG, confidence=0.9)],
            "b": [prob_signal("b", p_long=0.8, p_short=0.1, p_flat=0.1)],
        }
        sig = consensus_signals_v2(
            sba, {"a": 0.5, "b": 0.5}, settings=settings(herding_haircut=0.85)
        )[0]
        assert "unanimous" in sig.rationale
        assert sig.confidence == pytest.approx(max(sig.p_long, sig.p_short, sig.p_flat) * 0.85)


# ---------------------------------------------------------------------------
# LMSR mechanism
# ---------------------------------------------------------------------------


class TestLMSR:
    def test_prices_sum_to_one(self) -> None:
        sba = {
            "a": [prob_signal("a", p_long=0.7, p_short=0.2, p_flat=0.1)],
            "b": [prob_signal("b", p_long=0.1, p_short=0.8, p_flat=0.1)],
            "c": [dir_signal("c", direction=Direction.FLAT, confidence=0.4)],
        }
        sig = consensus_signals_v2(
            sba, {"a": 0.4, "b": 0.4, "c": 0.2}, settings=settings(mechanism="lmsr")
        )[0]
        assert sig.p_long + sig.p_short + sig.p_flat == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in (sig.p_long, sig.p_short, sig.p_flat))
        assert "lmsr" in sig.rationale

    def test_prices_favor_heavier_bettor(self) -> None:
        sba = {
            "long_agent": [prob_signal("long_agent", p_long=0.8, p_short=0.1, p_flat=0.1)],
            "short_agent": [prob_signal("short_agent", p_long=0.1, p_short=0.8, p_flat=0.1)],
        }
        lmsr = settings(mechanism="lmsr")
        heavy_long = consensus_signals_v2(
            sba, {"long_agent": 0.9, "short_agent": 0.1}, settings=lmsr
        )[0]
        heavy_short = consensus_signals_v2(
            sba, {"long_agent": 0.1, "short_agent": 0.9}, settings=lmsr
        )[0]
        assert heavy_long.direction is Direction.LONG
        assert heavy_long.p_long > heavy_long.p_short
        assert heavy_short.direction is Direction.SHORT
        assert heavy_short.p_short > heavy_short.p_long

    def test_no_bets_gives_uniform(self) -> None:
        sba = {"a": [prob_signal("a", p_long=0.8, p_short=0.1, p_flat=0.1)]}
        sig = consensus_signals_v2(sba, {}, settings=settings(mechanism="lmsr"))[0]
        assert sig.p_long == pytest.approx(1 / 3)
        assert sig.direction is Direction.FLAT


# ---------------------------------------------------------------------------
# no_trade_gate
# ---------------------------------------------------------------------------


class TestNoTradeGate:
    def make_consensus(self, top: float) -> list[Signal]:
        rest = (1.0 - top) / 2.0
        return [
            prob_signal("consensus", "BTC", top, rest, rest),
            prob_signal("consensus", "ETH", top, rest, rest),
        ]

    def test_gate_fires_below_threshold(self) -> None:
        signals, gated = no_trade_gate(self.make_consensus(0.4), 0.6)
        assert gated is True
        assert len(signals) == 2
        for s in signals:
            assert s.direction is Direction.FLAT
            assert s.confidence == 0.0
            assert s.p_long == pytest.approx(1 / 3)
            assert s.p_short == pytest.approx(1 / 3)
            assert s.p_flat == pytest.approx(1 / 3)

    def test_gate_passes_above_threshold(self) -> None:
        original = self.make_consensus(0.8)
        signals, gated = no_trade_gate(original, 0.6)
        assert gated is False
        assert signals is original

    def test_threshold_zero_disables(self) -> None:
        original = self.make_consensus(0.05)
        signals, gated = no_trade_gate(original, 0.0)
        assert gated is False
        assert signals is original

    def test_uses_mean_across_symbols(self) -> None:
        mixed = [
            prob_signal("consensus", "BTC", 0.9, 0.05, 0.05),
            prob_signal("consensus", "ETH", 0.4, 0.3, 0.3),
        ]  # mean top-prob 0.65
        _, gated = no_trade_gate(mixed, 0.6)
        assert gated is False
        _, gated = no_trade_gate(mixed, 0.7)
        assert gated is True

    def test_empty_list_not_gated(self) -> None:
        signals, gated = no_trade_gate([], 0.5)
        assert signals == [] and gated is False


# ---------------------------------------------------------------------------
# consensus_interval
# ---------------------------------------------------------------------------


class TestConsensusInterval:
    def test_interval_ordering_and_bounds(self) -> None:
        sba = {
            f"a{i}": [prob_signal(f"a{i}", p_long=p, p_short=(1 - p) / 2, p_flat=(1 - p) / 2)]
            for i, p in enumerate([0.5, 0.55, 0.6, 0.7, 0.8, 0.9])
        }
        lo, hi = consensus_interval(sba, "BTC")
        assert lo <= hi
        assert 0.5 <= lo and hi <= 0.9
        assert lo < hi  # genuinely spread agents -> non-degenerate band

    def test_single_agent_collapses(self) -> None:
        sba = {"a": [prob_signal("a", p_long=0.7, p_short=0.2, p_flat=0.1)]}
        lo, hi = consensus_interval(sba, "BTC")
        assert lo == hi == pytest.approx(0.7)

    def test_no_agents(self) -> None:
        assert consensus_interval({}, "BTC") == (0.0, 0.0)

    def test_uses_winning_direction(self) -> None:
        sba = {
            "a": [prob_signal("a", p_long=0.1, p_short=0.8, p_flat=0.1)],
            "b": [prob_signal("b", p_long=0.2, p_short=0.6, p_flat=0.2)],
        }
        lo, hi = consensus_interval(sba, "BTC")
        # winning direction is SHORT; band covers the agents' p_short values
        assert 0.6 <= lo <= hi <= 0.8

    def test_interval_is_finite_and_in_unit_range(self) -> None:
        sba = {
            "a": [dir_signal("a", direction=Direction.LONG, confidence=0.9)],
            "b": [dir_signal("b", direction=Direction.LONG, confidence=0.5)],
            "c": [dir_signal("c", direction=Direction.LONG, confidence=0.7)],
        }
        lo, hi = consensus_interval(sba, "BTC")
        assert math.isfinite(lo) and math.isfinite(hi)
        assert 0.0 <= lo <= hi <= 1.0
