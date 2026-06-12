"""Offline tests for the brain unit (debate, critic, memory, optimizer)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.brain import (
    VARIANTS,
    AgentMemory,
    PromptVariantBook,
    critic_review,
    debate_context,
)
from arena.models import CoinSnapshot, Direction, MarketSnapshot, ScoredSignal, Signal


def make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        as_of=datetime(2026, 6, 12, tzinfo=timezone.utc),
        fear_greed=72,
        coins=[
            CoinSnapshot(
                symbol="BTC",
                price_usd=65_000.0,
                change_24h_pct=3.2,
                change_7d_pct=8.0,
                rsi_14=78.0,
                funding_rate_pct=0.03,
                adx_14=31.0,
            ),
            CoinSnapshot(
                symbol="ETH",
                price_usd=3_200.0,
                change_24h_pct=-4.5,
                change_7d_pct=-9.0,
                rsi_14=24.0,
                funding_rate_pct=-0.02,
                adx_14=15.0,
            ),
            CoinSnapshot(symbol="SOL", price_usd=150.0),
        ],
    )


def sig(
    symbol: str,
    rationale: str,
    confidence: float = 0.8,
    direction: Direction = Direction.LONG,
    probs: tuple[float, float, float] | None = None,
) -> Signal:
    p_long, p_short, p_flat = probs if probs else (None, None, None)
    return Signal(
        agent="a1",
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        rationale=rationale,
        price_at_signal=1.0,
        p_long=p_long,
        p_short=p_short,
        p_flat=p_flat,
    )


# --- debate_context ---------------------------------------------------------


def test_debate_context_two_calls_and_brief_shape() -> None:
    snapshot = make_snapshot()
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return "Bull thesis." if "BULL" in prompt else "Bear thesis."

    brief = debate_context(snapshot, ask)
    assert len(prompts) == 2
    assert prompts[0] != prompts[1]
    assert "BULL" in prompts[0] and "BEAR" in prompts[1]
    # Snapshot numbers reach the debaters.
    assert "BTC" in prompts[0] and "65,000" in prompts[0]
    assert brief.startswith("DEBATE BRIEF")
    assert "BULL CASE: Bull thesis." in brief
    assert "BEAR CASE: Bear thesis." in brief


def test_debate_context_truncates_to_2000_chars() -> None:
    snapshot = make_snapshot()
    brief = debate_context(snapshot, lambda _p: "word " * 2_000)
    assert len(brief) <= 2000
    assert brief.startswith("DEBATE BRIEF")


def test_debate_context_deterministic() -> None:
    snapshot = make_snapshot()

    def ask(prompt: str) -> str:
        return "up" if "BULL" in prompt else "down"

    assert debate_context(snapshot, ask) == debate_context(snapshot, ask)


# --- critic_review ----------------------------------------------------------


def test_critic_halves_confidence_on_oversold_vs_high_rsi() -> None:
    snapshot = make_snapshot()  # BTC rsi_14 = 78
    original = sig("BTC", "Deeply oversold, bounce imminent", confidence=0.8)
    out = critic_review([original], snapshot)
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(0.4)
    assert "[critic:" in out[0].rationale
    assert "RSI14=78" in out[0].rationale
    # Input signal is never mutated.
    assert original.confidence == 0.8
    assert "[critic:" not in original.rationale


def test_critic_overbought_vs_low_rsi_and_funding_claims() -> None:
    snapshot = make_snapshot()  # ETH rsi_14 = 24, funding -0.02
    flagged = critic_review(
        [
            sig("ETH", "overbought after the run", confidence=0.6),
            sig("ETH", "positive funding shows greed", confidence=0.5),
            sig("BTC", "negative funding, shorts paying", confidence=0.5),
        ],
        snapshot,
    )
    assert flagged[0].confidence == pytest.approx(0.3)
    assert flagged[1].confidence == pytest.approx(0.25)
    assert flagged[2].confidence == pytest.approx(0.25)
    for s in flagged:
        assert "[critic:" in s.rationale


def test_critic_leaves_consistent_signals_untouched() -> None:
    snapshot = make_snapshot()
    signals = [
        sig("BTC", "overbought, fading the move", direction=Direction.SHORT),
        sig("ETH", "oversold bounce", confidence=0.7),
        sig("SOL", "no data either way", confidence=0.1),
        sig("BTC", "", confidence=0.9),  # empty rationale -> nothing to check
    ]
    out = critic_review(signals, snapshot)
    assert out == signals


def test_critic_never_drops_coverage_and_handles_unknown_symbols() -> None:
    snapshot = make_snapshot()
    signals = [sig("DOGE", "oversold"), sig("BTC", "oversold")]
    out = critic_review(signals, snapshot)
    assert [s.symbol for s in out] == ["DOGE", "BTC"]


def test_critic_attenuates_prob_vector_preserving_argmax() -> None:
    snapshot = make_snapshot()
    original = sig("BTC", "oversold", confidence=0.7, probs=(0.7, 0.1, 0.2))
    out = critic_review([original], snapshot)[0]
    assert out.p_long == pytest.approx(0.7 / 2 + 1 / 6)
    assert out.p_short == pytest.approx(0.1 / 2 + 1 / 6)
    assert out.p_long + out.p_short + out.p_flat == pytest.approx(1.0)
    assert out.p_long > out.p_flat > out.p_short  # ordering preserved
    assert out.direction == Direction.LONG


# --- AgentMemory ------------------------------------------------------------


def scored(symbol: str, score: float, direction: Direction = Direction.LONG,
           p0: float = 100.0, p1: float = 103.0) -> ScoredSignal:
    return ScoredSignal(
        agent="a1",
        symbol=symbol,
        direction=direction,
        confidence=0.5,
        price_at_signal=p0,
        price_at_eval=p1,
        score=score,
    )


def test_memory_round_trip_and_persistence(tmp_path) -> None:
    mem = AgentMemory(str(tmp_path), "claude")
    assert mem.lessons() == []
    mem.record_round(1, 0.25, "Round one went fine.")
    mem.record_round(2, -0.10, "Round two missed momentum.")
    lessons = mem.lessons()
    assert any("Round two missed momentum." in item for item in lessons)
    assert any("Lifetime: 2 rounds" in item for item in lessons)
    # A fresh instance reads the same file.
    again = AgentMemory(str(tmp_path), "claude")
    assert again.lessons() == lessons
    # No stray tmp files left behind by atomic writes.
    assert [p.name for p in tmp_path.iterdir()] == ["claude.json"]


def test_memory_lessons_capped_and_agents_isolated(tmp_path) -> None:
    a = AgentMemory(str(tmp_path), "a")
    b = AgentMemory(str(tmp_path), "b")
    for i in range(12):
        a.record_round(i, 0.1, f"reflection {i}")
    assert len(a.lessons(max_items=3)) == 3
    assert len(a.lessons()) <= 8
    assert b.lessons() == []


def test_memory_reflect_is_deterministic_template() -> None:
    mem = AgentMemory("/tmp/unused-root", "x")
    rows = [
        scored("BTC", 0.8, Direction.LONG, 100, 104),
        scored("ETH", -0.6, Direction.SHORT, 100, 103),
        scored("SOL", 0.0, Direction.FLAT, 100, 100.2),
    ]
    text = mem.reflect(0.07, rows)
    assert text == mem.reflect(0.07, rows)
    assert "Round score +0.070." in text
    assert "Worked: BTC LONG (+0.80)" in text
    assert "Failed: ETH SHORT (-0.60)" in text
    assert "Regime: risk-on drift" in text
    assert "Regime: unknown" in mem.reflect(0.0, [])


def test_memory_survives_corrupt_file(tmp_path) -> None:
    (tmp_path / "bot.json").write_text("{not json", encoding="utf-8")
    mem = AgentMemory(str(tmp_path), "bot")
    assert mem.lessons() == []
    mem.record_round(1, 0.5, "recovered")
    assert any("recovered" in item for item in mem.lessons())


# --- PromptVariantBook --------------------------------------------------------


def test_variant_book_ships_three_variants() -> None:
    assert set(VARIANTS) == {"neutral", "risk_emphasis", "contrarian_emphasis"}
    assert all(isinstance(v, str) and v for v in VARIANTS.values())


def test_variant_book_serves_known_preambles_and_persists(tmp_path) -> None:
    book = PromptVariantBook(str(tmp_path))
    pre = book.preamble_for("claude")
    assert pre in VARIANTS.values()
    assert (tmp_path / "prompt_variants.json").exists()
    # State (the pending serve) survives re-instantiation.
    book2 = PromptVariantBook(str(tmp_path))
    book2.record_result("claude", 0.5)  # must not raise; attributes to last served


def test_variant_book_promotes_best_variant(tmp_path) -> None:
    book = PromptVariantBook(str(tmp_path))
    inverse = {text: name for name, text in VARIANTS.items()}
    # Probe phase: each variant gets sampled; contrarian scores best.
    for _ in range(6):
        served = inverse[book.preamble_for("claude")]
        book.record_result(
            "claude", 0.9 if served == "contrarian_emphasis" else -0.5
        )
    # Champion is now the contrarian variant and dominates subsequent serves.
    serves = []
    for _ in range(4):
        serves.append(inverse[book.preamble_for("claude")])
        book.record_result("claude", 0.0)
    assert serves.count("contrarian_emphasis") >= 3


def test_variant_book_deterministic_serve_sequence(tmp_path) -> None:
    def run(sub: str) -> list[str]:
        book = PromptVariantBook(str(tmp_path / sub))
        inverse = {text: name for name, text in VARIANTS.items()}
        out = []
        for i in range(8):
            out.append(inverse[book.preamble_for("a")])
            book.record_result("a", 0.1 * i)
        return out

    assert run("one") == run("two")


def test_variant_book_agents_tracked_independently(tmp_path) -> None:
    book = PromptVariantBook(str(tmp_path))
    inverse = {text: name for name, text in VARIANTS.items()}
    for _ in range(6):
        served = inverse[book.preamble_for("alpha")]
        book.record_result("alpha", 1.0 if served == "risk_emphasis" else -1.0)
    # "beta" is untouched: probe starts from the first variant again.
    assert inverse[book.preamble_for("beta")] == "neutral"
