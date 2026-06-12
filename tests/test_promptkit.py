"""Offline tests for arena.agents.personas + arena.agents.promptkit (U2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.agents.personas import ROLES, persona_preamble
from arena.agents.promptkit import (
    MAX_HEADLINES,
    MAX_MEMORY_LESSONS,
    build_enhanced_prompt,
    chart_pattern_notes,
    detect_regime,
    fact_subjectivity_split,
    fincot_blueprint,
    regime_demos,
)
from arena.agents.prompts import parse_signals
from arena.models import CoinSnapshot, Direction, MarketSnapshot


def make_coin(symbol: str = "BTC", **kwargs: object) -> CoinSnapshot:
    defaults: dict[str, object] = {
        "symbol": symbol,
        "price_usd": 50_000.0,
        "change_1h_pct": 0.1,
        "change_24h_pct": 0.5,
        "change_7d_pct": 1.0,
        "rsi_14": 50.0,
        "ema_20_dist_pct": 0.5,
        "volatility_24h_pct": 1.2,
    }
    defaults.update(kwargs)
    return CoinSnapshot(**defaults)  # type: ignore[arg-type]


def make_snapshot(coins: list[CoinSnapshot] | None = None, **kwargs: object) -> MarketSnapshot:
    if coins is None:
        coins = [
            make_coin("BTC", extras={"bb_width_pct": 2.5, "ema_ribbon": 0.8}),
            make_coin("ETH", price_usd=3_000.0),
            make_coin("SOL", price_usd=150.0),
        ]
    defaults: dict[str, object] = {
        "as_of": datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc),
        "coins": coins,
        "fear_greed": 55,
    }
    defaults.update(kwargs)
    return MarketSnapshot(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# personas
# ---------------------------------------------------------------------------


def test_roles_dict_contains_all_required_personas() -> None:
    required = {
        "technical",
        "sentiment",
        "onchain",
        "macro",
        "risk",
        "contrarian",
        "synthesizer",
        "generalist",
    }
    assert required <= set(ROLES)
    assert all(isinstance(v, str) and v for v in ROLES.values())


def test_persona_preamble_known_and_fallback() -> None:
    assert persona_preamble("technical") == ROLES["technical"]
    assert persona_preamble("  Technical ") == ROLES["technical"]
    assert persona_preamble("quant-wizard") == ROLES["generalist"]
    assert persona_preamble("") == ROLES["generalist"]


# ---------------------------------------------------------------------------
# detect_regime
# ---------------------------------------------------------------------------


def test_detect_regime_bull() -> None:
    coins = [
        make_coin("BTC", change_24h_pct=3.0, change_7d_pct=8.0, adx_14=30.0),
        make_coin("ETH", change_24h_pct=2.5, change_7d_pct=6.0, adx_14=28.0),
    ]
    assert detect_regime(make_snapshot(coins)) == "bull"


def test_detect_regime_bear() -> None:
    coins = [
        make_coin("BTC", change_24h_pct=-3.0, change_7d_pct=-9.0, adx_14=30.0),
        make_coin("ETH", change_24h_pct=-2.0, change_7d_pct=-7.0, adx_14=27.0),
    ]
    assert detect_regime(make_snapshot(coins)) == "bear"


def test_detect_regime_chop_low_adx_small_moves() -> None:
    coins = [
        make_coin("BTC", change_24h_pct=0.3, change_7d_pct=-0.5, adx_14=14.0),
        make_coin("ETH", change_24h_pct=-0.2, change_7d_pct=0.8, adx_14=16.0),
    ]
    assert detect_regime(make_snapshot(coins)) == "chop"


def test_detect_regime_trending_7d_with_adx() -> None:
    # Moderate 24h breadth but a strong 7d trend confirmed by ADX -> bull.
    coins = [
        make_coin("BTC", change_24h_pct=0.8, change_7d_pct=9.0, adx_14=32.0),
        make_coin("ETH", change_24h_pct=0.6, change_7d_pct=7.0, adx_14=29.0),
    ]
    assert detect_regime(make_snapshot(coins)) == "bull"
    # Same breadth without trend confirmation -> chop.
    coins_weak = [
        make_coin("BTC", change_24h_pct=0.8, change_7d_pct=9.0, adx_14=15.0),
        make_coin("ETH", change_24h_pct=0.6, change_7d_pct=7.0, adx_14=12.0),
    ]
    assert detect_regime(make_snapshot(coins_weak)) == "chop"


def test_detect_regime_uses_ema_ribbon_extra_when_adx_missing() -> None:
    coins = [
        make_coin(
            "BTC",
            change_24h_pct=1.0,
            change_7d_pct=8.0,
            adx_14=None,
            extras={"ema_ribbon": 0.9},
        ),
    ]
    assert detect_regime(make_snapshot(coins)) == "bull"


def test_detect_regime_handles_missing_fields() -> None:
    coins = [
        CoinSnapshot(symbol="XYZ", price_usd=1.0),
    ]
    assert detect_regime(make_snapshot(coins)) == "chop"


# ---------------------------------------------------------------------------
# chart_pattern_notes
# ---------------------------------------------------------------------------


def test_chart_pattern_notes_triggers() -> None:
    coins = [
        make_coin(
            "BTC",
            rsi_14=78.0,
            change_24h_pct=2.0,
            ema_20_dist_pct=6.5,
            funding_rate_pct=0.08,
            extras={"bb_width_pct": 2.0, "ema_ribbon": 0.9},
        ),
        make_coin("ETH", rsi_14=22.0, extras={"ema_ribbon": -0.8}),
    ]
    notes = chart_pattern_notes(make_snapshot(coins))
    assert "BTC" in notes and "ETH" in notes
    assert "overbought" in notes
    assert "oversold" in notes
    assert "BB squeeze" in notes
    assert "stacked bullish" in notes
    assert "stacked bearish" in notes
    assert "crowded longs" in notes
    assert "snapback" in notes


def test_chart_pattern_notes_divergence_proxies() -> None:
    coins = [
        make_coin("BTC", change_24h_pct=2.5, rsi_14=40.0),
        make_coin("ETH", change_24h_pct=-2.5, rsi_14=60.0),
    ]
    notes = chart_pattern_notes(make_snapshot(coins))
    assert "bearish divergence proxy" in notes
    assert "bullish divergence proxy" in notes


def test_chart_pattern_notes_empty_when_nothing_notable() -> None:
    coins = [
        make_coin("BTC", rsi_14=50.0, change_24h_pct=0.2, ema_20_dist_pct=0.5,
                  funding_rate_pct=0.005, extras={"bb_width_pct": 8.0, "ema_ribbon": 0.1}),
    ]
    assert chart_pattern_notes(make_snapshot(coins)) == ""


# ---------------------------------------------------------------------------
# regime_demos / blueprint / split
# ---------------------------------------------------------------------------


def test_regime_demos_per_regime_and_fallback() -> None:
    for regime in ("bull", "bear", "chop"):
        demos = regime_demos(regime)
        assert "p_long" in demos and "p_flat" in demos
    assert regime_demos("bull") != regime_demos("bear")
    assert regime_demos("unknown") == regime_demos("chop")


def test_fincot_blueprint_and_fact_split() -> None:
    snap = make_snapshot()
    blueprint = fincot_blueprint(snap)
    assert "REGIME" in blueprint and "PROBABILITIES" in blueprint
    split = fact_subjectivity_split()
    assert "FACTS" in split and "NARRATIVE" in split


# ---------------------------------------------------------------------------
# build_enhanced_prompt
# ---------------------------------------------------------------------------


def test_prompt_contains_every_symbol_and_contract() -> None:
    snap = make_snapshot()
    prompt = build_enhanced_prompt(snap, role="technical")
    for symbol in snap.symbols:
        assert symbol in prompt
    assert "p_long" in prompt
    assert "sum to 1" in prompt
    assert "Brier" in prompt
    assert "STRICT JSON" in prompt
    assert ROLES["technical"].split(".")[0] in prompt


def test_prompt_persona_fallback_and_optional_blocks() -> None:
    snap = make_snapshot()
    prompt = build_enhanced_prompt(
        snap,
        role="nonexistent-role",
        memory_lessons=["lesson-alpha", "lesson-beta"],
        debate_brief="DEBATE BRIEF: bulls cite momentum; bears cite crowding.",
        preamble="OPTIMIZER PREAMBLE: stay calibrated.",
    )
    assert ROLES["generalist"].split(".")[0] in prompt
    assert "lesson-alpha" in prompt and "lesson-beta" in prompt
    assert "DEBATE BRIEF" in prompt
    assert "OPTIMIZER PREAMBLE" in prompt


def test_prompt_caps_memory_lessons_and_headlines() -> None:
    snap = make_snapshot(headlines=[f"headline-{i}" for i in range(20)])
    lessons = [f"lesson-{i}" for i in range(20)]
    prompt = build_enhanced_prompt(snap, memory_lessons=lessons)
    assert f"lesson-{MAX_MEMORY_LESSONS - 1}" in prompt
    assert f"lesson-{MAX_MEMORY_LESSONS}" not in prompt
    assert f"headline-{MAX_HEADLINES - 1}" in prompt
    assert f"headline-{MAX_HEADLINES}" not in prompt


def test_prompt_renders_extras_and_headlines() -> None:
    snap = make_snapshot(
        extras={"btc_dominance_pct": 54.2},
        headlines=["SEC approves new ETF"],
    )
    prompt = build_enhanced_prompt(snap)
    assert "btc_dominance_pct" in prompt
    assert "bb_width_pct" in prompt  # coin-level extra rendered
    assert "SEC approves new ETF" in prompt


def test_parse_signals_roundtrip_with_compliant_reply() -> None:
    snap = make_snapshot()
    prompt = build_enhanced_prompt(snap, role="risk")
    # A reply following the prompt's strict-JSON contract exactly.
    reply = json.dumps(
        {
            "signals": [
                {"symbol": "BTC", "p_long": 0.6, "p_short": 0.1, "p_flat": 0.3,
                 "rationale": "trend up"},
                {"symbol": "ETH", "p_long": 0.2, "p_short": 0.5, "p_flat": 0.3,
                 "rationale": "weak"},
                {"symbol": "SOL", "p_long": 0.3, "p_short": 0.3, "p_flat": 0.4,
                 "rationale": "chop"},
            ]
        }
    )
    assert '"p_long"' in prompt  # contract shape advertised in the prompt
    signals = parse_signals(reply, "tester", snap)
    assert [s.symbol for s in signals] == snap.symbols
    by_symbol = {s.symbol: s for s in signals}
    assert by_symbol["BTC"].direction == Direction.LONG
    assert by_symbol["ETH"].direction == Direction.SHORT
    assert by_symbol["SOL"].direction == Direction.FLAT
    assert all(s.has_probs for s in signals)
    assert abs(by_symbol["BTC"].p_long - 0.6) < 1e-9
    assert by_symbol["BTC"].price_at_signal == 50_000.0


def test_prompt_is_compact() -> None:
    snap = make_snapshot(headlines=[f"headline {i} " + "x" * 300 for i in range(50)])
    lessons = ["L" * 1000 for _ in range(50)]
    prompt = build_enhanced_prompt(
        snap, memory_lessons=lessons, debate_brief="D" * 10_000
    )
    # Caps keep the prompt bounded even with abusive inputs.
    assert len(prompt) < 12_000
