"""Enhanced prompt construction: FinCoT scaffold, regimes, patterns, memory.

FROZEN INTERFACE — see docs/INTERFACES2.md (Unit U2).

``build_enhanced_prompt`` composes a persona, an expert-workflow blueprint, a
fact-vs-narrative split, locally rendered market data, deterministic chart
pattern notes, regime-matched few-shot exemplars, memory lessons, and an
optional debate brief — and ends with the SAME strict-JSON probability output
contract as ``arena.agents.prompts`` so its replies parse with
``arena.agents.prompts.parse_signals``. This module must NOT import
``arena.agents.prompts``.
"""

from __future__ import annotations

from arena.agents.personas import persona_preamble
from arena.models import ArenaSettings, CoinSnapshot, MarketSnapshot

# ---------------------------------------------------------------------------
# Compactness caps (prompts must not grow without bound).
# ---------------------------------------------------------------------------
MAX_MEMORY_LESSONS = 8
MAX_HEADLINES = 8
MAX_LESSON_CHARS = 200
MAX_HEADLINE_CHARS = 160
MAX_DEBATE_CHARS = 2000
MAX_EXTRAS_PER_COIN = 8

# Deterministic thresholds (shared by detect_regime / chart_pattern_notes).
_BULL_24H_PCT = 1.5
_BEAR_24H_PCT = -1.5
_BULL_7D_PCT = 5.0
_BEAR_7D_PCT = -5.0
_ADX_TRENDING = 25.0
_BB_SQUEEZE_WIDTH_PCT = 4.0
_RSI_OVERBOUGHT = 70.0
_RSI_OVERSOLD = 30.0
_EMA_RIBBON_STACKED = 0.5
_EMA_DIST_EXTENDED_PCT = 5.0
_FUNDING_EXTREME_PCT = 0.05


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fincot_blueprint(snapshot: MarketSnapshot) -> str:
    """Expert-workflow scaffold (FinCoT): the order in which a pro reasons."""
    horizon = ArenaSettings().horizon_hours
    n = len(snapshot.coins)
    return (
        "ANALYST WORKFLOW — reason through these steps IN ORDER before answering "
        f"(for all {n} coins, {horizon:g}h horizon):\n"
        "1. REGIME: classify the market (trending bull / trending bear / chop) from "
        "breadth of 24h/7d changes and ADX.\n"
        "2. TECHNICALS: per coin — momentum (1h/24h/7d), RSI, EMA posture, "
        "volatility, squeeze/breakout setups.\n"
        "3. POSITIONING: funding rates, open interest, long/short ratios — is the "
        "trade crowded? Squeeze fuel?\n"
        "4. SENTIMENT & FLOWS: Fear & Greed, headlines, on-chain/market extras.\n"
        "5. CROSS-CHECK: where do the streams above conflict? Conflicts push "
        "probability mass toward FLAT.\n"
        "6. PROBABILITIES: convert the net view into honest p_long/p_short/p_flat "
        "per coin. Do all reasoning silently; output only the final JSON."
    )


def fact_subjectivity_split() -> str:
    """Instruction block demanding separate FACTS vs NARRATIVE analysis."""
    return (
        "FACTS vs NARRATIVE — keep these mentally separate:\n"
        "- FACTS: the numeric snapshot below (prices, changes, RSI, funding, OI, "
        "extras). These are ground truth; never contradict them.\n"
        "- NARRATIVE: headlines, vibes, and stories. Treat as subjective and "
        "possibly stale or planted; use only to adjust, never to override, facts.\n"
        "If a narrative conflicts with the numbers, trust the numbers and lower "
        "your confidence."
    )


# ---------------------------------------------------------------------------
# Deterministic regime detection.
# ---------------------------------------------------------------------------


def detect_regime(snapshot: MarketSnapshot) -> str:
    """Classify the snapshot as "bull" | "bear" | "chop". Deterministic.

    Uses the mean 24h and 7d change across coins plus trend strength from
    mean ADX-14 (falling back to the ``ema_ribbon`` extra when ADX is absent).
    Strong breadth alone calls the regime; moderate breadth needs a trending
    market (ADX >= 25 or stacked EMA ribbon) to confirm, else "chop".
    """
    mean24 = _mean([c.change_24h_pct for c in snapshot.coins if c.change_24h_pct is not None]) or 0.0
    mean7 = _mean([c.change_7d_pct for c in snapshot.coins if c.change_7d_pct is not None]) or 0.0
    adx = _mean([c.adx_14 for c in snapshot.coins if c.adx_14 is not None])
    ribbon = _mean(
        [c.extras["ema_ribbon"] for c in snapshot.coins if "ema_ribbon" in c.extras]
    )
    trending = (adx is not None and adx >= _ADX_TRENDING) or (
        ribbon is not None and abs(ribbon) >= _EMA_RIBBON_STACKED
    )

    if mean24 >= _BULL_24H_PCT and mean7 >= 0.0:
        return "bull"
    if mean24 <= _BEAR_24H_PCT and mean7 <= 0.0:
        return "bear"
    if trending:
        if mean7 >= _BULL_7D_PCT and mean24 > 0.0:
            return "bull"
        if mean7 <= _BEAR_7D_PCT and mean24 < 0.0:
            return "bear"
    return "chop"


# ---------------------------------------------------------------------------
# Deterministic chart-pattern annotations.
# ---------------------------------------------------------------------------


def _coin_pattern_notes(c: CoinSnapshot) -> list[str]:
    notes: list[str] = []
    rsi = c.rsi_14
    chg = c.change_24h_pct
    if rsi is not None:
        if rsi >= _RSI_OVERBOUGHT:
            notes.append(f"RSI {rsi:.0f} overbought")
        elif rsi <= _RSI_OVERSOLD:
            notes.append(f"RSI {rsi:.0f} oversold")
        if chg is not None:
            if chg > 1.0 and rsi < 45.0:
                notes.append("bearish divergence proxy (price up, RSI weak)")
            elif chg < -1.0 and rsi > 55.0:
                notes.append("bullish divergence proxy (price down, RSI firm)")
    bb_width = c.extras.get("bb_width_pct")
    if bb_width is not None and 0.0 < bb_width <= _BB_SQUEEZE_WIDTH_PCT:
        notes.append(f"BB squeeze (width {bb_width:.1f}%) — breakout watch")
    ribbon = c.extras.get("ema_ribbon")
    if ribbon is not None:
        if ribbon >= _EMA_RIBBON_STACKED:
            notes.append("EMA ribbon stacked bullish")
        elif ribbon <= -_EMA_RIBBON_STACKED:
            notes.append("EMA ribbon stacked bearish")
    if c.ema_20_dist_pct is not None and abs(c.ema_20_dist_pct) >= _EMA_DIST_EXTENDED_PCT:
        side = "above" if c.ema_20_dist_pct > 0 else "below"
        notes.append(f"extended {abs(c.ema_20_dist_pct):.1f}% {side} EMA20 — snapback risk")
    if c.funding_rate_pct is not None and abs(c.funding_rate_pct) >= _FUNDING_EXTREME_PCT:
        side = "longs" if c.funding_rate_pct > 0 else "shorts"
        notes.append(f"extreme funding {c.funding_rate_pct:+.3f}%/8h — crowded {side}")
    return notes


def chart_pattern_notes(snapshot: MarketSnapshot) -> str:
    """Deterministic pattern annotations from numeric fields; "" when quiet."""
    lines = []
    for c in snapshot.coins:
        notes = _coin_pattern_notes(c)
        if notes:
            lines.append(f"- {c.symbol}: " + "; ".join(notes))
    if not lines:
        return ""
    return "PATTERN NOTES (deterministic, from the numbers above):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardcoded few-shot exemplars per regime.
# ---------------------------------------------------------------------------

_REGIME_DEMOS: dict[str, str] = {
    "bull": (
        "FEW-SHOT EXEMPLARS (bull regime — calibrated, not euphoric):\n"
        '1. COIN +3.2% 24h, +9% 7d, RSI 62, ADX 31, funding +0.012% -> {"p_long": 0.55, "p_short": 0.15, "p_flat": 0.30, "rationale": "uptrend intact, momentum confirmed, funding sane"}\n'
        '2. COIN +6.0% 24h, +20% 7d, RSI 81, funding +0.09%, L/S 2.6 -> {"p_long": 0.30, "p_short": 0.35, "p_flat": 0.35, "rationale": "overbought and crowded longs despite uptrend"}\n'
        '3. COIN +0.4% 24h, +6% 7d, RSI 55, BB squeeze 3% -> {"p_long": 0.42, "p_short": 0.18, "p_flat": 0.40, "rationale": "squeeze in uptrend favors upside break"}'
    ),
    "bear": (
        "FEW-SHOT EXEMPLARS (bear regime — calibrated, not panicked):\n"
        '1. COIN -3.5% 24h, -11% 7d, RSI 38, ADX 29, funding -0.005% -> {"p_long": 0.15, "p_short": 0.55, "p_flat": 0.30, "rationale": "downtrend intact, momentum confirmed"}\n'
        '2. COIN -7% 24h, -22% 7d, RSI 19, funding -0.08%, L/S 0.6 -> {"p_long": 0.35, "p_short": 0.28, "p_flat": 0.37, "rationale": "oversold, crowded shorts, squeeze risk"}\n'
        '3. COIN -1.0% 24h, -8% 7d, RSI 44, OI rising -> {"p_long": 0.18, "p_short": 0.47, "p_flat": 0.35, "rationale": "weak bounce in downtrend, sellers in control"}'
    ),
    "chop": (
        "FEW-SHOT EXEMPLARS (chop regime — FLAT is often correct):\n"
        '1. COIN +0.3% 24h, -1% 7d, RSI 51, ADX 14 -> {"p_long": 0.25, "p_short": 0.25, "p_flat": 0.50, "rationale": "rangebound, no edge, FLAT favored"}\n'
        '2. COIN +1.8% 24h, +1% 7d, RSI 64, ADX 17 -> {"p_long": 0.28, "p_short": 0.32, "p_flat": 0.40, "rationale": "pop in chop tends to fade"}\n'
        '3. COIN -0.5% 24h, +0.5% 7d, RSI 47, BB squeeze 2.5% -> {"p_long": 0.30, "p_short": 0.30, "p_flat": 0.40, "rationale": "tight squeeze, direction unclear, wait"}'
    ),
}


def regime_demos(regime: str) -> str:
    """Hardcoded few-shot exemplars for "bull" | "bear" | "chop"."""
    return _REGIME_DEMOS.get(regime.strip().lower(), _REGIME_DEMOS["chop"])


# ---------------------------------------------------------------------------
# Local market-data rendering (re-implemented; do NOT import arena.agents.prompts).
# ---------------------------------------------------------------------------


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def _fmt_extra(value: float) -> str:
    if value == int(value) and abs(value) < 1e9:
        return f"{int(value)}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _coin_line(c: CoinSnapshot) -> str:
    line = (
        f"- {c.symbol}: price=${c.price_usd:,.4f} | "
        f"1h={_fmt(c.change_1h_pct, '%')} 24h={_fmt(c.change_24h_pct, '%')} "
        f"7d={_fmt(c.change_7d_pct, '%')} | RSI14={_fmt(c.rsi_14)} | "
        f"EMA20dist={_fmt(c.ema_20_dist_pct, '%')} | vol24h={_fmt(c.volatility_24h_pct, '%')}"
    )
    bits: list[str] = []
    if c.funding_rate_pct is not None:
        bits.append(f"funding8h={c.funding_rate_pct:+.4f}%")
    if c.funding_7d_avg_pct is not None:
        bits.append(f"funding7dAvg={c.funding_7d_avg_pct:+.4f}%")
    if c.open_interest_usd is not None:
        bits.append(f"OI=${c.open_interest_usd / 1e6:,.0f}M")
    if c.long_short_ratio is not None:
        bits.append(f"longShortAcct={c.long_short_ratio:.2f}")
    if c.adx_14 is not None:
        bits.append(f"ADX14(1h)={c.adx_14:.1f}")
    for key in sorted(c.extras)[:MAX_EXTRAS_PER_COIN]:
        bits.append(f"{key}={_fmt_extra(c.extras[key])}")
    if bits:
        line += " | " + " ".join(bits)
    return line


def _market_block(snapshot: MarketSnapshot) -> str:
    parts = [f"MARKET SNAPSHOT as of {snapshot.as_of.isoformat()}:"]
    if snapshot.fear_greed is not None:
        parts.append(f"Fear & Greed index: {snapshot.fear_greed}/100")
    if snapshot.extras:
        kv = " ".join(f"{k}={_fmt_extra(v)}" for k, v in sorted(snapshot.extras.items()))
        parts.append(f"Market extras: {kv}")
    parts.extend(_coin_line(c) for c in snapshot.coins)
    if snapshot.headlines:
        parts.append("RECENT HEADLINES (narrative — see FACTS vs NARRATIVE):")
        for h in snapshot.headlines[:MAX_HEADLINES]:
            parts.append(f"* {h[:MAX_HEADLINE_CHARS]}")
    return "\n".join(parts)


def _output_contract(snapshot: MarketSnapshot, horizon_hours: float) -> str:
    symbols = ", ".join(snapshot.symbols)
    return f"""Task: for EACH coin above, predict the price direction over the next {horizon_hours:g} hours.
- LONG  = you expect the price to rise meaningfully.
- SHORT = you expect the price to fall meaningfully.
- FLAT  = you expect no meaningful move (or you are unsure).

A move bigger than +1% counts as LONG, below -1% counts as SHORT, in between is FLAT.

Respond with STRICT JSON only — no prose, no markdown fences, exactly this shape:
{{"signals": [{{"symbol": "BTC", "p_long": 0.0, "p_short": 0.0, "p_flat": 0.0, "rationale": "..."}}, ...]}}

Rules:
- Exactly one entry per coin, covering every symbol: {symbols}.
- "p_long", "p_short", "p_flat" are your honest probabilities for each outcome; they must sum to 1.0.
- You are scored with a strictly proper rule (Brier): reporting your true probabilities maximizes your expected score; overconfidence is punished.
- "rationale" is a brief (max ~20 words) justification."""


# ---------------------------------------------------------------------------
# Composition.
# ---------------------------------------------------------------------------


def build_enhanced_prompt(
    snapshot: MarketSnapshot,
    *,
    role: str = "generalist",
    memory_lessons: list[str] | None = None,
    debate_brief: str = "",
    preamble: str = "",
) -> str:
    """Compose the full enhanced prompt for one agent role.

    Order: persona -> optional optimizer preamble -> FinCoT blueprint ->
    fact/narrative split -> market data (rendered locally) -> pattern notes ->
    regime few-shots -> memory lessons (max 8) -> debate brief -> the strict
    JSON probability output contract shared with ``arena.agents.prompts``.
    """
    lessons = list(memory_lessons or [])
    blocks: list[str] = [persona_preamble(role)]
    if preamble.strip():
        blocks.append(preamble.strip())
    blocks.append(fincot_blueprint(snapshot))
    blocks.append(fact_subjectivity_split())
    blocks.append(_market_block(snapshot))
    patterns = chart_pattern_notes(snapshot)
    if patterns:
        blocks.append(patterns)
    regime = detect_regime(snapshot)
    blocks.append(f"DETECTED REGIME: {regime}")
    blocks.append(regime_demos(regime))
    if lessons:
        rendered = "\n".join(
            f"- {lesson[:MAX_LESSON_CHARS]}" for lesson in lessons[:MAX_MEMORY_LESSONS]
        )
        blocks.append("LESSONS FROM PAST ROUNDS (your own track record):\n" + rendered)
    if debate_brief.strip():
        blocks.append(debate_brief.strip()[:MAX_DEBATE_CHARS])
    blocks.append(_output_contract(snapshot, ArenaSettings().horizon_hours))
    return "\n\n".join(blocks)
