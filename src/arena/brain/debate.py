"""Bull/bear debate brief and a deterministic critic fact-checker.

`debate_context` runs one bull-case and one bear-case pass over the market
snapshot through an injected ``ask`` callable and packs both into a compact
"DEBATE BRIEF" block to prepend to agent prompts.

`critic_review` is a zero-LLM fact check: it attenuates signals whose
rationale contradicts the snapshot numbers (e.g. claims "oversold" while
RSI > 70) by halving confidence and annotating the rationale. It never drops
coverage — output has exactly one signal per input signal.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from arena.models import CoinSnapshot, MarketSnapshot, Signal

MAX_BRIEF_CHARS = 2000
_SIDE_CHARS = 750
_DIGEST_COINS = 10

_BULL_PROMPT = """\
You are the BULL advocate in a trading debate. Given the market snapshot
below, argue the strongest honest case for upside over the next 24 hours.
Cite concrete numbers from the data. Max 120 words, plain text.

{digest}
"""

_BEAR_PROMPT = """\
You are the BEAR advocate in a trading debate. Given the market snapshot
below, argue the strongest honest case for downside over the next 24 hours.
Cite concrete numbers from the data. Max 120 words, plain text.

{digest}
"""


def _fmt(value: Optional[float], suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _snapshot_digest(snapshot: MarketSnapshot, max_coins: int = _DIGEST_COINS) -> str:
    """Compact per-coin stat lines used inside the debate prompts."""
    lines = []
    for c in snapshot.coins[:max_coins]:
        lines.append(
            f"- {c.symbol}: ${c.price_usd:,.4f} | 24h={_fmt(c.change_24h_pct, '%')} "
            f"7d={_fmt(c.change_7d_pct, '%')} | RSI14={_fmt(c.rsi_14)} | "
            f"funding8h={_fmt(c.funding_rate_pct, '%')} | ADX14={_fmt(c.adx_14)}"
        )
    if snapshot.fear_greed is not None:
        lines.append(f"Fear & Greed index: {snapshot.fear_greed}/100")
    return "\n".join(lines)


def _squash(text: str, limit: int) -> str:
    """Collapse whitespace and truncate to `limit` chars (ellipsis marker)."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def debate_context(snapshot: MarketSnapshot, ask: Callable[[str], str]) -> str:
    """Run a bull pass and a bear pass via `ask`; return a <=2000-char brief."""
    digest = _snapshot_digest(snapshot)
    bull = _squash(ask(_BULL_PROMPT.format(digest=digest)), _SIDE_CHARS)
    bear = _squash(ask(_BEAR_PROMPT.format(digest=digest)), _SIDE_CHARS)
    brief = (
        "DEBATE BRIEF (weigh both sides before committing to probabilities):\n"
        f"BULL CASE: {bull or 'n/a'}\n"
        f"BEAR CASE: {bear or 'n/a'}\n"
        "Do not anchor on either side; let the data decide."
    )
    return brief[:MAX_BRIEF_CHARS]


# --- deterministic critic -------------------------------------------------

# Each check: (claim regex, contradiction predicate, note builder).
_Check = tuple[
    "re.Pattern[str]",
    Callable[[CoinSnapshot], bool],
    Callable[[CoinSnapshot], str],
]

_CHECKS: list[_Check] = [
    (
        re.compile(r"\boversold\b"),
        lambda c: c.rsi_14 is not None and c.rsi_14 > 70,
        lambda c: f"claims oversold but RSI14={c.rsi_14:.0f}",
    ),
    (
        re.compile(r"\boverbought\b"),
        lambda c: c.rsi_14 is not None and c.rsi_14 < 30,
        lambda c: f"claims overbought but RSI14={c.rsi_14:.0f}",
    ),
    (
        re.compile(r"negative funding|shorts\s+(?:are\s+)?paying"),
        lambda c: c.funding_rate_pct is not None and c.funding_rate_pct > 0.005,
        lambda c: f"claims negative funding but funding8h={c.funding_rate_pct:+.4f}%",
    ),
    (
        re.compile(r"positive funding|longs\s+(?:are\s+)?paying"),
        lambda c: c.funding_rate_pct is not None and c.funding_rate_pct < -0.005,
        lambda c: f"claims positive funding but funding8h={c.funding_rate_pct:+.4f}%",
    ),
    (
        re.compile(r"strong(?:ly)?\s+trend|\btrending\b"),
        lambda c: c.adx_14 is not None and c.adx_14 < 20,
        lambda c: f"claims trending but ADX14={c.adx_14:.1f} (chop)",
    ),
    (
        re.compile(r"\bchop(?:py)?\b|range[\s-]?bound|\bsideways\b"),
        lambda c: c.adx_14 is not None and c.adx_14 > 30,
        lambda c: f"claims chop but ADX14={c.adx_14:.1f} (trending)",
    ),
    (
        re.compile(r"\brally(?:ing)?\b|\bsurg(?:e|ing)\b|\bpump(?:ing)?\b"),
        lambda c: c.change_24h_pct is not None and c.change_24h_pct < -2.0,
        lambda c: f"claims rally but 24h={c.change_24h_pct:+.2f}%",
    ),
    (
        re.compile(r"\bdump(?:ing)?\b|\bsell-?off\b|\bcrash(?:ing)?\b"),
        lambda c: c.change_24h_pct is not None and c.change_24h_pct > 2.0,
        lambda c: f"claims selloff but 24h={c.change_24h_pct:+.2f}%",
    ),
]


def _contradictions(rationale: str, coin: CoinSnapshot) -> list[str]:
    low = rationale.lower()
    return [
        note(coin)
        for claim, contradicts, note in _CHECKS
        if claim.search(low) and contradicts(coin)
    ]


def critic_review(signals: list[Signal], snapshot: MarketSnapshot) -> list[Signal]:
    """Attenuate signals whose rationale contradicts snapshot data.

    Deterministic, no LLM. On contradiction: confidence is halved, the
    rationale gains a "[critic: ...]" note and any probability vector is
    blended 50/50 toward uniform (argmax direction is preserved). Signals
    without contradictions pass through unchanged; coverage never shrinks.
    """
    reviewed: list[Signal] = []
    for sig in signals:
        coin = snapshot.coin(sig.symbol)
        notes = _contradictions(sig.rationale, coin) if coin and sig.rationale else []
        if not notes:
            reviewed.append(sig)
            continue
        updates: dict[str, object] = {
            "confidence": sig.confidence / 2.0,
            "rationale": (sig.rationale + " [critic: " + "; ".join(notes) + "]").strip(),
        }
        if sig.has_probs:
            # 50/50 blend with the uniform vector keeps the argmax direction
            # while pulling the stated probabilities toward ignorance.
            uniform = 1.0 / 3.0
            updates["p_long"] = 0.5 * float(sig.p_long or 0.0) + 0.5 * uniform
            updates["p_short"] = 0.5 * float(sig.p_short or 0.0) + 0.5 * uniform
            updates["p_flat"] = 0.5 * float(sig.p_flat or 0.0) + 0.5 * uniform
        reviewed.append(sig.model_copy(update=updates))
    return reviewed
