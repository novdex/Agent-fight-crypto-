"""Deterministic multi-timeframe technical sub-team (improvement #18).

QuantAgent-style: three fast "specialists" — indicator, trend and pattern —
each vote LONG(+1)/FLAT(0)/SHORT(-1) from 1h closes. Their votes land in
``CoinSnapshot.extras`` and the prompt, giving the 24h fighters a faster
timeframe's read without extra LLM calls.
"""

from __future__ import annotations

from typing import Optional, Sequence

from arena.data.technicals import (
    bollinger_width_pct,
    ema_ribbon_slope,
    macd_histogram,
    stoch_rsi,
)
from arena.models import MarketSnapshot


def indicator_vote(closes: Sequence[float]) -> Optional[int]:
    """Oscillator specialist: Stoch-RSI extremes + MACD histogram sign."""
    s = stoch_rsi(list(closes))
    m = macd_histogram(list(closes))
    if s is None or m is None:
        return None
    if s < 20 and m > 0:
        return 1
    if s > 80 and m < 0:
        return -1
    return 0


def trend_vote(closes: Sequence[float]) -> Optional[int]:
    """Trend specialist: EMA-ribbon posture."""
    slope = ema_ribbon_slope(list(closes))
    if slope is None:
        return None
    if slope > 0.5:
        return 1
    if slope < -0.5:
        return -1
    return 0


def pattern_vote(closes: Sequence[float]) -> Optional[int]:
    """Pattern specialist: Bollinger squeeze breakout direction."""
    width = bollinger_width_pct(list(closes))
    if width is None or len(closes) < 25:
        return None
    if width > 4.0:  # no squeeze, no edge
        return 0
    recent = closes[-3:]
    ma20 = sum(closes[-20:]) / 20.0
    if min(recent) > ma20:
        return 1
    if max(recent) < ma20:
        return -1
    return 0


def subteam_votes(closes: Sequence[float]) -> dict[str, int]:
    """All three specialists' votes; specialists without data abstain."""
    votes: dict[str, int] = {}
    for name, fn in (
        ("sub_indicator", indicator_vote),
        ("sub_trend", trend_vote),
        ("sub_pattern", pattern_vote),
    ):
        v = fn(closes)
        if v is not None:
            votes[name] = v
    return votes


def enrich_subteam(
    snapshot: MarketSnapshot, closes_by_symbol: dict[str, Sequence[float]]
) -> MarketSnapshot:
    """Write sub-team votes into each coin's extras (floats: -1/0/+1)."""
    coins = [c.model_copy(deep=True) for c in snapshot.coins]
    for coin in coins:
        closes = closes_by_symbol.get(coin.symbol)
        if not closes:
            continue
        for key, vote in subteam_votes(closes).items():
            coin.extras[key] = float(vote)
    return snapshot.model_copy(update={"coins": coins})
