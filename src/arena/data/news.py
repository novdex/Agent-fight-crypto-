"""Headline enrichment (RAG-lite) and event detection."""

from __future__ import annotations

import sys
from typing import Any

import httpx

from arena.models import MarketSnapshot

CRYPTOPANIC = "https://cryptopanic.com/api/v1/posts/?public=true"
COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"
_TIMEOUT_S = 15.0
_MAX_LEN = 120

#: Keyword -> event tag for special "event round" awareness (improvement #11).
EVENT_KEYWORDS: dict[str, str] = {
    "etf": "ETF_FLOW",
    "sec": "REGULATORY",
    "lawsuit": "REGULATORY",
    "hack": "SECURITY_INCIDENT",
    "exploit": "SECURITY_INCIDENT",
    "fomc": "MACRO_EVENT",
    "cpi": "MACRO_EVENT",
    "rate cut": "MACRO_EVENT",
    "unlock": "TOKEN_UNLOCK",
    "halving": "SUPPLY_EVENT",
    "upgrade": "PROTOCOL_UPGRADE",
    "hard fork": "PROTOCOL_UPGRADE",
}


def detect_events(headlines: list[str]) -> list[str]:
    """Distinct event tags found in the headlines (order of first appearance)."""
    tags: list[str] = []
    for h in headlines:
        low = h.lower()
        for kw, tag in EVENT_KEYWORDS.items():
            if kw in low and tag not in tags:
                tags.append(tag)
    return tags


def _parse_cryptopanic(payload: Any) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for post in payload.get("results", []):
            title = str(post.get("title", "")).strip()
            if title:
                out.append(title[:_MAX_LEN])
    return out


def _parse_trending(payload: Any) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for item in payload.get("coins", []):
            name = (item.get("item") or {}).get("name")
            if name:
                out.append(f"Trending: {name}"[:_MAX_LEN])
    return out


def fetch_headlines(limit: int = 8) -> list[str]:
    """Recent crypto headlines, best-effort; [] when no source is reachable."""
    for url, parse, source in (
        (CRYPTOPANIC, _parse_cryptopanic, "cryptopanic"),
        (COINGECKO_TRENDING, _parse_trending, "coingecko_trending"),
    ):
        try:
            with httpx.Client(timeout=_TIMEOUT_S) as client:
                resp = client.get(url)
                resp.raise_for_status()
                headlines = parse(resp.json())
                if headlines:
                    return headlines[:limit]
        except Exception as exc:
            print(f"warning: news source {source!r} unavailable ({exc})", file=sys.stderr)
    return []


REDDIT_NEW = "https://www.reddit.com/r/CryptoCurrency/new.json?limit=100"


def social_mention_counts(titles: list[str], symbols: list[str]) -> dict[str, int]:
    """Count symbol mentions across post titles (whole-word, case-insensitive)."""
    import re

    counts: dict[str, int] = {}
    joined = "\n".join(titles).upper()
    for sym in symbols:
        s = sym.upper()
        if len(s) < 2:
            continue
        counts[sym] = len(re.findall(rf"\b{re.escape(s)}\b", joined))
    return counts


def fetch_social_titles() -> list[str]:
    """Recent r/CryptoCurrency post titles (free, no key); [] on failure."""
    try:
        with httpx.Client(
            timeout=_TIMEOUT_S, headers={"User-Agent": "crypto-llm-arena/0.1"}
        ) as client:
            resp = client.get(REDDIT_NEW)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
            return [
                str(c.get("data", {}).get("title", "")) for c in children if c
            ]
    except Exception as exc:
        print(f"warning: social source 'reddit' unavailable ({exc})", file=sys.stderr)
        return []


def enrich_news(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Attach headlines, event tags and social mention counts; never raises."""
    headlines = fetch_headlines()
    events = detect_events(headlines)
    if events:
        headlines = [f"[EVENTS: {', '.join(events)}]"] + headlines
    coins = snapshot.coins
    titles = fetch_social_titles()
    if titles:
        # Social-volume anomaly proxy (improvement #60): raw mention counts;
        # the LLMs see spikes relative to the rest of the board.
        counts = social_mention_counts(titles, [c.symbol for c in coins])
        coins = [c.model_copy(deep=True) for c in coins]
        for coin in coins:
            n = counts.get(coin.symbol, 0)
            if n > 0:
                coin.extras["social_mentions"] = float(n)
    return snapshot.model_copy(update={"headlines": headlines[:9], "coins": coins})
