"""Prompt construction and robust signal parsing for LLM agents.

FROZEN INTERFACE — see docs/INTERFACES.md (Unit 2).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from arena.models import ArenaSettings, CoinSnapshot, Direction, MarketSnapshot, Signal

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\n?|```", re.MULTILINE)


def _fmt(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def _coin_line(c: CoinSnapshot) -> str:
    return (
        f"- {c.symbol}: price=${c.price_usd:,.4f} | "
        f"1h={_fmt(c.change_1h_pct, '%')} 24h={_fmt(c.change_24h_pct, '%')} "
        f"7d={_fmt(c.change_7d_pct, '%')} | RSI14={_fmt(c.rsi_14)} | "
        f"EMA20dist={_fmt(c.ema_20_dist_pct, '%')} | vol24h={_fmt(c.volatility_24h_pct, '%')}"
    )


def build_prompt(snapshot: MarketSnapshot) -> str:
    """Build the trading-signal prompt presenting each coin's stats compactly."""
    horizon_hours = ArenaSettings().horizon_hours
    coin_lines = "\n".join(_coin_line(c) for c in snapshot.coins)
    symbols = ", ".join(snapshot.symbols)
    return f"""You are a crypto trading-signal agent competing in an arena.

Market snapshot as of {snapshot.as_of.isoformat()}:
{coin_lines}

Task: for EACH coin above, predict the price direction over the next {horizon_hours:g} hours.
- LONG  = you expect the price to rise meaningfully.
- SHORT = you expect the price to fall meaningfully.
- FLAT  = you expect no meaningful move (or you are unsure).

Respond with STRICT JSON only — no prose, no markdown fences, exactly this shape:
{{"signals": [{{"symbol": "BTC", "direction": "LONG|SHORT|FLAT", "confidence": 0.0, "rationale": "..."}}, ...]}}

Rules:
- Exactly one entry per coin, covering every symbol: {symbols}.
- "direction" must be one of LONG, SHORT, FLAT.
- "confidence" is a number between 0.0 and 1.0.
- "rationale" is a brief (max ~20 words) justification.
"""


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Strip markdown fences and extract the outermost {...} JSON object."""
    cleaned = _FENCE_RE.sub("", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _clamp(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf != conf:  # NaN
        return 0.0
    return max(0.0, min(1.0, conf))


def _flat_signal(agent_name: str, symbol: str, price: float) -> Signal:
    return Signal(
        agent=agent_name,
        symbol=symbol,
        direction=Direction.FLAT,
        confidence=0.0,
        rationale="",
        price_at_signal=price,
    )


def parse_signals(text: str, agent_name: str, snapshot: MarketSnapshot) -> list[Signal]:
    """Robustly parse an LLM reply into one Signal per snapshot coin.

    - Strips code fences and extracts the outermost JSON object.
    - Clamps confidence to [0, 1]; drops unknown symbols.
    - Fills price_at_signal from the snapshot.
    - Backfills any missing symbols with FLAT/0.0.
    - A completely unparseable reply yields all-FLAT signals.
    """
    prices = {c.symbol.upper(): c.price_usd for c in snapshot.coins}
    parsed: dict[str, Signal] = {}

    obj = _extract_json_object(text)
    entries = obj.get("signals") if obj else None
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol", "")).upper().strip()
            if symbol not in prices:
                continue  # unknown symbol — drop
            raw_dir = str(entry.get("direction", "")).upper().strip()
            try:
                direction = Direction(raw_dir)
            except ValueError:
                direction = Direction.FLAT
            parsed[symbol] = Signal(
                agent=agent_name,
                symbol=symbol,
                direction=direction,
                confidence=_clamp(entry.get("confidence")),
                rationale=str(entry.get("rationale", "") or ""),
                price_at_signal=prices[symbol],
            )

    # Backfill missing symbols with FLAT/0.0 so every agent covers every coin.
    signals: list[Signal] = []
    for coin in snapshot.coins:
        sym = coin.symbol.upper()
        signals.append(parsed.get(sym) or _flat_signal(agent_name, sym, coin.price_usd))
    return signals
