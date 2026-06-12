"""Perp-specific snapshot enrichment (Binance futures + Fear & Greed).

Best-effort by design: every fetch failure degrades to ``None`` fields and a
single stderr warning per source — a data outage must never block a round.
Coins without a USDT perpetual simply keep ``None`` perp fields.

Sources (all free, no key):
- Binance USD-M futures: funding rate (current + 7d history), open interest,
  global long/short account ratio, 1h klines (for ADX).
- alternative.me: crypto Fear & Greed index.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Sequence

import httpx

from arena.models import MarketSnapshot

BINANCE_FAPI = "https://fapi.binance.com"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

_TIMEOUT_S = 15.0

#: Funding payments per 7 days (every 8h).
_FUNDING_7D_LIMIT = 21


# ---------------------------------------------------------------------------
# Pure indicator / parsing helpers (unit-testable, no network)
# ---------------------------------------------------------------------------


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> Optional[float]:
    """Wilder's Average Directional Index from OHLC series.

    Needs at least ``2 * period`` bars; returns ``None`` otherwise.
    ADX > 25 marks a trending market, < 20 a choppy one.
    """
    n = min(len(highs), len(lows), len(closes))
    if n < 2 * period + 1 or period <= 0:
        return None

    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )

    # Wilder smoothing seeds: plain sums of the first `period` values.
    atr = sum(trs[:period])
    p_dm = sum(plus_dm[:period])
    m_dm = sum(minus_dm[:period])

    dxs: list[float] = []
    for i in range(period, len(trs)):
        atr = atr - atr / period + trs[i]
        p_dm = p_dm - p_dm / period + plus_dm[i]
        m_dm = m_dm - m_dm / period + minus_dm[i]
        if atr <= 0:
            continue
        di_plus = 100.0 * p_dm / atr
        di_minus = 100.0 * m_dm / atr
        di_sum = di_plus + di_minus
        if di_sum > 0:
            dxs.append(100.0 * abs(di_plus - di_minus) / di_sum)

    if len(dxs) < period:
        return None
    adx_val = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx_val = (adx_val * (period - 1) + dx) / period
    return adx_val


def parse_funding_history(rows: list[dict[str, Any]]) -> Optional[float]:
    """Mean funding rate (% per 8h) from a /fundingRate history response."""
    rates = []
    for row in rows:
        try:
            rates.append(float(row["fundingRate"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not rates:
        return None
    return 100.0 * sum(rates) / len(rates)


def parse_klines_ohlc(rows: list[list[Any]]) -> tuple[list[float], list[float], list[float]]:
    """(highs, lows, closes) from a Binance klines response."""
    highs, lows, closes = [], [], []
    for row in rows:
        try:
            highs.append(float(row[2]))
            lows.append(float(row[3]))
            closes.append(float(row[4]))
        except (IndexError, TypeError, ValueError):
            continue
    return highs, lows, closes


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def _warn_once(warned: set[str], source: str, exc: Exception) -> None:
    if source not in warned:
        warned.add(source)
        print(
            f"warning: perp enrichment source {source!r} unavailable ({exc}); "
            "continuing without it",
            file=sys.stderr,
        )


def fetch_fear_greed(client: httpx.Client) -> Optional[int]:
    """Current crypto Fear & Greed index (0-100) from alternative.me."""
    resp = client.get(FEAR_GREED_URL, timeout=_TIMEOUT_S)
    resp.raise_for_status()
    return int(resp.json()["data"][0]["value"])


def enrich_snapshot(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Return a copy of ``snapshot`` with perp fields filled best-effort.

    Network failures (or symbols without a USDT perp) leave fields ``None``;
    each failing source warns once on stderr. Never raises.
    """
    warned: set[str] = set()
    coins = [c.model_copy() for c in snapshot.coins]
    fear_greed = snapshot.fear_greed

    try:
        with httpx.Client() as client:
            try:
                fear_greed = fetch_fear_greed(client)
            except Exception as exc:
                _warn_once(warned, "fear_greed", exc)

            for coin in coins:
                pair = f"{coin.symbol.upper()}USDT"

                def _get(path: str, source: str, **params: Any) -> Optional[Any]:
                    try:
                        resp = client.get(
                            f"{BINANCE_FAPI}{path}",
                            params={"symbol": pair, **params},
                            timeout=_TIMEOUT_S,
                        )
                        if resp.status_code == 400:
                            return None  # no such perp pair — not an outage
                        resp.raise_for_status()
                        return resp.json()
                    except Exception as exc:
                        _warn_once(warned, source, exc)
                        return None

                premium = _get("/fapi/v1/premiumIndex", "funding_current")
                if isinstance(premium, dict):
                    try:
                        coin.funding_rate_pct = 100.0 * float(premium["lastFundingRate"])
                    except (KeyError, TypeError, ValueError):
                        pass

                history = _get(
                    "/fapi/v1/fundingRate", "funding_history", limit=_FUNDING_7D_LIMIT
                )
                if isinstance(history, list):
                    coin.funding_7d_avg_pct = parse_funding_history(history)

                oi = _get("/fapi/v1/openInterest", "open_interest")
                if isinstance(oi, dict):
                    try:
                        coin.open_interest_usd = float(oi["openInterest"]) * coin.price_usd
                    except (KeyError, TypeError, ValueError):
                        pass

                ls = _get(
                    "/futures/data/globalLongShortAccountRatio",
                    "long_short_ratio",
                    period="1h",
                    limit=1,
                )
                if isinstance(ls, list) and ls:
                    try:
                        coin.long_short_ratio = float(ls[-1]["longShortRatio"])
                    except (KeyError, TypeError, ValueError):
                        pass

                klines = _get("/fapi/v1/klines", "klines", interval="1h", limit=100)
                if isinstance(klines, list):
                    highs, lows, closes = parse_klines_ohlc(klines)
                    coin.adx_14 = adx(highs, lows, closes, 14)
                    try:
                        # Fast multi-timeframe sub-team votes (improvement #18)
                        # + extra 1h technicals, from the same kline fetch.
                        from arena.agents.subteam import subteam_votes
                        from arena.data.technicals import (
                            bollinger_width_pct,
                            ema_ribbon_slope,
                            hurst_exponent,
                            macd_histogram,
                            stoch_rsi,
                        )

                        for key, vote in subteam_votes(closes).items():
                            coin.extras[key] = float(vote)
                        for key, fn in (
                            ("hurst", hurst_exponent),
                            ("macd_hist", macd_histogram),
                            ("stoch_rsi", stoch_rsi),
                            ("bb_width_pct", bollinger_width_pct),
                            ("ema_ribbon", ema_ribbon_slope),
                        ):
                            value = fn(closes)
                            if value is not None:
                                coin.extras[key] = round(float(value), 4)
                    except Exception:
                        pass
    except Exception as exc:  # client construction or anything unforeseen
        _warn_once(warned, "perp_enrichment", exc)

    return MarketSnapshot(as_of=snapshot.as_of, coins=coins, fear_greed=fear_greed)
