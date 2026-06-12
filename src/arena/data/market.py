"""Unit 1 — Market data layer (CoinGecko client + offline fixtures).

Implements the frozen interface from docs/INTERFACES.md:

    fetch_top_coins(n=20, *, offline=False) -> MarketSnapshot
    fetch_prices(symbols, *, offline=False) -> dict[str, float]

Online mode hits the CoinGecko free API (no key). Offline mode reads the
hand-written fixtures ``tests/fixtures/market_snapshot.json`` and
``tests/fixtures/prices_later.json``.

Fixture path resolution: the CLI runs from the repo root, so we first try
``<cwd>/tests/fixtures/<name>``; if that does not exist we walk up from this
module's own location (``Path(__file__).resolve()``) looking for a
``tests/fixtures`` directory. This keeps offline mode working both for an
editable install run from the repo root and for tests run from anywhere
inside the repo.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

from arena.models import CoinSnapshot, MarketSnapshot

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

#: Symbols excluded from the arena — pegged assets produce degenerate signals.
STABLECOINS: frozenset[str] = frozenset(
    {"USDT", "USDC", "DAI", "FDUSD", "USDE", "TUSD", "PYUSD", "USDS", "BUSD"}
)

#: Extra rows requested beyond ``n`` so that n real coins remain after
#: stablecoin filtering. There are 9 known stablecoins; a few spare rows
#: cover newly listed pegged assets sneaking into the top ranks.
_EXTRA_ROWS = 15

_TIMEOUT_S = 30.0

SNAPSHOT_FIXTURE = "market_snapshot.json"
PRICES_FIXTURE = "prices_later.json"


# ---------------------------------------------------------------------------
# Pure indicator helpers (module-level so they are unit-testable)
# ---------------------------------------------------------------------------


def ema(values: Sequence[float], period: int) -> Optional[float]:
    """Exponential moving average of ``values`` (last value of the series).

    Seeded with the SMA of the first ``period`` values, then smoothed with
    k = 2 / (period + 1). Returns ``None`` if there are fewer than ``period``
    values.
    """
    if len(values) < period or period <= 0:
        return None
    k = 2.0 / (period + 1.0)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1.0 - k)
    return e


def ema_distance_pct(values: Sequence[float], period: int = 20) -> Optional[float]:
    """% distance of the last price from the ``period``-EMA of ``values``."""
    e = ema(values, period)
    if e is None or e == 0 or not values:
        return None
    return (values[-1] - e) / e * 100.0


def rsi(values: Sequence[float], period: int = 14) -> Optional[float]:
    """Wilder's RSI over ``values`` (one value per hour for hourly RSI).

    Returns ``None`` with fewer than ``period + 1`` values. A flat series
    (no gains, no losses) conventionally returns 50.0; a series with no
    losses returns 100.0.
    """
    if len(values) < period + 1 or period <= 0:
        return None
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0.0:
        return 50.0 if avg_gain == 0.0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def volatility_pct(values: Sequence[float], window: int = 24) -> Optional[float]:
    """Stdev of the last ``window`` simple hourly returns of ``values``, x100.

    Uses the sample standard deviation. Returns ``None`` if fewer than two
    returns can be computed (i.e. fewer than three prices).
    """
    returns: list[float] = []
    for i in range(len(values) - 1):
        prev = values[i]
        if prev:  # skip zero prices to avoid division by zero
            returns.append((values[i + 1] - prev) / prev)
    returns = returns[-window:]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * 100.0


# ---------------------------------------------------------------------------
# Row parsing / snapshot construction
# ---------------------------------------------------------------------------


def is_stablecoin(symbol: str) -> bool:
    """True if ``symbol`` (any case) is a known stablecoin ticker."""
    return symbol.upper() in STABLECOINS


def _coin_from_row(row: dict[str, Any]) -> CoinSnapshot:
    """Build a CoinSnapshot from one CoinGecko /coins/markets row."""
    sparkline = (row.get("sparkline_in_7d") or {}).get("price") or []
    prices = [float(p) for p in sparkline if p is not None]
    has_spark = len(prices) > 0
    return CoinSnapshot(
        symbol=str(row.get("symbol", "")).upper(),
        name=str(row.get("name", "")),
        price_usd=float(row["current_price"]),
        market_cap=row.get("market_cap"),
        volume_24h=row.get("total_volume"),
        change_1h_pct=row.get("price_change_percentage_1h_in_currency"),
        change_24h_pct=row.get("price_change_percentage_24h_in_currency"),
        change_7d_pct=row.get("price_change_percentage_7d_in_currency"),
        rsi_14=rsi(prices, 14) if has_spark else None,
        ema_20_dist_pct=ema_distance_pct(prices, 20) if has_spark else None,
        volatility_24h_pct=volatility_pct(prices, 24) if has_spark else None,
    )


def build_snapshot(
    rows: list[dict[str, Any]], n: int, as_of: Optional[datetime] = None
) -> MarketSnapshot:
    """Filter stablecoins out of raw market rows and build a MarketSnapshot.

    Keeps at most ``n`` coins, preserving the (market-cap) order of ``rows``.
    Rows without a usable price are skipped.
    """
    coins: list[CoinSnapshot] = []
    for row in rows:
        if len(coins) >= n:
            break
        sym = str(row.get("symbol", "")).upper()
        if not sym or is_stablecoin(sym):
            continue
        if row.get("current_price") is None:
            continue
        coins.append(_coin_from_row(row))
    return MarketSnapshot(as_of=as_of or datetime.now(timezone.utc), coins=coins)


# ---------------------------------------------------------------------------
# Fixture handling
# ---------------------------------------------------------------------------


def _fixture_path(name: str) -> Path:
    """Locate a fixture file. See module docstring for the resolution order."""
    cwd_candidate = Path.cwd() / "tests" / "fixtures" / name
    if cwd_candidate.is_file():
        return cwd_candidate
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tests" / "fixtures" / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"Offline fixture {name!r} not found. Looked in {cwd_candidate} and in "
        "tests/fixtures/ directories above the arena package. Run from the "
        "repo root or restore the fixture files."
    )


def _load_fixture(name: str) -> Any:
    path = _fixture_path(name)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read offline fixture {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
    """GET /coins/markets with error mapping to RuntimeError."""
    url = f"{COINGECKO_BASE}/coins/markets"
    try:
        resp = httpx.get(url, params=params, timeout=_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"CoinGecko returned HTTP {exc.response.status_code} for {url}. "
            "The free API is rate-limited (~30 req/min); wait and retry, or "
            "use --offline."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Network error talking to CoinGecko ({url}): {exc}. "
            "Check connectivity or use --offline."
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CoinGecko response shape from {url}: {data!r}")
    return data


def fetch_top_coins(n: int = 20, *, offline: bool = False) -> MarketSnapshot:
    """Top ``n`` non-stablecoin coins by market cap, with hourly indicators.

    Online: CoinGecko ``/coins/markets`` with the 7d hourly sparkline, from
    which ``rsi_14``, ``ema_20_dist_pct`` and ``volatility_24h_pct`` are
    computed (``None`` when the sparkline is missing). Offline: reads the
    serialized MarketSnapshot fixture and returns its first ``n`` coins.
    """
    if offline:
        snap = MarketSnapshot.model_validate(_load_fixture(SNAPSHOT_FIXTURE))
        return MarketSnapshot(as_of=snap.as_of, coins=snap.coins[:n])

    rows = _get_markets(
        {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(250, n + _EXTRA_ROWS),
            "page": 1,
            "sparkline": "true",
            "price_change_percentage": "1h,24h,7d",
        }
    )
    return build_snapshot(rows, n)


def fetch_prices(symbols: list[str], *, offline: bool = False) -> dict[str, float]:
    """Current USD price for each requested symbol (uppercase keys).

    Symbols that cannot be resolved are silently omitted from the result.
    Online lookups use the same ``/coins/markets`` endpoint (top 250 by
    market cap); when multiple listings share a ticker the highest-cap one
    wins. Offline mode reads ``tests/fixtures/prices_later.json``.
    """
    wanted = [s.upper() for s in symbols]
    if offline:
        raw = _load_fixture(PRICES_FIXTURE)
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"Fixture {PRICES_FIXTURE} must be a JSON object of "
                "{symbol: price}."
            )
        data = {str(k).upper(): float(v) for k, v in raw.items()}
        return {s: data[s] for s in wanted if s in data}

    rows = _get_markets(
        {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
        }
    )
    prices: dict[str, float] = {}
    for row in rows:  # rows are cap-descending, so first hit wins
        sym = str(row.get("symbol", "")).upper()
        price = row.get("current_price")
        if sym in wanted and sym not in prices and price is not None:
            prices[sym] = float(price)
    return prices
