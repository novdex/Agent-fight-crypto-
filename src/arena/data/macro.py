"""Macro context enrichment: BTC dominance, ETH/BTC, DXY/SPX (best-effort)."""

from __future__ import annotations

import sys
from typing import Optional

import httpx

from arena.models import MarketSnapshot

COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
STOOQ_CSV = "https://stooq.com/q/l/"
_TIMEOUT_S = 15.0


def parse_stooq_change_pct(csv_text: str) -> Optional[float]:
    """Daily % change from a stooq quote CSV line (open..close columns)."""
    lines = csv_text.strip().splitlines()
    if len(lines) < 2:
        return None
    cols = lines[1].split(",")
    try:
        open_, close = float(cols[3]), float(cols[6])
    except (IndexError, ValueError):
        return None
    if open_ <= 0:
        return None
    return 100.0 * (close - open_) / open_


def _warn_once(warned: set[str], source: str, exc: Exception) -> None:
    if source not in warned:
        warned.add(source)
        print(f"warning: macro source {source!r} unavailable ({exc})", file=sys.stderr)


def enrich_macro(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Fill MarketSnapshot.extras with macro context; never raises."""
    warned: set[str] = set()
    extras = dict(snapshot.extras)

    btc = snapshot.coin("BTC")
    eth = snapshot.coin("ETH")
    if btc and eth and btc.price_usd:
        extras["eth_btc_ratio"] = round(eth.price_usd / btc.price_usd, 6)

    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            try:
                g = client.get(COINGECKO_GLOBAL)
                g.raise_for_status()
                dom = g.json()["data"]["market_cap_percentage"]["btc"]
                extras["btc_dominance_pct"] = round(float(dom), 2)
            except Exception as exc:
                _warn_once(warned, "coingecko_global", exc)
            for key, ticker in (("dxy_change_pct", "dx.f"), ("spx_change_pct", "^spx")):
                try:
                    r = client.get(
                        STOOQ_CSV, params={"s": ticker, "f": "sd2t2ohlcv", "h": "", "e": "csv"}
                    )
                    r.raise_for_status()
                    chg = parse_stooq_change_pct(r.text)
                    if chg is not None:
                        extras[key] = round(chg, 3)
                except Exception as exc:
                    _warn_once(warned, f"stooq:{ticker}", exc)
    except Exception as exc:
        _warn_once(warned, "macro_enrichment", exc)

    return snapshot.model_copy(update={"extras": extras})
