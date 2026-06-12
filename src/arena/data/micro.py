"""Orderbook microstructure + derivatives enrichment (best-effort, free APIs).

Fills ``CoinSnapshot.extras``: ``ob_imbalance`` (bid share of top-of-book
depth), ``taker_buy_ratio`` (aggressor buy share of recent trades),
``basis_pct`` (perp mark vs spot), and ``iv_dvol`` (Deribit implied-vol index,
BTC/ETH only). Follows the perp.py pattern: pure parsers, warn-once fetchers,
never raises.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import httpx

from arena.models import MarketSnapshot

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"
DERIBIT = "https://www.deribit.com"
_TIMEOUT_S = 15.0


def parse_depth_imbalance(depth: dict, levels: int = 5) -> Optional[float]:
    """Bid share of total displayed size over the top ``levels``: in [0, 1].

    > 0.6 = bid-heavy book (short-horizon upward pressure), < 0.4 = ask-heavy.
    """
    try:
        bids = sum(float(q) for _, q in depth.get("bids", [])[:levels])
        asks = sum(float(q) for _, q in depth.get("asks", [])[:levels])
    except (TypeError, ValueError):
        return None
    total = bids + asks
    if total <= 0:
        return None
    return bids / total


def parse_taker_ratio(rows: list[dict]) -> Optional[float]:
    """Aggressive-buy share of recent aggTrades volume, in [0, 1].

    Binance aggTrades carry ``m`` = "buyer is maker"; ``m == False`` means an
    aggressive (taker) buy.
    """
    buy = sell = 0.0
    for row in rows:
        try:
            qty = float(row["q"])
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("m"):
            sell += qty
        else:
            buy += qty
    total = buy + sell
    if total <= 0:
        return None
    return buy / total


def parse_basis_pct(mark: float, spot: float) -> float:
    """Perp-vs-spot basis in % (positive = perp premium / bullish structure)."""
    return 100.0 * (mark - spot) / spot if spot else 0.0


def _warn_once(warned: set[str], source: str, exc: Exception) -> None:
    if source not in warned:
        warned.add(source)
        print(
            f"warning: micro source {source!r} unavailable ({exc}); continuing",
            file=sys.stderr,
        )


def enrich_micro(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Best-effort microstructure enrichment; never raises."""
    warned: set[str] = set()
    coins = [c.model_copy(deep=True) for c in snapshot.coins]
    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:

            def _get(url: str, source: str, **params: Any) -> Optional[Any]:
                try:
                    resp = client.get(url, params=params or None)
                    if resp.status_code == 400:
                        return None  # pair doesn't exist — not an outage
                    resp.raise_for_status()
                    return resp.json()
                except Exception as exc:
                    _warn_once(warned, source, exc)
                    return None

            for coin in coins:
                pair = f"{coin.symbol.upper()}USDT"
                depth = _get(f"{BINANCE_SPOT}/api/v3/depth", "depth", symbol=pair, limit=20)
                if isinstance(depth, dict):
                    imb = parse_depth_imbalance(depth)
                    if imb is not None:
                        coin.extras["ob_imbalance"] = round(imb, 4)
                trades = _get(
                    f"{BINANCE_FAPI}/fapi/v1/aggTrades", "aggtrades", symbol=pair, limit=500
                )
                if isinstance(trades, list):
                    ratio = parse_taker_ratio(trades)
                    if ratio is not None:
                        coin.extras["taker_buy_ratio"] = round(ratio, 4)
                mark = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", "mark", symbol=pair)
                if isinstance(mark, dict) and coin.price_usd:
                    try:
                        coin.extras["basis_pct"] = round(
                            parse_basis_pct(float(mark["markPrice"]), coin.price_usd), 4
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
                if coin.symbol.upper() in ("BTC", "ETH"):
                    dvol = _get(
                        f"{DERIBIT}/api/v2/public/get_volatility_index_data",
                        "dvol",
                        currency=coin.symbol.upper(),
                        resolution="3600",
                        start_timestamp=0,
                        end_timestamp=2**53,
                    )
                    try:
                        coin.extras["iv_dvol"] = float(dvol["result"]["data"][-1][4])
                    except (KeyError, IndexError, TypeError, ValueError):
                        pass
    except Exception as exc:
        _warn_once(warned, "micro_enrichment", exc)
    return snapshot.model_copy(update={"coins": coins})
