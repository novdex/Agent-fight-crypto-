"""On-chain enrichment — active only when free-tier API keys are configured.

Without keys this module is a silent no-op (the arena must run keyless).
Supported sources: Glassnode (GLASSNODE_API_KEY): MVRV + SOPR for BTC;
DefiLlama (no key): aggregate stablecoin supply change.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import httpx

from arena.models import MarketSnapshot

GLASSNODE = "https://api.glassnode.com/v1/metrics"
CRYPTOQUANT = "https://api.cryptoquant.com/v1"
DEFILLAMA_STABLES = "https://stablecoins.llama.fi/stablecoins?includePrices=false"
_TIMEOUT_S = 15.0


def parse_netflow_musd(payload: dict) -> Optional[float]:
    """Latest exchange netflow (USD millions) from a CryptoQuant-style reply.

    Positive = net inflow to exchanges (distribution / bearish), negative =
    outflow (accumulation / bullish).
    """
    try:
        rows = payload["result"]["data"]
        return float(rows[-1]["netflow_total"]) / 1e6
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_whale_tx_musd(payload: dict) -> Optional[float]:
    """24h whale-transfer volume (USD millions) from a whale-alert-style reply."""
    try:
        txs = payload.get("transactions", [])
        total = sum(float(t.get("amount_usd", 0) or 0) for t in txs)
        return total / 1e6 if txs else None
    except (TypeError, ValueError):
        return None


def parse_stablecoin_supply_chg_pct(payload: dict) -> Optional[float]:
    """Aggregate 7d % change of circulating stablecoin supply (USD pegged)."""
    total_now = total_prev = 0.0
    for asset in payload.get("peggedAssets", []):
        try:
            circ = asset["circulating"]["peggedUSD"]
            prev = asset.get("circulatingPrevWeek", {}).get("peggedUSD")
        except (KeyError, TypeError):
            continue
        if isinstance(circ, (int, float)) and isinstance(prev, (int, float)):
            total_now += circ
            total_prev += prev
    if total_prev <= 0:
        return None
    return 100.0 * (total_now - total_prev) / total_prev


def _warn_once(warned: set[str], source: str, exc: Exception) -> None:
    if source not in warned:
        warned.add(source)
        print(f"warning: onchain source {source!r} unavailable ({exc})", file=sys.stderr)


def enrich_onchain(snapshot: MarketSnapshot) -> MarketSnapshot:
    """Best-effort on-chain extras; silent no-op without keys; never raises."""
    warned: set[str] = set()
    extras = dict(snapshot.extras)
    coins = [c.model_copy(deep=True) for c in snapshot.coins]

    try:
        with httpx.Client(timeout=_TIMEOUT_S) as client:
            try:
                r = client.get(DEFILLAMA_STABLES)
                r.raise_for_status()
                chg = parse_stablecoin_supply_chg_pct(r.json())
                if chg is not None:
                    extras["stablecoin_supply_chg_pct"] = round(chg, 3)
            except Exception as exc:
                _warn_once(warned, "defillama", exc)

            cq_key = os.environ.get("CRYPTOQUANT_API_KEY", "")
            if cq_key:  # exchange netflow (improvement #53)
                try:
                    r = client.get(
                        f"{CRYPTOQUANT}/btc/exchange-flows/netflow",
                        params={"exchange": "all_exchange", "window": "day", "limit": 1},
                        headers={"Authorization": f"Bearer {cq_key}"},
                    )
                    r.raise_for_status()
                    nf = parse_netflow_musd(r.json())
                    if nf is not None:
                        extras["exch_netflow_musd"] = round(nf, 2)
                except Exception as exc:
                    _warn_once(warned, "cryptoquant:netflow", exc)

            wa_key = os.environ.get("WHALE_ALERT_API_KEY", "")
            if wa_key:  # whale transfers (improvement #55)
                try:
                    r = client.get(
                        "https://api.whale-alert.io/v1/transactions",
                        params={"api_key": wa_key, "min_value": 1_000_000, "limit": 100},
                    )
                    r.raise_for_status()
                    wt = parse_whale_tx_musd(r.json())
                    if wt is not None:
                        extras["whale_tx_musd"] = round(wt, 1)
                except Exception as exc:
                    _warn_once(warned, "whale_alert", exc)

            gn_key = os.environ.get("GLASSNODE_API_KEY", "")
            if gn_key:
                for metric, path in (("mvrv", "market/mvrv"), ("sopr", "indicators/sopr")):
                    try:
                        r = client.get(
                            f"{GLASSNODE}/{path}",
                            params={"a": "BTC", "api_key": gn_key, "i": "24h"},
                        )
                        r.raise_for_status()
                        rows = r.json()
                        value = float(rows[-1]["v"])
                        for coin in coins:
                            if coin.symbol.upper() == "BTC":
                                coin.extras[metric] = round(value, 4)
                    except Exception as exc:
                        _warn_once(warned, f"glassnode:{metric}", exc)
    except Exception as exc:
        _warn_once(warned, "onchain_enrichment", exc)

    return snapshot.model_copy(update={"extras": extras, "coins": coins})
