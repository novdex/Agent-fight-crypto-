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
DEFILLAMA_STABLES = "https://stablecoins.llama.fi/stablecoins?includePrices=false"
_TIMEOUT_S = 15.0


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
