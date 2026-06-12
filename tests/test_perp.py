"""Offline tests for the perp enrichment layer (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from arena.data.perp import adx, enrich_snapshot, parse_funding_history, parse_klines_ohlc
from arena.models import CoinSnapshot, MarketSnapshot


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        as_of=datetime(2026, 6, 12, tzinfo=timezone.utc),
        coins=[CoinSnapshot(symbol="BTC", price_usd=100_000.0)],
    )


# ---------------------------------------------------------------- ADX


def test_adx_strong_trend_reads_high() -> None:
    # Steadily rising market: +DM dominates -> ADX should read strongly trending.
    closes = [100.0 + 2 * i for i in range(60)]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    value = adx(highs, lows, closes, 14)
    assert value is not None and value > 25


def test_adx_choppy_series_reads_low() -> None:
    # Perfect oscillation: directional movement cancels out.
    closes = [100.0 + (1 if i % 2 else -1) for i in range(60)]
    highs = [c + 1.5 for c in closes]
    lows = [c - 1.5 for c in closes]
    value = adx(highs, lows, closes, 14)
    assert value is not None and value < 20


def test_adx_insufficient_data_returns_none() -> None:
    assert adx([1.0] * 10, [0.5] * 10, [0.8] * 10, 14) is None


# ---------------------------------------------------------------- parsers


def test_parse_funding_history_mean_pct() -> None:
    rows = [{"fundingRate": "0.0001"}, {"fundingRate": "0.0003"}, {"fundingRate": "junk"}]
    assert parse_funding_history(rows) == pytest.approx(0.02)  # mean of 1bp & 3bp, in %


def test_parse_funding_history_empty() -> None:
    assert parse_funding_history([]) is None
    assert parse_funding_history([{"nope": 1}]) is None


def test_parse_klines_ohlc() -> None:
    rows = [
        [0, "1.0", "2.0", "0.5", "1.5", "999"],
        [0, "1.5", "2.5", "1.0", "2.0", "999"],
        ["bad row"],
    ]
    highs, lows, closes = parse_klines_ohlc(rows)
    assert highs == [2.0, 2.5] and lows == [0.5, 1.0] and closes == [1.5, 2.0]


# ---------------------------------------------------------------- enrichment


def test_enrich_snapshot_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """Total network failure: snapshot comes back intact with None perp fields."""

    def dead_get(self, url, **kwargs):  # noqa: ANN001
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(httpx.Client, "get", dead_get)
    snap = _snapshot()
    out = enrich_snapshot(snap)
    coin = out.coins[0]
    assert coin.symbol == "BTC" and coin.price_usd == 100_000.0
    assert coin.funding_rate_pct is None and coin.adx_14 is None
    assert out.fear_greed is None


def test_enrich_snapshot_fills_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = [100.0 + 2 * i for i in range(60)]
    klines = [[0, "0", str(c + 1), str(c - 1), str(c), "0"] for c in closes]

    class FakeResponse:
        def __init__(self, payload):  # noqa: ANN001
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self):  # noqa: ANN201
            return self._payload

    def fake_get(self, url, params=None, **kwargs):  # noqa: ANN001
        if "alternative.me" in url:
            return FakeResponse({"data": [{"value": "27"}]})
        if "premiumIndex" in url:
            return FakeResponse({"lastFundingRate": "0.0007"})
        if "fundingRate" in url:
            return FakeResponse([{"fundingRate": "0.0002"}] * 21)
        if "openInterest" in url:
            return FakeResponse({"openInterest": "1234.5"})
        if "LongShortAccountRatio" in url:
            return FakeResponse([{"longShortRatio": "2.41"}])
        if "klines" in url:
            return FakeResponse(klines)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    out = enrich_snapshot(_snapshot())
    coin = out.coins[0]
    assert out.fear_greed == 27
    assert coin.funding_rate_pct == pytest.approx(0.07)
    assert coin.funding_7d_avg_pct == pytest.approx(0.02)
    assert coin.open_interest_usd == pytest.approx(1234.5 * 100_000.0)
    assert coin.long_short_ratio == pytest.approx(2.41)
    assert coin.adx_14 is not None and coin.adx_14 > 25


def test_prompt_renders_perp_fields() -> None:
    from arena.agents.prompts import build_prompt

    snap = MarketSnapshot(
        as_of=datetime(2026, 6, 12, tzinfo=timezone.utc),
        fear_greed=81,
        coins=[
            CoinSnapshot(
                symbol="BTC",
                price_usd=100_000.0,
                funding_rate_pct=0.08,
                open_interest_usd=5e9,
                long_short_ratio=2.5,
                adx_14=31.0,
            )
        ],
    )
    prompt = build_prompt(snap)
    assert "funding8h=+0.0800%" in prompt
    assert "Fear & Greed index: 81/100" in prompt
    assert "ADX14(1h)=31.0" in prompt
    assert "Reading the perp data" in prompt
