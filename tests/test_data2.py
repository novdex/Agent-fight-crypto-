"""Offline tests for wave-2 data modules (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from arena.data.macro import enrich_macro, parse_stooq_change_pct
from arena.data.micro import (
    enrich_micro,
    parse_basis_pct,
    parse_depth_imbalance,
    parse_taker_ratio,
)
from arena.data.news import detect_events, enrich_news
from arena.data.onchain import parse_stablecoin_supply_chg_pct
from arena.data.technicals import (
    bollinger_width_pct,
    ema_ribbon_slope,
    enrich_technicals,
    hurst_exponent,
    macd_histogram,
    stoch_rsi,
)
from arena.models import CoinSnapshot, MarketSnapshot


def _snap() -> MarketSnapshot:
    return MarketSnapshot(
        as_of=datetime(2026, 6, 12, tzinfo=timezone.utc),
        coins=[CoinSnapshot(symbol="BTC", price_usd=100_000.0),
               CoinSnapshot(symbol="ETH", price_usd=5_000.0)],
    )


# ------------------------------------------------------------- technicals


def test_technicals_on_synthetic_series() -> None:
    trend = [100.0 * (1.01 ** i) for i in range(120)]
    chop = [100.0 + (1.0 if i % 2 else -1.0) for i in range(120)]
    h_trend, h_chop = hurst_exponent(trend), hurst_exponent(chop)
    assert h_trend is not None and h_chop is not None
    assert h_trend > h_chop  # trending > mean-reverting persistence
    assert macd_histogram(trend) is not None
    sr = stoch_rsi(trend)
    assert sr is not None and 0.0 <= sr <= 100.0
    bw = bollinger_width_pct(chop)
    assert bw is not None and bw >= 0.0
    slope = ema_ribbon_slope(trend)
    assert slope is not None and slope > 0
    assert ema_ribbon_slope(list(reversed(trend))) < 0


def test_technicals_insufficient_data() -> None:
    short = [1.0, 2.0, 3.0]
    assert hurst_exponent(short) is None
    assert macd_histogram(short) is None
    assert stoch_rsi(short) is None


def test_enrich_technicals_writes_extras() -> None:
    closes = {"BTC": [100.0 * (1.005 ** i) for i in range(120)]}
    out = enrich_technicals(_snap(), closes)
    btc = out.coin("BTC")
    assert btc is not None and "macd_hist" in btc.extras and "hurst" in btc.extras
    eth = out.coin("ETH")
    assert eth is not None and "macd_hist" not in eth.extras  # no series given


# ------------------------------------------------------------- micro parsers


def test_parse_depth_imbalance() -> None:
    depth = {"bids": [["100", "3.0"], ["99", "3.0"]], "asks": [["101", "2.0"]]}
    assert parse_depth_imbalance(depth) == pytest.approx(0.75)
    assert parse_depth_imbalance({"bids": [], "asks": []}) is None


def test_parse_taker_ratio() -> None:
    rows = [{"q": "8", "m": False}, {"q": "2", "m": True}]  # 8 buy, 2 sell
    assert parse_taker_ratio(rows) == pytest.approx(0.8)
    assert parse_taker_ratio([]) is None


def test_parse_basis_pct() -> None:
    assert parse_basis_pct(101.0, 100.0) == pytest.approx(1.0)
    assert parse_basis_pct(0.0, 0.0) == 0.0


def test_enrich_micro_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    def dead_get(self, url, **kw):  # noqa: ANN001
        raise httpx.ConnectError("blocked")

    monkeypatch.setattr(httpx.Client, "get", dead_get)
    out = enrich_micro(_snap())
    assert out.coins[0].extras == {}


# ------------------------------------------------------------- macro / onchain


def test_parse_stooq_change_pct() -> None:
    csv = "Symbol,Date,Time,Open,High,Low,Close,Volume\nDX.F,2026-06-12,22:00,100,101,99,102,0\n"
    assert parse_stooq_change_pct(csv) == pytest.approx(2.0)
    assert parse_stooq_change_pct("header only") is None


def test_enrich_macro_eth_btc_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def dead_get(self, url, **kw):  # noqa: ANN001
        raise httpx.ConnectError("blocked")

    monkeypatch.setattr(httpx.Client, "get", dead_get)
    out = enrich_macro(_snap())
    assert out.extras["eth_btc_ratio"] == pytest.approx(0.05)
    assert "btc_dominance_pct" not in out.extras


def test_parse_stablecoin_supply() -> None:
    payload = {"peggedAssets": [
        {"circulating": {"peggedUSD": 105.0}, "circulatingPrevWeek": {"peggedUSD": 100.0}},
        {"circulating": {"peggedUSD": 210.0}, "circulatingPrevWeek": {"peggedUSD": 200.0}},
    ]}
    assert parse_stablecoin_supply_chg_pct(payload) == pytest.approx(5.0)
    assert parse_stablecoin_supply_chg_pct({}) is None


# ------------------------------------------------------------- news


def test_detect_events() -> None:
    heads = ["SEC sues exchange", "Bitcoin ETF inflows surge", "Quiet day"]
    tags = detect_events(heads)
    assert "REGULATORY" in tags and "ETF_FLOW" in tags and len(tags) == 2


def test_enrich_news_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def dead_get(self, url, **kw):  # noqa: ANN001
        raise httpx.ConnectError("blocked")

    monkeypatch.setattr(httpx.Client, "get", dead_get)
    out = enrich_news(_snap())
    assert out.headlines == []
