"""Offline tests for Unit 1 (market data layer). No network access."""

from __future__ import annotations

import math
import statistics

import pytest

from arena.data import fetch_prices, fetch_top_coins
from arena.data.market import (
    STABLECOINS,
    build_snapshot,
    ema,
    ema_distance_pct,
    is_stablecoin,
    rsi,
    volatility_pct,
)
from arena.models import MarketSnapshot

# ---------------------------------------------------------------------------
# Pure indicator helpers
# ---------------------------------------------------------------------------


class TestEma:
    def test_constant_series(self) -> None:
        assert ema([5.0] * 30, 20) == pytest.approx(5.0)

    def test_known_small_case(self) -> None:
        # period=3: seed SMA(1,2,3)=2, k=0.5 -> ema = 4*0.5 + 2*0.5 = 3
        assert ema([1.0, 2.0, 3.0, 4.0], 3) == pytest.approx(3.0)

    def test_exact_period_is_sma(self) -> None:
        assert ema([1.0, 2.0, 3.0], 3) == pytest.approx(2.0)

    def test_insufficient_data(self) -> None:
        assert ema([1.0, 2.0], 3) is None
        assert ema([], 20) is None

    def test_distance_pct(self) -> None:
        # constant series: last price equals EMA -> 0% distance
        assert ema_distance_pct([10.0] * 25, 20) == pytest.approx(0.0)
        # period=3 case above: last=4, ema=3 -> +33.33%
        assert ema_distance_pct([1.0, 2.0, 3.0, 4.0], 3) == pytest.approx(100.0 / 3.0)

    def test_distance_insufficient_data(self) -> None:
        assert ema_distance_pct([1.0, 2.0], 20) is None


class TestRsi:
    def test_monotonic_up_is_100(self) -> None:
        assert rsi([float(i) for i in range(1, 20)], 14) == pytest.approx(100.0)

    def test_monotonic_down_is_0(self) -> None:
        assert rsi([float(i) for i in range(20, 1, -1)], 14) == pytest.approx(0.0)

    def test_flat_series_is_50(self) -> None:
        assert rsi([7.0] * 20, 14) == pytest.approx(50.0)

    def test_balanced_alternation_near_50(self) -> None:
        # +1/-1 alternation: Wilder smoothing oscillates around 50 (it does
        # not land exactly on 50 for a finite series), so assert the band.
        values = [100.0 + (i % 2) for i in range(30)]
        r = rsi(values, 14)
        assert r is not None and 45.0 < r < 55.0

    def test_insufficient_data(self) -> None:
        assert rsi([1.0] * 14, 14) is None  # needs period+1 points

    def test_known_wilder_value(self) -> None:
        # 2 gains of 2 and 1 loss of 1 in the seed window (period=3):
        # avg_gain=4/3, avg_loss=1/3, RS=4, RSI=80
        assert rsi([10.0, 12.0, 11.0, 13.0], 3) == pytest.approx(80.0)

    def test_bounds(self) -> None:
        values = [100.0, 103.0, 101.0, 104.0, 102.0, 106.0, 105.0, 108.0,
                  107.0, 110.0, 109.0, 112.0, 111.0, 114.0, 113.0, 116.0]
        r = rsi(values, 14)
        assert r is not None and 0.0 < r < 100.0


class TestVolatility:
    def test_constant_series_is_zero(self) -> None:
        assert volatility_pct([42.0] * 30, 24) == pytest.approx(0.0)

    def test_matches_manual_stdev(self) -> None:
        values = [100.0, 102.0, 101.0, 103.0, 99.0]
        rets = [
            (values[i + 1] - values[i]) / values[i] for i in range(len(values) - 1)
        ]
        expected = statistics.stdev(rets) * 100.0
        assert volatility_pct(values, 24) == pytest.approx(expected)

    def test_uses_only_last_window_returns(self) -> None:
        # Big early move outside the 24-return window must be ignored.
        noisy_head = [100.0, 200.0]
        calm_tail = [100.0 + 0.1 * (i % 2) for i in range(25)]  # 24 returns
        full = volatility_pct(noisy_head + calm_tail, 24)
        calm = volatility_pct(calm_tail, 24)
        assert full == pytest.approx(calm)

    def test_insufficient_data(self) -> None:
        assert volatility_pct([1.0, 2.0], 24) is None
        assert volatility_pct([], 24) is None


# ---------------------------------------------------------------------------
# Stablecoin filtering / snapshot building from raw rows
# ---------------------------------------------------------------------------


def _row(symbol: str, price: float, sparkline: list[float] | None = None) -> dict:
    row: dict = {
        "symbol": symbol.lower(),
        "name": symbol,
        "current_price": price,
        "market_cap": 1e9,
        "total_volume": 1e8,
        "price_change_percentage_1h_in_currency": 0.1,
        "price_change_percentage_24h_in_currency": 1.0,
        "price_change_percentage_7d_in_currency": 2.0,
    }
    if sparkline is not None:
        row["sparkline_in_7d"] = {"price": sparkline}
    return row


class TestStablecoinFiltering:
    def test_is_stablecoin(self) -> None:
        for sym in ("USDT", "usdc", "Dai", "FDUSD", "USDE", "TUSD", "PYUSD",
                    "USDS", "BUSD"):
            assert is_stablecoin(sym)
        assert not is_stablecoin("BTC")
        assert not is_stablecoin("ETH")

    def test_build_snapshot_excludes_stablecoins(self) -> None:
        rows = [
            _row("BTC", 100000.0),
            _row("USDT", 1.0),
            _row("ETH", 5000.0),
            _row("USDC", 1.0),
            _row("SOL", 200.0),
        ]
        snap = build_snapshot(rows, 3)
        assert snap.symbols == ["BTC", "ETH", "SOL"]

    def test_build_snapshot_caps_at_n_preserving_order(self) -> None:
        rows = [_row(s, 1.0) for s in ("BTC", "ETH", "SOL", "XRP")]
        snap = build_snapshot(rows, 2)
        assert snap.symbols == ["BTC", "ETH"]

    def test_missing_sparkline_gives_none_indicators(self) -> None:
        snap = build_snapshot([_row("BTC", 100000.0)], 1)
        coin = snap.coins[0]
        assert coin.rsi_14 is None
        assert coin.ema_20_dist_pct is None
        assert coin.volatility_24h_pct is None

    def test_sparkline_indicators_computed(self) -> None:
        spark = [100.0 + 0.5 * math.sin(i / 3.0) + 0.05 * i for i in range(168)]
        snap = build_snapshot([_row("BTC", spark[-1], sparkline=spark)], 1)
        coin = snap.coins[0]
        assert coin.rsi_14 == pytest.approx(rsi(spark, 14))
        assert coin.ema_20_dist_pct == pytest.approx(ema_distance_pct(spark, 20))
        assert coin.volatility_24h_pct == pytest.approx(volatility_pct(spark, 24))


# ---------------------------------------------------------------------------
# Offline fixture paths
# ---------------------------------------------------------------------------


class TestFetchTopCoinsOffline:
    def test_roundtrips_into_market_snapshot(self) -> None:
        snap = fetch_top_coins(20, offline=True)
        assert isinstance(snap, MarketSnapshot)
        assert len(snap.coins) == 20
        assert snap.as_of.year >= 2024

    def test_has_expected_majors(self) -> None:
        snap = fetch_top_coins(20, offline=True)
        for sym in ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX",
                    "AVAX", "LINK"):
            assert sym in snap.symbols
            coin = snap.coin(sym)
            assert coin is not None and coin.price_usd > 0

    def test_no_stablecoins_in_fixture(self) -> None:
        snap = fetch_top_coins(50, offline=True)
        assert not any(is_stablecoin(s) for s in snap.symbols)
        assert len(snap.coins) >= 20  # fixture carries at least 20 real coins

    def test_indicators_populated(self) -> None:
        snap = fetch_top_coins(20, offline=True)
        for coin in snap.coins:
            assert coin.rsi_14 is not None and 0.0 <= coin.rsi_14 <= 100.0
            assert coin.ema_20_dist_pct is not None
            assert coin.volatility_24h_pct is not None
            assert coin.volatility_24h_pct >= 0.0

    def test_truncates_to_n(self) -> None:
        snap = fetch_top_coins(5, offline=True)
        assert len(snap.coins) == 5
        assert snap.symbols[0] == "BTC"


class TestFetchPricesOffline:
    def test_returns_prices_for_all_snapshot_symbols(self) -> None:
        snap = fetch_top_coins(20, offline=True)
        prices = fetch_prices(snap.symbols, offline=True)
        assert set(prices) == set(snap.symbols)
        assert all(p > 0 for p in prices.values())

    def test_prices_drifted_realistically(self) -> None:
        snap = fetch_top_coins(20, offline=True)
        prices = fetch_prices(snap.symbols, offline=True)
        moves = []
        for coin in snap.coins:
            move = 100.0 * (prices[coin.symbol] - coin.price_usd) / coin.price_usd
            assert 0.4 <= abs(move) <= 8.5, f"{coin.symbol} drift {move:.2f}%"
            moves.append(move)
        assert any(m > 0 for m in moves) and any(m < 0 for m in moves)

    def test_case_insensitive_symbols(self) -> None:
        prices = fetch_prices(["btc", "Eth"], offline=True)
        assert set(prices) == {"BTC", "ETH"}

    def test_unknown_symbols_omitted(self) -> None:
        prices = fetch_prices(["BTC", "NOTACOIN"], offline=True)
        assert "BTC" in prices
        assert "NOTACOIN" not in prices

    def test_only_unknown_symbols_gives_empty_dict(self) -> None:
        assert fetch_prices(["NOTACOIN", "FAKE"], offline=True) == {}
