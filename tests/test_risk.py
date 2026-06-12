"""Offline tests for the risk unit."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.models import (
    CoinSnapshot,
    Direction,
    MarketSnapshot,
    RiskSettings,
    ScoredSignal,
    Signal,
)
from arena.risk import (
    apply_round_to_equity_v2,
    check_stops,
    circuit_breaker_state,
    kelly_fraction,
    market_neutral_book,
    position_fractions,
    pre_trade_checks,
)


def _sig(symbol: str, direction: Direction, conf: float = 0.8) -> Signal:
    return Signal(
        agent="a", symbol=symbol, direction=direction, confidence=conf,
        price_at_signal=100.0,
    )


def _scored(symbol: str, direction: Direction, eval_price: float) -> ScoredSignal:
    return ScoredSignal(
        agent="a", symbol=symbol, direction=direction, confidence=0.8,
        price_at_signal=100.0, price_at_eval=eval_price, score=0.0,
    )


# ------------------------------------------------------------------ sizing


def test_kelly_fraction_values() -> None:
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)  # p - (1-p)/b
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)
    assert kelly_fraction(0.4, 1.0) == 0.0  # clamped at 0
    assert kelly_fraction(0.9, 0.0) == 0.0  # no payoff info


def test_position_fractions_caps_and_rescale() -> None:
    settings = RiskSettings(kelly_fraction=0.0, max_position_pct=10.0,
                            max_gross_leverage=0.3)
    sigs = [_sig(s, Direction.LONG) for s in ("A", "B", "C", "D", "E")]
    fr = position_fractions({}, sigs, 10_000.0, settings)
    assert all(f <= 0.10 + 1e-9 for f in fr.values())
    assert sum(fr.values()) <= 0.3 + 1e-9


def test_position_fractions_liquidity_screen() -> None:
    settings = RiskSettings(min_volume_mcap_ratio=0.01)
    snap = MarketSnapshot(
        as_of=datetime(2026, 6, 12, tzinfo=timezone.utc),
        coins=[
            CoinSnapshot(symbol="OK", price_usd=1.0, market_cap=1e9, volume_24h=5e7),
            CoinSnapshot(symbol="ILLIQ", price_usd=1.0, market_cap=1e9, volume_24h=1e6),
        ],
    )
    fr = position_fractions(
        {}, [_sig("OK", Direction.LONG), _sig("ILLIQ", Direction.SHORT)],
        10_000.0, settings, snap,
    )
    assert fr.get("ILLIQ", 0.0) == 0.0
    assert fr.get("OK", 0.0) > 0.0


def test_position_fractions_drawdown_halving() -> None:
    settings = RiskSettings(drawdown_scale_threshold_pct=10.0)
    sigs = [_sig("A", Direction.LONG)]
    base = position_fractions({}, sigs, 10_000.0, settings)["A"]
    scaled = position_fractions({}, sigs, 10_000.0, settings, drawdown_pct=12.0)["A"]
    assert scaled == pytest.approx(base / 2)


# ------------------------------------------------------------------ controls


def test_check_stops_precedence_and_sides() -> None:
    kw = dict(stop_loss_pct=3.0, take_profit_pct=5.0)
    assert check_stops(100.0, 99.0, Direction.LONG, **kw) == "open"
    assert check_stops(100.0, 96.0, Direction.LONG, **kw) == "stopped"
    assert check_stops(100.0, 106.0, Direction.LONG, **kw) == "took_profit"
    assert check_stops(100.0, 106.0, Direction.SHORT, **kw) == "stopped"
    assert check_stops(100.0, 94.0, Direction.SHORT, **kw) == "took_profit"
    assert check_stops(100.0, 50.0, Direction.FLAT, **kw) == "open"
    # disabled thresholds
    assert check_stops(100.0, 50.0, Direction.LONG,
                       stop_loss_pct=0.0, take_profit_pct=0.0) == "open"


def test_circuit_breaker_states() -> None:
    s = RiskSettings(daily_loss_limit_pct=2.0, max_drawdown_halt_pct=15.0)
    assert circuit_breaker_state(-1.0, 5.0, s) == "ok"
    assert circuit_breaker_state(-2.5, 5.0, s) == "daily_halt"
    assert circuit_breaker_state(-2.5, 16.0, s) == "drawdown_halt"  # worst wins
    off = RiskSettings(daily_loss_limit_pct=0.0, max_drawdown_halt_pct=0.0)
    assert circuit_breaker_state(-99.0, 99.0, off) == "ok"


def test_pre_trade_checks() -> None:
    s = RiskSettings(max_position_pct=10.0, max_gross_leverage=1.0)
    assert pre_trade_checks(10_000.0, 500.0, 5_000.0, s) == []
    assert pre_trade_checks(10_000.0, 2_000.0, 5_000.0, s)  # per-coin cap
    assert pre_trade_checks(10_000.0, 500.0, 11_000.0, s)  # leverage
    assert pre_trade_checks(0.0, 500.0, 500.0, s)


# ------------------------------------------------------------------ portfolio


def test_apply_v2_worked_example() -> None:
    # LONG, +10% move, 10% stake of 10k = 1000.
    # PnL = 100; fees = 2*5bps*1000 = 1.0; slippage = 2*(2bps*(1+10/5))*1000 = 1.2
    # funding: +0.01%/8h over 24h = 0.03% of stake, long pays -> -0.30
    settings = RiskSettings(slippage_base_bps=2.0, funding_in_pnl=True,
                            stop_loss_pct=0.0, take_profit_pct=0.0)
    out = apply_round_to_equity_v2(
        10_000.0, [_scored("BTC", Direction.LONG, 110.0)],
        fractions={"BTC": 0.10}, fee_rate=0.0005, settings=settings,
        funding_by_symbol={"BTC": 0.01}, horizon_hours=24.0,
    )
    assert out["fees"] == pytest.approx(1.0)
    assert out["slippage"] == pytest.approx(1.2)
    assert out["funding"] == pytest.approx(-0.30)
    assert out["equity"] == pytest.approx(10_000.0 + 100.0 - 1.0 - 1.2 - 0.30)
    assert out["stopped"] == []


def test_apply_v2_stop_loss_exit_price() -> None:
    settings = RiskSettings(stop_loss_pct=3.0, take_profit_pct=0.0,
                            slippage_base_bps=0.0, funding_in_pnl=False)
    out = apply_round_to_equity_v2(
        10_000.0, [_scored("BTC", Direction.LONG, 90.0)],  # -10% but stop at -3%
        fractions={"BTC": 0.10}, fee_rate=0.0, settings=settings,
    )
    assert out["stopped"] == ["BTC"]
    assert out["equity"] == pytest.approx(10_000.0 - 1_000.0 * 0.03)


def test_apply_v2_flat_and_unsized_hold_cash() -> None:
    settings = RiskSettings()
    out = apply_round_to_equity_v2(
        10_000.0,
        [_scored("A", Direction.FLAT, 90.0), _scored("B", Direction.LONG, 120.0)],
        fractions={"A": 0.5}, fee_rate=0.0005, settings=settings,
    )
    assert out["equity"] == pytest.approx(10_000.0)  # FLAT sized, LONG unsized


def test_market_neutral_book_balanced_legs() -> None:
    sigs = [
        Signal(agent="c", symbol=f"S{i}", direction=Direction.LONG,
               confidence=0.5, price_at_signal=1.0,
               p_long=p, p_short=1.0 - p - 0.1, p_flat=0.1)
        for i, p in enumerate([0.8, 0.7, 0.5, 0.3, 0.2, 0.1])
    ]
    book = market_neutral_book(sigs, top_k=2)
    dirs = {s.symbol: s.direction for s in book}
    assert [dirs[f"S{i}"] for i in range(6)] == [
        Direction.LONG, Direction.LONG, Direction.FLAT,
        Direction.FLAT, Direction.SHORT, Direction.SHORT,
    ]
    longs = sum(1 for d in dirs.values() if d == Direction.LONG)
    shorts = sum(1 for d in dirs.values() if d == Direction.SHORT)
    assert longs == shorts == 2


def test_apply_v2_wick_stress_triggers_stop() -> None:
    # Close at -1% survives a 3% stop, but a 2.5% adverse wick below the
    # close (-3.5% total) crosses it -> stopped at the stop price.
    settings = RiskSettings(stop_loss_pct=3.0, take_profit_pct=0.0,
                            wick_stress_pct=2.5, slippage_base_bps=0.0,
                            funding_in_pnl=False)
    out = apply_round_to_equity_v2(
        10_000.0, [_scored("BTC", Direction.LONG, 99.0)],
        fractions={"BTC": 0.10}, fee_rate=0.0, settings=settings,
    )
    assert out["stopped"] == ["BTC"]
    assert out["equity"] == pytest.approx(10_000.0 - 1_000.0 * 0.03)
    # without wick stress the same position survives
    calm = RiskSettings(stop_loss_pct=3.0, wick_stress_pct=0.0,
                        slippage_base_bps=0.0, funding_in_pnl=False)
    out2 = apply_round_to_equity_v2(
        10_000.0, [_scored("BTC", Direction.LONG, 99.0)],
        fractions={"BTC": 0.10}, fee_rate=0.0, settings=calm,
    )
    assert out2["stopped"] == []
