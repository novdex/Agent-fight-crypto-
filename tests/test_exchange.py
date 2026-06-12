"""Tests for arena.exchange (Unit 9) — offline, PaperConnector only."""

from __future__ import annotations

import sys

import pytest

from arena.exchange import (
    CcxtConnector,
    ExchangeConnector,
    Order,
    PaperConnector,
    execute_with_fallback,
    twap_orders,
)

PRICES = {"BTC": 100_000.0, "ETH": 5_000.0}


def make_conn(slippage_bps: float = 0.0, balance: float = 1_000_000.0) -> PaperConnector:
    return PaperConnector(dict(PRICES), slippage_bps=slippage_bps, balance=balance)


# -- Order model -----------------------------------------------------------


def test_order_defaults_and_unique_ids() -> None:
    a = Order(symbol="BTC", side="buy", qty=1.0)
    b = Order(symbol="BTC", side="buy", qty=1.0)
    assert a.id != b.id and len(a.id) == 32  # uuid4 hex
    assert a.status == "new" and a.type == "market"
    assert a.price is None and a.fill_price is None


def test_order_rejects_bad_values() -> None:
    with pytest.raises(Exception):
        Order(symbol="BTC", side="hold", qty=1.0)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        Order(symbol="BTC", side="buy", qty=0.0)


# -- PaperConnector: market fills with slippage ------------------------------


def test_market_buy_fills_with_slippage_up() -> None:
    conn = make_conn(slippage_bps=10.0)
    placed = conn.place_order(Order(symbol="BTC", side="buy", qty=0.5, type="market"))
    assert placed.status == "filled"
    assert placed.fill_price == pytest.approx(100_000.0 * 1.001)


def test_market_sell_fills_with_slippage_down() -> None:
    conn = make_conn(slippage_bps=10.0)
    placed = conn.place_order(Order(symbol="BTC", side="sell", qty=0.5, type="market"))
    assert placed.status == "filled"
    assert placed.fill_price == pytest.approx(100_000.0 * 0.999)


def test_unknown_symbol_rejected_and_get_price_raises() -> None:
    conn = make_conn()
    placed = conn.place_order(Order(symbol="DOGE", side="buy", qty=1.0))
    assert placed.status == "rejected"
    with pytest.raises(KeyError):
        conn.get_price("DOGE")


def test_limit_without_price_rejected() -> None:
    conn = make_conn()
    placed = conn.place_order(Order(symbol="BTC", side="buy", qty=1.0, type="limit"))
    assert placed.status == "rejected"


# -- PaperConnector: limit orders --------------------------------------------


def test_marketable_limit_fills_immediately_at_limit() -> None:
    conn = make_conn()
    placed = conn.place_order(
        Order(symbol="BTC", side="buy", qty=1.0, type="limit", price=101_000.0)
    )
    assert placed.status == "filled"
    assert placed.fill_price == pytest.approx(101_000.0)


def test_resting_limit_fills_after_advance_crosses() -> None:
    conn = make_conn()
    placed = conn.place_order(
        Order(symbol="BTC", side="buy", qty=2.0, type="limit", price=99_000.0)
    )
    assert placed.status == "new"

    assert conn.advance() == []  # no price change -> still resting
    assert conn.get_order(placed.id).status == "new"

    filled = conn.advance({"BTC": 98_500.0})  # crosses the limit
    assert [o.id for o in filled] == [placed.id]
    tracked = conn.get_order(placed.id)
    assert tracked is not None and tracked.status == "filled"
    assert tracked.fill_price == pytest.approx(99_000.0)
    assert conn.positions["BTC"] == pytest.approx(2.0)


def test_resting_sell_limit_fills_when_price_rises() -> None:
    conn = make_conn()
    placed = conn.place_order(
        Order(symbol="ETH", side="sell", qty=3.0, type="limit", price=5_200.0)
    )
    assert placed.status == "new"
    conn.advance({"ETH": 5_250.0})
    assert conn.get_order(placed.id).status == "filled"
    assert conn.positions["ETH"] == pytest.approx(-3.0)


def test_cancel_order_only_when_resting() -> None:
    conn = make_conn()
    resting = conn.place_order(
        Order(symbol="BTC", side="buy", qty=1.0, type="limit", price=90_000.0)
    )
    assert conn.cancel_order(resting.id) is True
    assert conn.get_order(resting.id).status == "canceled"
    assert conn.cancel_order(resting.id) is False  # already canceled
    filled = conn.place_order(Order(symbol="BTC", side="buy", qty=1.0))
    assert conn.cancel_order(filled.id) is False
    assert conn.cancel_order("nope") is False
    conn.advance({"BTC": 80_000.0})  # canceled order must never fill
    assert conn.get_order(resting.id).status == "canceled"


# -- PaperConnector: balance & position accounting ----------------------------


def test_balance_and_position_accounting_roundtrip() -> None:
    conn = make_conn(balance=500_000.0)
    conn.place_order(Order(symbol="BTC", side="buy", qty=2.0))
    assert conn.get_balance() == pytest.approx(500_000.0 - 2.0 * 100_000.0)
    assert conn.positions["BTC"] == pytest.approx(2.0)

    conn.advance({"BTC": 110_000.0})
    conn.place_order(Order(symbol="BTC", side="sell", qty=2.0))
    assert conn.positions["BTC"] == pytest.approx(0.0)
    assert conn.get_balance() == pytest.approx(500_000.0 + 2.0 * 10_000.0)


def test_place_order_does_not_mutate_caller_order() -> None:
    conn = make_conn()
    mine = Order(symbol="BTC", side="buy", qty=1.0)
    placed = conn.place_order(mine)
    assert mine.status == "new" and mine.fill_price is None
    assert placed.status == "filled"


# -- execute_with_fallback ----------------------------------------------------


def test_fallback_limit_fills_during_wait() -> None:
    conn = make_conn()
    waited: list[float] = []

    def wait_fn(timeout: float) -> None:
        waited.append(timeout)
        conn.advance({"BTC": 98_000.0})  # price crosses while we "wait"

    order = Order(symbol="BTC", side="buy", qty=1.0, type="limit", price=99_000.0)
    result = execute_with_fallback(conn, order, limit_timeout_s=60.0, wait_fn=wait_fn)
    assert waited == [60.0]
    assert result.type == "limit" and result.status == "filled"
    assert result.fill_price == pytest.approx(99_000.0)
    assert conn.positions["BTC"] == pytest.approx(1.0)


def test_fallback_times_out_then_cancels_and_markets() -> None:
    conn = make_conn(slippage_bps=10.0)
    order = Order(symbol="BTC", side="buy", qty=1.0, type="limit", price=99_000.0)
    result = execute_with_fallback(
        conn, order, limit_timeout_s=5.0, wait_fn=lambda _t: None
    )
    assert result.type == "market" and result.status == "filled"
    assert result.fill_price == pytest.approx(100_000.0 * 1.001)
    assert conn.get_order(order.id).status == "canceled"  # original limit canceled
    assert conn.positions["BTC"] == pytest.approx(1.0)  # filled exactly once


def test_fallback_marketable_limit_returns_without_waiting() -> None:
    conn = make_conn()

    def boom(_t: float) -> None:  # wait_fn must never run
        raise AssertionError("should not wait when limit fills immediately")

    order = Order(symbol="ETH", side="sell", qty=1.0, type="limit", price=4_900.0)
    result = execute_with_fallback(conn, order, limit_timeout_s=60.0, wait_fn=boom)
    assert result.status == "filled" and result.type == "limit"


def test_fallback_defaults_limit_price_to_market() -> None:
    conn = make_conn()
    order = Order(symbol="BTC", side="buy", qty=1.0)  # no price given
    result = execute_with_fallback(
        conn, order, limit_timeout_s=1.0, wait_fn=lambda _t: None
    )
    # limit at current market is marketable -> fills at that price
    assert result.status == "filled"
    assert result.fill_price == pytest.approx(100_000.0)


# -- twap_orders --------------------------------------------------------------


def test_twap_slices_sum_to_total() -> None:
    orders = twap_orders("BTC", "buy", total_qty=10.0, slices=3)
    assert len(orders) == 3
    assert sum(o.qty for o in orders) == 10.0  # exact, last slice absorbs remainder
    assert all(o.type == "market" and o.side == "buy" and o.symbol == "BTC"
               for o in orders)
    assert len({o.id for o in orders}) == 3


def test_twap_single_slice_and_validation() -> None:
    (only,) = twap_orders("ETH", "sell", total_qty=2.5, slices=1)
    assert only.qty == pytest.approx(2.5) and only.side == "sell"
    with pytest.raises(ValueError):
        twap_orders("ETH", "sell", total_qty=1.0, slices=0)
    with pytest.raises(ValueError):
        twap_orders("ETH", "sell", total_qty=0.0, slices=2)
    with pytest.raises(ValueError):
        twap_orders("ETH", "hold", total_qty=1.0, slices=2)


def test_twap_fills_sum_through_paper_connector() -> None:
    conn = make_conn()
    for order in twap_orders("BTC", "buy", total_qty=0.9, slices=4):
        assert conn.place_order(order).status == "filled"
    assert conn.positions["BTC"] == pytest.approx(0.9)


# -- CcxtConnector: only the missing-ccxt error path (offline) ----------------


def test_ccxt_connector_missing_ccxt_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None in sys.modules makes `import ccxt` raise ImportError.
    monkeypatch.setitem(sys.modules, "ccxt", None)
    with pytest.raises(RuntimeError, match="pip install ccxt"):
        CcxtConnector(exchange_id="binanceusdm", testnet=True)


def test_paper_connector_satisfies_abc() -> None:
    assert isinstance(make_conn(), ExchangeConnector)
