"""Exchange connector abstraction (Unit 9). See docs/INTERFACES2.md.

Bridge from paper trading toward live/testnet execution:

- :class:`Order` — minimal order model shared by all connectors.
- :class:`ExchangeConnector` — the ABC every connector implements.
- :class:`PaperConnector` — in-memory, deterministic fills against a caller
  supplied price map; used by tests and dry runs.
- :class:`CcxtConnector` — thin bridge to real exchanges via the optional
  ``ccxt`` dependency (imported lazily; install with ``pip install ccxt``).
- :func:`execute_with_fallback` — limit order with cancel+market fallback.
- :func:`twap_orders` — split a large order into equal market slices.

No retries, no reconnect logic: callers own error handling. Everything here
is stdlib + pydantic except the strictly-lazy ``ccxt`` import.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

OrderSide = Literal["buy", "sell"]
OrderType = Literal["limit", "market"]
OrderStatus = Literal["new", "filled", "canceled", "rejected"]

# ccxt order-status strings -> our OrderStatus values.
_CCXT_STATUS_MAP: dict[str, str] = {
    "open": "new",
    "closed": "filled",
    "canceled": "canceled",
    "cancelled": "canceled",
    "rejected": "rejected",
    "expired": "canceled",
}


class Order(BaseModel):
    """One order, paper or live. ``id`` is auto-assigned (uuid4 hex)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    symbol: str
    side: OrderSide
    qty: float = Field(gt=0.0)
    type: OrderType = "market"
    price: Optional[float] = None  # limit price; ignored for market orders
    status: OrderStatus = "new"
    fill_price: Optional[float] = None


class ExchangeConnector(ABC):
    """Minimal connector contract used by execution helpers and the CLI."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Last traded price for ``symbol``."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Submit ``order``; returns the (possibly filled) tracked order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a resting order. Returns False when it cannot be canceled."""

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Current state of a previously placed order (None if unknown)."""

    @abstractmethod
    def get_balance(self) -> float:
        """Free/total quote balance in USD(T)."""


class PaperConnector(ExchangeConnector):
    """Deterministic in-memory exchange for tests and dry runs.

    Fill model:

    - Market orders fill immediately at ``price * (1 + slip)`` for buys and
      ``price * (1 - slip)`` for sells, where ``slip = slippage_bps / 10_000``.
    - Limit orders fill immediately *at the limit price* when marketable
      (buy: market <= limit, sell: market >= limit), else rest as ``"new"``.
    - :meth:`advance` updates the price map and fills any resting limit
      orders that became marketable — this is the test hook standing in for
      the passage of time (e.g. call it from a custom ``wait_fn`` passed to
      :func:`execute_with_fallback`).

    Accounting: ``balance`` is debited/credited with the filled notional
    (qty * fill_price, sign by side) and ``positions[symbol]`` accumulates
    signed base quantity (buys positive, sells negative). Orders for symbols
    missing from the price map are rejected, as are limit orders without a
    price.
    """

    def __init__(
        self,
        prices: dict[str, float],
        slippage_bps: float = 0.0,
        balance: float = 10_000.0,
    ) -> None:
        self.prices: dict[str, float] = dict(prices)
        self.slippage_bps = slippage_bps
        self.balance = balance
        self.positions: dict[str, float] = {}
        self._orders: dict[str, Order] = {}

    # -- ExchangeConnector ------------------------------------------------

    def get_price(self, symbol: str) -> float:
        if symbol not in self.prices:
            raise KeyError(f"no paper price for symbol {symbol!r}")
        return self.prices[symbol]

    def place_order(self, order: Order) -> Order:
        tracked = order.model_copy()  # never mutate the caller's object
        if tracked.symbol not in self.prices or (
            tracked.type == "limit" and tracked.price is None
        ):
            tracked.status = "rejected"
            self._orders[tracked.id] = tracked
            return tracked

        tracked.status = "new"
        tracked.fill_price = None
        self._orders[tracked.id] = tracked
        market = self.prices[tracked.symbol]
        if tracked.type == "market":
            slip = self.slippage_bps / 10_000.0
            px = market * (1 + slip) if tracked.side == "buy" else market * (1 - slip)
            self._fill(tracked, px)
        elif self._marketable(tracked, market):
            self._fill(tracked, float(tracked.price))  # type: ignore[arg-type]
        return tracked

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != "new":
            return False
        order.status = "canceled"
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_balance(self) -> float:
        return self.balance

    # -- paper-specific hooks ---------------------------------------------

    def advance(self, fills: dict[str, float] | None = None) -> list[Order]:
        """Move time forward: apply price updates, fill crossed resting limits.

        Returns the orders filled during this step.
        """
        if fills:
            self.prices.update(fills)
        filled: list[Order] = []
        for order in self._orders.values():
            if order.status != "new" or order.type != "limit":
                continue
            market = self.prices.get(order.symbol)
            if market is not None and self._marketable(order, market):
                self._fill(order, float(order.price))  # type: ignore[arg-type]
                filled.append(order)
        return filled

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _marketable(order: Order, market: float) -> bool:
        assert order.price is not None
        if order.side == "buy":
            return market <= order.price
        return market >= order.price

    def _fill(self, order: Order, px: float) -> None:
        order.status = "filled"
        order.fill_price = px
        signed_qty = order.qty if order.side == "buy" else -order.qty
        self.balance -= signed_qty * px
        self.positions[order.symbol] = (
            self.positions.get(order.symbol, 0.0) + signed_qty
        )


class CcxtConnector(ExchangeConnector):
    """Thin ccxt bridge for ``binanceusdm`` / ``bybit`` (testnet by default).

    ``ccxt`` is imported lazily inside ``__init__`` so the arena runs without
    it; a missing install raises a clear RuntimeError. No retries or
    rate-limit handling beyond ccxt's own ``enableRateLimit``.
    """

    def __init__(
        self,
        exchange_id: str = "binanceusdm",
        api_key: str = "",
        secret: str = "",
        testnet: bool = True,
    ) -> None:
        try:
            import ccxt  # noqa: PLC0415 — strictly lazy optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "ccxt is required for live/testnet execution but is not "
                "installed — run `pip install ccxt` (or install the project's "
                "[live] extra)."
            ) from exc
        try:
            exchange_cls = getattr(ccxt, exchange_id)
        except AttributeError as exc:
            raise ValueError(f"unknown ccxt exchange id {exchange_id!r}") from exc
        self.exchange = exchange_cls(
            {"apiKey": api_key, "secret": secret, "enableRateLimit": True}
        )
        if testnet:
            self.exchange.set_sandbox_mode(True)
        # ccxt needs the symbol for cancel/fetch — remember it per order id.
        self._symbol_by_id: dict[str, str] = {}

    def get_price(self, symbol: str) -> float:
        return float(self.exchange.fetch_ticker(symbol)["last"])

    def place_order(self, order: Order) -> Order:
        raw = self.exchange.create_order(
            order.symbol, order.type, order.side, order.qty, order.price
        )
        placed = self._from_ccxt(raw, order)
        self._symbol_by_id[placed.id] = placed.symbol
        return placed

    def cancel_order(self, order_id: str) -> bool:
        symbol = self._symbol_by_id.get(order_id)
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        symbol = self._symbol_by_id.get(order_id)
        try:
            raw = self.exchange.fetch_order(order_id, symbol)
        except Exception:
            return None
        return self._from_ccxt(raw)

    def get_balance(self) -> float:
        balance = self.exchange.fetch_balance()
        return float((balance.get("USDT") or {}).get("total") or 0.0)

    @staticmethod
    def _from_ccxt(raw: dict[str, Any], template: Order | None = None) -> Order:
        status = _CCXT_STATUS_MAP.get(str(raw.get("status")), "new")
        fill = raw.get("average") or None
        return Order(
            id=str(raw.get("id") or (template.id if template else uuid.uuid4().hex)),
            symbol=str(raw.get("symbol") or (template.symbol if template else "")),
            side=raw.get("side") or (template.side if template else "buy"),
            qty=float(raw.get("amount") or (template.qty if template else 0.0)),
            type=raw.get("type") or (template.type if template else "market"),
            price=raw.get("price") if raw.get("price") is not None
            else (template.price if template else None),
            status=status,  # type: ignore[arg-type]
            fill_price=float(fill) if fill is not None else None,
        )


def execute_with_fallback(
    conn: ExchangeConnector,
    order: Order,
    *,
    limit_timeout_s: float,
    wait_fn: Callable[[float], None] = time.sleep,
) -> Order:
    """Place a limit order; on timeout cancel it and fall back to market.

    Flow: submit ``order`` as a limit (limit price defaults to the current
    market price when ``order.price`` is None); if it fills immediately,
    return it. Otherwise call ``wait_fn(limit_timeout_s)`` — ``time.sleep``
    by default, injectable so tests (and :class:`PaperConnector`, which only
    advances time via its ``advance()`` hook) can simulate the wait, e.g. by
    calling ``conn.advance({...})`` inside a fake ``wait_fn``. After the
    wait, re-check via ``conn.get_order``; if still unfilled, cancel and
    submit a market order for the same qty/side, returning that order.
    """
    limit = order.model_copy(update={"type": "limit"})
    if limit.price is None:
        limit.price = conn.get_price(limit.symbol)
    placed = conn.place_order(limit)
    if placed.status == "filled":
        return placed

    wait_fn(limit_timeout_s)
    current = conn.get_order(placed.id) or placed
    if current.status == "filled":
        return current

    conn.cancel_order(placed.id)
    market = Order(symbol=order.symbol, side=order.side, qty=order.qty, type="market")
    return conn.place_order(market)


def twap_orders(symbol: str, side: str, total_qty: float, slices: int) -> list[Order]:
    """Split ``total_qty`` into ``slices`` equal market orders (TWAP).

    The last slice absorbs floating-point remainder so quantities sum to
    exactly ``total_qty``. The caller is responsible for spacing the
    submissions in time.
    """
    if slices < 1:
        raise ValueError("slices must be >= 1")
    if total_qty <= 0:
        raise ValueError("total_qty must be > 0")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    qty = total_qty / slices
    orders = [
        Order(symbol=symbol, side=side, qty=qty, type="market")
        for _ in range(slices - 1)
    ]
    orders.append(
        Order(symbol=symbol, side=side, qty=total_qty - qty * (slices - 1),
              type="market")
    )
    return orders
