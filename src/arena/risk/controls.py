"""Risk controls: stops, circuit breakers, pre-trade checks."""

from __future__ import annotations

from arena.models import Direction, RiskSettings


def check_stops(
    entry: float,
    price_now: float,
    direction: Direction,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> str:
    """Classify a position against its stop-loss / take-profit levels.

    Returns ``"open"``, ``"stopped"`` or ``"took_profit"``. The price path
    between entry and ``price_now`` is assumed monotone (straight line), so
    when both levels are crossed the stop is conservatively assumed to have
    been hit first. A threshold of 0 disables that side. FLAT positions are
    always ``"open"``.
    """
    if direction == Direction.FLAT or entry <= 0:
        return "open"
    move = (price_now - entry) / entry
    signed = move if direction == Direction.LONG else -move
    if stop_loss_pct > 0 and signed <= -stop_loss_pct / 100.0:
        return "stopped"
    if take_profit_pct > 0 and signed >= take_profit_pct / 100.0:
        return "took_profit"
    return "open"


def liquidation_price(
    entry: float, direction: Direction, *, leverage: float, maintenance_margin_pct: float = 0.5
) -> float | None:
    """Price at which a leveraged perp position is force-closed.

    Standard isolated-margin approximation: a position is liquidated when the
    adverse move consumes ``1/leverage`` of notional minus the maintenance
    margin. At 1x (or below) a long can only liquidate at ~0 and a short
    can't be squeezed into liquidation by this model — returns None for
    leverage <= 1 and for FLAT.
    """
    if direction == Direction.FLAT or leverage <= 1.0 or entry <= 0:
        return None
    bust_move = (1.0 / leverage) - maintenance_margin_pct / 100.0
    if bust_move <= 0:
        return None
    if direction == Direction.LONG:
        return entry * (1.0 - bust_move)
    return entry * (1.0 + bust_move)


def circuit_breaker_state(
    daily_pnl_pct: float, drawdown_pct: float, settings: RiskSettings
) -> str:
    """``"ok"``, ``"daily_halt"`` or ``"drawdown_halt"`` (worst wins).

    ``daily_pnl_pct`` is today's PnL (negative = loss), ``drawdown_pct`` the
    positive peak-to-now drawdown. Thresholds of 0 disable a breaker.
    """
    if (
        settings.max_drawdown_halt_pct > 0
        and drawdown_pct >= settings.max_drawdown_halt_pct
    ):
        return "drawdown_halt"
    if (
        settings.daily_loss_limit_pct > 0
        and daily_pnl_pct <= -settings.daily_loss_limit_pct
    ):
        return "daily_halt"
    return "ok"


def pre_trade_checks(
    equity: float, stake_usd: float, gross_usd: float, settings: RiskSettings
) -> list[str]:
    """Validate one prospective position; [] means pass, else reasons.

    ``gross_usd`` is the total gross exposure *including* the new stake.
    """
    reasons: list[str] = []
    if equity <= 0:
        reasons.append("equity is non-positive")
        return reasons
    if stake_usd <= 0:
        reasons.append("stake is non-positive")
    if stake_usd > equity * settings.max_position_pct / 100.0:
        reasons.append(
            f"stake exceeds per-coin cap ({settings.max_position_pct}% of equity)"
        )
    if gross_usd > equity * settings.max_gross_leverage:
        reasons.append(
            f"gross exposure exceeds max leverage ({settings.max_gross_leverage}x)"
        )
    return reasons
