"""Realistic perp paper-PnL and portfolio construction."""

from __future__ import annotations

from arena.models import Direction, RiskSettings, ScoredSignal, Signal

from arena.risk.controls import check_stops


def apply_round_to_equity_v2(
    equity: float,
    scored: list[ScoredSignal],
    *,
    fractions: dict[str, float],
    fee_rate: float,
    settings: RiskSettings,
    funding_by_symbol: dict[str, float] | None = None,
    horizon_hours: float = 24.0,
) -> dict:
    """One round of realistic perp paper trading.

    Per non-FLAT position with a stake fraction: directional PnL on the
    realized move, minus per-side taker fees, minus a move-scaled slippage
    cost, plus/minus funding accrual (longs pay positive funding, shorts
    receive it), with straight-line-path stop-loss / take-profit exits via
    :func:`check_stops` (exit at the stop/TP price instead of the eval
    price). Returns ``{"equity", "fees", "funding", "slippage", "stopped"}``.
    """
    fees = funding = slippage = 0.0
    stopped: list[str] = []
    pnl = 0.0
    funding_by_symbol = funding_by_symbol or {}

    for sig in scored:
        frac = fractions.get(sig.symbol, 0.0)
        if sig.direction == Direction.FLAT or frac <= 0 or sig.price_at_signal <= 0:
            continue
        stake = equity * frac
        exit_price = sig.price_at_eval
        state = check_stops(
            sig.price_at_signal,
            sig.price_at_eval,
            sig.direction,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
        )
        # Intrabar wick stress (improvement #69): even when the close survives,
        # an adverse wick of wick_stress_pct beyond the worst close can take
        # out the stop. Checked against the stop side only.
        if (
            state == "open"
            and settings.wick_stress_pct > 0
            and settings.stop_loss_pct > 0
        ):
            wick = settings.wick_stress_pct / 100.0
            if sig.direction == Direction.LONG:
                adverse = min(sig.price_at_signal, sig.price_at_eval) * (1.0 - wick)
            else:
                adverse = max(sig.price_at_signal, sig.price_at_eval) * (1.0 + wick)
            if (
                check_stops(
                    sig.price_at_signal,
                    adverse,
                    sig.direction,
                    stop_loss_pct=settings.stop_loss_pct,
                    take_profit_pct=0.0,
                )
                == "stopped"
            ):
                state = "stopped"
        sl_mult = settings.stop_loss_pct / 100.0
        tp_mult = settings.take_profit_pct / 100.0
        dir_mult = 1.0 if sig.direction == Direction.LONG else -1.0
        if state == "stopped":
            exit_price = sig.price_at_signal * (1.0 - dir_mult * sl_mult)
            stopped.append(sig.symbol)
        elif state == "took_profit":
            exit_price = sig.price_at_signal * (1.0 + dir_mult * tp_mult)
            stopped.append(sig.symbol)

        r = (exit_price - sig.price_at_signal) / sig.price_at_signal
        pnl += stake * r * dir_mult

        position_fees = 2.0 * fee_rate * stake
        fees += position_fees

        if settings.slippage_base_bps > 0:
            slip = settings.slippage_base_bps * (1.0 + abs(r * 100.0) / 5.0) / 10_000.0
            slippage += 2.0 * slip * stake  # entry + exit

        if settings.funding_in_pnl:
            rate_pct = funding_by_symbol.get(sig.symbol)
            if rate_pct is not None:
                # rate is % per 8h; longs pay positive funding.
                cost = stake * (rate_pct / 100.0) * (horizon_hours / 8.0)
                funding += -dir_mult * cost

    new_equity = equity + pnl - fees - slippage + funding
    return {
        "equity": new_equity,
        "fees": fees,
        "funding": funding,
        "slippage": slippage,
        "stopped": stopped,
    }


def market_neutral_book(signals: list[Signal], top_k: int = 5) -> list[Signal]:
    """Rank-based market-neutral book: strongest longs vs weakest shorts.

    Signals are ranked by ``p_long - p_short`` (falling back to a
    direction-signed confidence for legacy signals). The strongest ``k`` get
    LONG, the weakest ``k`` get SHORT (equal leg counts, k bounded by half
    the book), everything else is flattened. Probability vectors are
    preserved; directions/confidence are overwritten to express the book.
    """

    def strength(sig: Signal) -> float:
        if sig.has_probs:
            return (sig.p_long or 0.0) - (sig.p_short or 0.0)
        if sig.direction == Direction.LONG:
            return sig.confidence
        if sig.direction == Direction.SHORT:
            return -sig.confidence
        return 0.0

    ranked = sorted(signals, key=strength, reverse=True)
    k = max(0, min(top_k, len(ranked) // 2))
    longs = {s.symbol for s in ranked[:k]}
    shorts = {s.symbol for s in ranked[-k:]} if k else set()

    book: list[Signal] = []
    for sig in signals:
        if sig.symbol in longs:
            new_dir = Direction.LONG
        elif sig.symbol in shorts:
            new_dir = Direction.SHORT
        else:
            new_dir = Direction.FLAT
        book.append(
            sig.model_copy(
                update={
                    "direction": new_dir,
                    "confidence": abs(strength(sig)) if new_dir != Direction.FLAT else 0.0,
                }
            )
        )
    return book
