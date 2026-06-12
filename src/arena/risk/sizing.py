"""Position sizing: fractional Kelly, liquidity screen, vol targeting.

``position_fractions`` turns a round's signals into per-symbol stake
fractions of equity. Steps are applied in this exact order (per
docs/INTERFACES2.md, U7):

1. fractional-Kelly base from each agent's past round scores (proxy for
   win rate / win-loss ratio); equal-stake fallback with short history,
2. liquidity screen (volume_24h / market_cap),
3. per-coin cap (``max_position_pct / 100``),
4. volatility-target scalar when vol data is present,
5. drawdown scaling (halve when above threshold),
6. rescale so the gross sum stays within ``max_gross_leverage``.

Because the vol-target scalar (clamped to [0.25, 2.0]) is applied *after*
the per-coin cap, a low-vol coin may end above the per-coin cap before the
final gross rescale; this is intentional per the frozen contract order.
"""

from __future__ import annotations

import math

from arena.models import Direction, MarketSnapshot, RiskSettings, Signal

_HOURS_PER_YEAR = 24.0 * 365.0
_VOL_SCALAR_MIN = 0.25
_VOL_SCALAR_MAX = 2.0
_MIN_KELLY_HISTORY = 5


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Full-Kelly fraction ``f* = p - (1 - p) / b``, clamped to [0, 1].

    ``win_rate`` is clamped into [0, 1] first; a non-positive
    ``win_loss_ratio`` (no payoff information) yields 0.
    """
    if win_loss_ratio <= 0.0:
        return 0.0
    p = min(max(win_rate, 0.0), 1.0)
    f = p - (1.0 - p) / win_loss_ratio
    return min(max(f, 0.0), 1.0)


def _kelly_base(history: list[float], settings: RiskSettings, equal_stake: float) -> float:
    """Fractional-Kelly base stake from an agent's past round scores.

    Past round scores act as a win/loss proxy: positive scores are wins,
    negative scores losses. With fewer than ``_MIN_KELLY_HISTORY`` scores
    (or ``settings.kelly_fraction == 0`` — legacy mode) the equal-stake
    base is used instead. A history with no wins sizes to 0 (no edge).
    """
    if settings.kelly_fraction <= 0.0 or len(history) < _MIN_KELLY_HISTORY:
        return equal_stake
    wins = [s for s in history if s > 0.0]
    losses = [-s for s in history if s < 0.0]
    if not wins:
        return 0.0
    win_rate = len(wins) / len(history)
    if losses:
        win_loss_ratio = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        full = kelly_fraction(win_rate, win_loss_ratio)
    else:
        # No losses on record: f* = p - (1-p)/b -> p as b -> infinity.
        full = win_rate
    return settings.kelly_fraction * full


def _vol_scalar(realized_hourly_vol_pct: float, target_annual_pct: float) -> float:
    """target/realized annualized-vol ratio, clamped to [0.25, 2.0]."""
    realized_annual = realized_hourly_vol_pct * math.sqrt(_HOURS_PER_YEAR)
    if realized_annual <= 0.0:
        return 1.0
    scalar = target_annual_pct / realized_annual
    return min(max(scalar, _VOL_SCALAR_MIN), _VOL_SCALAR_MAX)


def position_fractions(
    scored_history: dict[str, list[float]],
    signals: list[Signal],
    equity: float,
    settings: RiskSettings,
    snapshot: MarketSnapshot | None = None,
    *,
    drawdown_pct: float = 0.0,
) -> dict[str, float]:
    """Map non-FLAT signals to per-symbol stake fractions of equity.

    ``scored_history`` maps agent name -> past round scores (most recent
    last) used as the Kelly win/loss proxy. FLAT signals, liquidity-screened
    coins and zero-sized (no-edge) positions are omitted from the result.
    When several signals target the same symbol, the first one wins.
    ``drawdown_pct`` is the current peak-to-trough drawdown in percent;
    above ``settings.drawdown_scale_threshold_pct`` all fractions are
    halved. ``equity`` is accepted for signature stability (fractions are
    relative, so it does not affect the output).
    """
    del equity  # fractions are relative to equity by definition
    cap = max(settings.max_position_pct, 0.0) / 100.0
    fractions: dict[str, float] = {}
    for sig in signals:
        if sig.direction == Direction.FLAT or sig.symbol in fractions:
            continue
        coin = snapshot.coin(sig.symbol) if snapshot is not None else None
        # 2. Liquidity screen: skip illiquid coins (both fields present).
        if (
            settings.min_volume_mcap_ratio > 0.0
            and coin is not None
            and coin.volume_24h is not None
            and coin.market_cap is not None
            and coin.market_cap > 0.0
            and coin.volume_24h / coin.market_cap < settings.min_volume_mcap_ratio
        ):
            continue
        # 1. fractional-Kelly base (equal-stake = per-coin cap; the gross
        # rescale below turns that into an equal split when binding).
        frac = _kelly_base(scored_history.get(sig.agent, []), settings, cap)
        # 3. per-coin cap.
        frac = min(frac, cap)
        # 4. vol-target scalar when vol data is present.
        if settings.vol_target_annual_pct > 0.0 and coin is not None:
            vol = coin.volatility_24h_pct
            if vol is None:
                vol = coin.extras.get("vol_24h_pct")
            if vol is not None and vol > 0.0:
                frac *= _vol_scalar(vol, settings.vol_target_annual_pct)
        if frac > 0.0:
            fractions[sig.symbol] = frac
    # 5. drawdown scaling.
    if (
        settings.drawdown_scale_threshold_pct > 0.0
        and drawdown_pct > settings.drawdown_scale_threshold_pct
    ):
        fractions = {sym: f * 0.5 for sym, f in fractions.items()}
    # 6. gross-leverage rescale.
    total = sum(fractions.values())
    if settings.max_gross_leverage > 0.0 and total > settings.max_gross_leverage:
        scale = settings.max_gross_leverage / total
        fractions = {sym: f * scale for sym, f in fractions.items()}
    return fractions
