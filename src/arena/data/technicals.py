"""Extra technical indicators computed from close-price series (pure, offline).

Every indicator is a pure function over a list of closes (oldest first) that
returns ``None`` when the series is too short or degenerate — never raises.
``enrich_technicals`` writes the results into ``CoinSnapshot.extras`` under
short snake_case keys: ``hurst``, ``macd_hist``, ``stoch_rsi``,
``bb_width_pct``, ``ema_ribbon``.
"""

from __future__ import annotations

import math
import sys
from typing import Optional, Sequence

from arena.models import MarketSnapshot

#: EMA periods of the trend "ribbon" used by :func:`ema_ribbon_slope`.
_RIBBON_PERIODS = (5, 10, 20, 50)

#: Minimum points for a Hurst estimate (need a spread of lags).
_HURST_MIN_POINTS = 20


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _ema_series(values: Sequence[float], period: int) -> list[float]:
    """Full EMA series, seeded with the first value."""
    if not values or period <= 0:
        return []
    k = 2.0 / (period + 1.0)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(out[-1] + k * (float(v) - out[-1]))
    return out


def _rsi_series(closes: Sequence[float], period: int = 14) -> list[float]:
    """Wilder RSI series; one value per bar from index ``period`` onwards."""
    if period <= 0 or len(closes) < period + 1:
        return []
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out: list[float] = []

    def _rsi(g: float, lo: float) -> float:
        if g == 0.0 and lo == 0.0:
            return 50.0  # flat market: neutral by convention
        if lo == 0.0:
            return 100.0
        rs = g / lo
        return 100.0 - 100.0 / (1.0 + rs)

    out.append(_rsi(avg_gain, avg_loss))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out.append(_rsi(avg_gain, avg_loss))
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def hurst_exponent(closes: list[float]) -> Optional[float]:
    """Hurst exponent via diffusive scaling (variance of lagged differences).

    Regresses ``log(std(x[t+lag] - x[t]))`` on ``log(lag)`` for lags 2..20;
    the slope estimates H. Works on log prices when all closes are positive
    (scale invariance), raw values otherwise. ~0.5 = random walk, > 0.5 =
    persistent/trending, < 0.5 = mean-reverting. Clamped to [0, 1].
    """
    n = len(closes)
    if n < _HURST_MIN_POINTS:
        return None
    try:
        if all(c > 0 for c in closes):
            xs = [math.log(c) for c in closes]
        else:
            xs = [float(c) for c in closes]
    except (TypeError, ValueError):
        return None

    max_lag = min(20, n // 2)
    log_lags: list[float] = []
    log_taus: list[float] = []
    for lag in range(2, max_lag + 1):
        diffs = [xs[i + lag] - xs[i] for i in range(n - lag)]
        m = sum(diffs) / len(diffs)
        var = sum((d - m) ** 2 for d in diffs) / len(diffs)
        if var <= 0:
            continue
        log_lags.append(math.log(lag))
        log_taus.append(0.5 * math.log(var))

    if len(log_lags) < 2:
        return None
    # Least-squares slope.
    mx = sum(log_lags) / len(log_lags)
    my = sum(log_taus) / len(log_taus)
    denom = sum((x - mx) ** 2 for x in log_lags)
    if denom <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(log_lags, log_taus)) / denom
    return _clamp(slope, 0.0, 1.0)


def macd_histogram(closes: list[float]) -> Optional[float]:
    """MACD histogram (12/26 EMA difference minus its 9-EMA signal line)."""
    if len(closes) < 35:  # 26 for the slow EMA + 9 for the signal line
        return None
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd_line, 9)
    if not signal:
        return None
    return macd_line[-1] - signal[-1]


def stoch_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Stochastic RSI in [0, 100]: where the current RSI-``period`` sits in
    its own min/max range over the last ``period`` RSI readings."""
    if period <= 0:
        return None
    rsis = _rsi_series(closes, period)
    if len(rsis) < period:
        return None
    window = rsis[-period:]
    lo, hi = min(window), max(window)
    if hi - lo <= 0:
        return 50.0  # flat RSI: neutral
    return _clamp(100.0 * (window[-1] - lo) / (hi - lo), 0.0, 100.0)


def bollinger_width_pct(closes: list[float], period: int = 20) -> Optional[float]:
    """Bollinger band width as % of the middle band: 4*stdev / SMA * 100."""
    if period <= 0 or len(closes) < period:
        return None
    window = [float(c) for c in closes[-period:]]
    sma = sum(window) / period
    if sma <= 0:
        return None
    var = sum((c - sma) ** 2 for c in window) / period
    return 4.0 * math.sqrt(var) / sma * 100.0


def ema_ribbon_slope(closes: list[float]) -> Optional[float]:
    """Average slope score of the 5/10/20/50 EMA ribbon, mapped to [-1, 1].

    For each EMA the recent fractional slope per bar (over a 5-bar lookback)
    is squashed with tanh, so a sustained ~1%/bar trend saturates near +/-1
    and a flat market scores ~0; the four scores are averaged.
    """
    if len(closes) < max(_RIBBON_PERIODS) + 1:
        return None
    scores: list[float] = []
    for period in _RIBBON_PERIODS:
        series = _ema_series(closes, period)
        lookback = min(5, len(series) - 1)
        base = series[-1 - lookback]
        if lookback <= 0 or base <= 0:
            continue
        slope_per_bar = (series[-1] - base) / (base * lookback)
        scores.append(math.tanh(200.0 * slope_per_bar))
    if not scores:
        return None
    return _clamp(sum(scores) / len(scores), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Snapshot enrichment (pure: closes are passed in, no network)
# ---------------------------------------------------------------------------


def enrich_technicals(
    snapshot: MarketSnapshot, closes_by_symbol: dict[str, list[float]]
) -> MarketSnapshot:
    """Return a copy of ``snapshot`` with technical extras filled per coin.

    ``closes_by_symbol`` maps uppercase tickers to close series (oldest
    first), e.g. hourly closes from the sparkline/klines already on the
    pipeline. Coins without a series (or with a too-short one) are left
    untouched. Never raises.
    """
    by_symbol = {str(k).upper(): v for k, v in (closes_by_symbol or {}).items()}
    coins = []
    for coin in snapshot.coins:
        coin = coin.model_copy(update={"extras": dict(coin.extras)})
        closes = by_symbol.get(coin.symbol.upper())
        if closes:
            try:
                values = {
                    "hurst": hurst_exponent(closes),
                    "macd_hist": macd_histogram(closes),
                    "stoch_rsi": stoch_rsi(closes),
                    "bb_width_pct": bollinger_width_pct(closes),
                    "ema_ribbon": ema_ribbon_slope(closes),
                }
                coin.extras.update(
                    {k: float(v) for k, v in values.items() if v is not None}
                )
            except Exception as exc:  # malformed series must not block a round
                print(
                    f"warning: technicals for {coin.symbol} failed ({exc}); skipping",
                    file=sys.stderr,
                )
        coins.append(coin)
    return snapshot.model_copy(update={"coins": coins})
