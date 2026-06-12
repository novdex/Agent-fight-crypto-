"""Pure-stdlib performance metrics for backtest results.

Every function is deterministic, takes plain Python lists, and returns
``None`` when the metric is undefined for the given input (fewer than two
observations, zero standard deviation, no losing trades for profit factor,
and so on) instead of raising or returning a misleading number.
"""

from __future__ import annotations

import math
import statistics

__all__ = [
    "calmar",
    "deflated_sharpe",
    "hit_rate",
    "max_drawdown_pct",
    "profit_factor",
    "regime_split",
    "sharpe",
    "sortino",
]


def sharpe(returns: list[float], periods_per_year: float = 365.0) -> float | None:
    """Annualized Sharpe ratio of per-period returns (risk-free rate 0).

    ``mean / sample_std * sqrt(periods_per_year)``. ``None`` when fewer than
    2 observations or the standard deviation is zero.
    """
    if len(returns) < 2:
        return None
    std = statistics.stdev(returns)
    if std == 0.0:
        return None
    return statistics.fmean(returns) / std * math.sqrt(periods_per_year)


def sortino(returns: list[float], periods_per_year: float = 365.0) -> float | None:
    """Annualized Sortino ratio: mean over downside deviation (target 0).

    Downside deviation is the root mean square of the negative returns
    (lower partial moment of order 2). ``None`` when fewer than 2
    observations or there is no downside (deviation is zero).
    """
    if len(returns) < 2:
        return None
    downside = math.sqrt(statistics.fmean([min(r, 0.0) ** 2 for r in returns]))
    if downside == 0.0:
        return None
    return statistics.fmean(returns) / downside * math.sqrt(periods_per_year)


def calmar(
    equity_curve: list[float], periods_per_year: float = 365.0
) -> float | None:
    """Calmar ratio: annualized (CAGR) return over maximum drawdown.

    ``None`` when the curve has fewer than 2 points, contains non-positive
    endpoint values, or has zero drawdown (ratio undefined).
    """
    if len(equity_curve) < 2:
        return None
    start, end = equity_curve[0], equity_curve[-1]
    if start <= 0.0 or end <= 0.0:
        return None
    mdd = max_drawdown_pct(equity_curve) / 100.0
    if mdd == 0.0:
        return None
    n_periods = len(equity_curve) - 1
    annualized = (end / start) ** (periods_per_year / n_periods) - 1.0
    return annualized / mdd


def max_drawdown_pct(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, in percent.

    Returned as a non-negative number (25.0 means a 25% drawdown). An empty
    or monotonically rising curve yields 0.0.
    """
    max_dd = 0.0
    peak = float("-inf")
    for value in equity_curve:
        if value > peak:
            peak = value
        elif peak > 0.0:
            dd = 100.0 * (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def hit_rate(scores: list[float]) -> float | None:
    """Fraction of strictly positive scores. ``None`` on an empty list."""
    if not scores:
        return None
    return sum(1 for s in scores if s > 0.0) / len(scores)


def profit_factor(returns: list[float]) -> float | None:
    """Gross gains divided by gross losses.

    ``None`` when the list is empty or there are no losing periods (the
    ratio is undefined / infinite).
    """
    if not returns:
        return None
    gains = sum(r for r in returns if r > 0.0)
    losses = -sum(r for r in returns if r < 0.0)
    if losses == 0.0:
        return None
    return gains / losses


def deflated_sharpe(sr: float, n_obs: int, n_trials: int = 4) -> float | None:
    """Simplified Bailey-style deflation of a Sharpe ratio.

    ``sr * sqrt(max(1 - n_trials / (n_obs - 1), 0))`` — a first-order
    haircut for multiple testing (Bailey & Lopez de Prado, "The Deflated
    Sharpe Ratio", simplified): the more strategies/agents were tried
    (``n_trials``) relative to the sample size (``n_obs``), the more the
    observed Sharpe is shrunk toward zero. ``None`` when ``n_obs < 2``
    (the deflation factor is undefined).
    """
    if n_obs < 2:
        return None
    return sr * math.sqrt(max(1.0 - n_trials / (n_obs - 1), 0.0))


def regime_split(rounds: list[dict]) -> dict[str, dict]:
    """Aggregate per-(regime, agent) mean score and count.

    ``rounds`` items are ``{"regime": str, "agent": str, "round_score":
    float}``. Returns ``{regime: {agent: {"mean_score": float, "count":
    int}}}``. Unknown/missing keys default to regime ``"unknown"``, agent
    ``"unknown"``, score 0.0.
    """
    sums: dict[str, dict[str, list[float]]] = {}
    for row in rounds:
        regime = str(row.get("regime", "unknown"))
        agent = str(row.get("agent", "unknown"))
        score = float(row.get("round_score", 0.0))
        sums.setdefault(regime, {}).setdefault(agent, []).append(score)
    return {
        regime: {
            agent: {"mean_score": statistics.fmean(scores), "count": len(scores)}
            for agent, scores in agents.items()
        }
        for regime, agents in sums.items()
    }
