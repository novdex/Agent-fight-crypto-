"""Backtesting toolkit: walk-forward replay, metrics, and round reports."""

from arena.backtest.metrics import (
    calmar,
    deflated_sharpe,
    hit_rate,
    max_drawdown_pct,
    profit_factor,
    regime_split,
    sharpe,
    sortino,
)
from arena.backtest.replay import ReplayBacktester, synthetic_snapshots
from arena.backtest.report import round_report_markdown

__all__ = [
    "ReplayBacktester",
    "calmar",
    "deflated_sharpe",
    "hit_rate",
    "max_drawdown_pct",
    "profit_factor",
    "regime_split",
    "round_report_markdown",
    "sharpe",
    "sortino",
    "synthetic_snapshots",
]
