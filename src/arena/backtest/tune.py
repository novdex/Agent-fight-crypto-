"""Gradient-free policy search over arena parameters (improvement #8).

The PPO papers learn ensemble weights end-to-end on rolling windows; without
a training corpus we get the same *shape* of benefit — parameters learned
from data rather than hand-picked — via deterministic grid search on the
replay backtester, scored by consensus-proxy Sharpe. Cheap, reproducible,
and runnable on journaled history (`arena tune --from-journal`).
"""

from __future__ import annotations

from typing import Sequence

from arena.backtest.replay import ReplayBacktester
from arena.models import MarketSnapshot


def tune_parameters(
    snapshots: list[MarketSnapshot],
    agents: Sequence,
    *,
    etas: Sequence[float] = (0.15, 0.35, 0.7),
    min_weights: Sequence[float] = (0.05,),
    max_weights: Sequence[float] = (0.4, 0.6, 0.8),
    fee_rate: float = 0.0005,
    flat_threshold_pct: float = 1.0,
) -> dict:
    """Grid-search (eta, min_weight, max_weight) on replayed history.

    Objective: mean Sharpe of the top-weight agent's equity curve (a proxy
    for "the consensus would have followed the right fighter"). Returns
    {"best": {...params, "objective"}, "trials": [...]} — deterministic for
    deterministic agents.
    """
    trials: list[dict] = []
    for eta in etas:
        for lo in min_weights:
            for hi in max_weights:
                if lo * 2 > 1.0 or hi < lo:
                    continue
                result = ReplayBacktester(snapshots, list(agents)).run(
                    flat_threshold_pct=flat_threshold_pct,
                    fee_rate=fee_rate,
                    eta_adaptive=False,
                    eta=eta,
                    min_weight=lo,
                    max_weight=hi,
                )
                board = result["leaderboard"]
                top = board[0] if board else {}
                objective = top.get("sharpe")
                trials.append(
                    {
                        "eta": eta,
                        "min_weight": lo,
                        "max_weight": hi,
                        "objective": objective if objective is not None else float("-inf"),
                        "top_agent": top.get("agent", ""),
                        "top_equity": top.get("equity", 0.0),
                    }
                )
    trials.sort(key=lambda t: -t["objective"])
    return {"best": trials[0] if trials else {}, "trials": trials}
