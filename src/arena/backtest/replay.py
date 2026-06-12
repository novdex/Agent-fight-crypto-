"""Walk-forward replay backtester over a sequence of market snapshots.

``ReplayBacktester`` replays history round by round: agents see only
``snapshots[t]``, their signals are scored against the prices in
``snapshots[t+1]`` (walk-forward by construction — no lookahead), decision
weights evolve via the same multiplicative-weights update the live arena
uses, and each agent runs a paper-equity book under the simple v1 stake
rule (50% of equity split equally across non-FLAT positions, ``fee_rate``
charged per side).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from arena.engine import adaptive_eta, score_signal, update_weights
from arena.models import (
    CoinSnapshot,
    Direction,
    MarketSnapshot,
    ScoredSignal,
)

from arena.backtest.metrics import sharpe

__all__ = ["ReplayBacktester", "synthetic_snapshots"]

_STAKE_FRACTION = 0.5  # v1 rule: stake 50% of equity, split equally


def _apply_round_to_equity(
    equity: float, scored: list[ScoredSignal], *, fee_rate: float
) -> float:
    """Simple v1 paper-equity rule (mirrors ``arena.store.paper``).

    ``equity * 0.5`` is divided equally across the round's non-FLAT
    positions; each pays ``fee_rate`` per side (entry + exit). FLAT signals
    hold cash.
    """
    positions = [
        s for s in scored if s.direction != Direction.FLAT and s.price_at_signal > 0
    ]
    if not positions:
        return equity
    stake = equity * _STAKE_FRACTION / len(positions)
    pnl = 0.0
    for sig in positions:
        r = (sig.price_at_eval - sig.price_at_signal) / sig.price_at_signal
        dir_mult = 1.0 if sig.direction == Direction.LONG else -1.0
        pnl += stake * r * dir_mult
        pnl -= 2.0 * fee_rate * stake  # entry + exit taker fees
    return equity + pnl


class ReplayBacktester:
    """Replays snapshots through a set of agents, walk-forward.

    ``agents`` is any iterable of objects exposing ``.name`` and
    ``.generate_signals(snapshot) -> list[Signal]`` (e.g. ``MockAgent``).
    With ``n`` snapshots the backtest runs ``n - 1`` rounds.
    """

    def __init__(self, snapshots: list[MarketSnapshot], agents) -> None:
        self.snapshots = list(snapshots)
        self.agents = list(agents)

    def run(
        self,
        *,
        flat_threshold_pct: float = 1.0,
        fee_rate: float = 0.0005,
        eta_adaptive: bool = True,
        eta: float = 0.35,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
        start_equity: float = 10_000.0,
    ) -> dict:
        """Run the full replay and return the result dict.

        Returns::

            {"leaderboard": [{"agent", "weight", "avg_score", "equity",
                              "sharpe"}, ...],   # sorted by weight desc
             "equity_curves": {agent: [post-round equity floats]},
             "regret": {agent: best_cumulative_score - own_cumulative}}

        Round ``t`` (1-based) scores ``snapshots[t-1]`` signals against the
        same symbol's price in ``snapshots[t]`` (symbols missing from the
        next snapshot are skipped). When ``eta_adaptive`` is true the weight
        update uses ``adaptive_eta(t, n_agents)``, otherwise the fixed
        ``eta``.
        """
        names = [a.name for a in self.agents]
        n_agents = len(names)
        weights: dict[str, float] = (
            {name: 1.0 / n_agents for name in names} if n_agents else {}
        )
        equity: dict[str, float] = {name: start_equity for name in names}
        equity_curves: dict[str, list[float]] = {name: [] for name in names}
        cum_scores: dict[str, float] = {name: 0.0 for name in names}

        n_rounds = max(len(self.snapshots) - 1, 0)
        for t in range(1, n_rounds + 1):
            snap = self.snapshots[t - 1]
            nxt = self.snapshots[t]

            round_scores: dict[str, float] = {}
            scored_by_agent: dict[str, list[ScoredSignal]] = {}
            for agent in self.agents:
                scored: list[ScoredSignal] = []
                for sig in agent.generate_signals(snap):
                    coin = nxt.coin(sig.symbol)
                    if coin is None:
                        continue  # symbol missing from the next snapshot
                    score = score_signal(
                        sig, coin.price_usd, flat_threshold_pct=flat_threshold_pct
                    )
                    scored.append(
                        ScoredSignal(
                            **sig.model_dump(),
                            price_at_eval=coin.price_usd,
                            score=score,
                        )
                    )
                scored_by_agent[agent.name] = scored
                if scored:
                    round_scores[agent.name] = sum(s.score for s in scored) / len(
                        scored
                    )

            cur_eta = adaptive_eta(t, n_agents) if eta_adaptive else eta
            weights = update_weights(
                weights,
                round_scores,
                eta=cur_eta,
                min_weight=min_weight,
                max_weight=max_weight,
            )

            for name in names:
                cum_scores[name] += round_scores.get(name, 0.0)
                equity[name] = _apply_round_to_equity(
                    equity[name], scored_by_agent.get(name, []), fee_rate=fee_rate
                )
                equity_curves[name].append(equity[name])

        best_cum = max(cum_scores.values(), default=0.0)
        regret = {name: best_cum - cum_scores[name] for name in names}

        leaderboard = []
        for name in names:
            curve = equity_curves[name]
            returns = []
            prev = start_equity
            for value in curve:
                returns.append(value / prev - 1.0 if prev > 0 else 0.0)
                prev = value
            leaderboard.append(
                {
                    "agent": name,
                    "weight": weights.get(name, 0.0),
                    "avg_score": cum_scores[name] / n_rounds if n_rounds else 0.0,
                    "equity": equity[name],
                    "sharpe": sharpe(returns),
                }
            )
        leaderboard.sort(key=lambda row: (-row["weight"], -row["avg_score"], row["agent"]))

        return {
            "leaderboard": leaderboard,
            "equity_curves": equity_curves,
            "regret": regret,
        }


def synthetic_snapshots(
    n_rounds: int, n_coins: int = 5, seed: int = 0
) -> list[MarketSnapshot]:
    """Deterministic synthetic snapshot series for tests and demos.

    Seeded GBM-ish random walks (``random.Random(seed)``) for ``n_coins``
    symbols ``"C0".."C{n-1}"`` over ``n_rounds`` hourly snapshots, with
    realistic derived fields (price, 1h/24h/7d changes, volume, market cap,
    volatility). Same arguments -> byte-identical output.
    """
    rng = random.Random(seed)
    base_time = datetime(2024, 1, 1)
    symbols = [f"C{i}" for i in range(n_coins)]

    # Pre-generate full price paths so changes can look back consistently.
    paths: dict[str, list[float]] = {}
    for sym in symbols:
        price = rng.uniform(0.5, 50_000.0)
        path = [price]
        drift = rng.uniform(-0.002, 0.002)
        vol = rng.uniform(0.005, 0.03)
        for _ in range(max(n_rounds - 1, 0)):
            price *= max(1.0 + drift + rng.gauss(0.0, vol), 0.01)
            path.append(price)
        paths[sym] = path

    def _chg(path: list[float], t: int, lookback: int) -> float:
        then = path[max(t - lookback, 0)]
        return 100.0 * (path[t] - then) / then if then > 0 else 0.0

    snapshots: list[MarketSnapshot] = []
    for t in range(n_rounds):
        coins = []
        for i, sym in enumerate(symbols):
            path = paths[sym]
            price = path[t]
            window = path[max(t - 24, 0) : t + 1]
            if len(window) >= 2:
                rets = [
                    100.0 * (b - a) / a for a, b in zip(window, window[1:]) if a > 0
                ]
                mean = sum(rets) / len(rets)
                vol_24h = (
                    sum((r - mean) ** 2 for r in rets) / len(rets)
                ) ** 0.5
            else:
                vol_24h = 0.0
            coins.append(
                CoinSnapshot(
                    symbol=sym,
                    name=f"Coin {i}",
                    price_usd=price,
                    market_cap=price * 1_000_000.0,
                    volume_24h=price * 50_000.0,
                    change_1h_pct=_chg(path, t, 1),
                    change_24h_pct=_chg(path, t, 24),
                    change_7d_pct=_chg(path, t, 168),
                    volatility_24h_pct=vol_24h,
                )
            )
        snapshots.append(
            MarketSnapshot(
                as_of=base_time + timedelta(hours=t),
                coins=coins,
                fear_greed=50,
            )
        )
    return snapshots
