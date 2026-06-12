"""Crypto LLM Arena command-line interface.

Entry point: ``arena`` (see ``pyproject.toml`` -> ``arena.cli:main``).

The other arena modules (``arena.data``, ``arena.agents``, ``arena.engine``,
``arena.store``) are developed in parallel; they are imported lazily *inside*
the command functions so that importing ``arena.cli`` itself never fails and
tests can inject stub modules via ``sys.modules``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table

from arena.models import ArenaConfig, Direction, ScoredSignal, Signal

#: Hard wall-clock budget per agent per round (the providers' own HTTP
#: timeouts are shorter; this is the safety net for anything hung).
AGENT_TIMEOUT_S = 300.0

_DIRECTION_STYLES: dict[Direction, str] = {
    Direction.LONG: "green",
    Direction.SHORT: "red",
    Direction.FLAT: "dim",
}


def _console() -> Console:
    return Console()


def _err_console() -> Console:
    return Console(stderr=True)


def _load_config(args: argparse.Namespace) -> ArenaConfig:
    from arena.config import load_config

    return load_config(args.config)


def _db_path(args: argparse.Namespace, cfg: ArenaConfig) -> str:
    return args.db if args.db else cfg.arena.db_path


def _fmt_direction(direction: Direction) -> str:
    style = _DIRECTION_STYLES.get(direction, "")
    return f"[{style}]{direction.value}[/{style}]" if style else direction.value


def _pool_signal_samples(samples: list[list[Signal]]) -> list[Signal]:
    """Average probability vectors per symbol across reasoning samples.

    Self-consistency pooling: the per-symbol mean of each sample's
    (p_long, p_short, p_flat); direction = argmax of the pooled vector,
    confidence = pooled max. Legacy samples without probs contribute a
    confidence-weighted one-hot vector.
    """
    by_symbol: dict[str, list[Signal]] = {}
    for sample in samples:
        for sig in sample:
            by_symbol.setdefault(sig.symbol, []).append(sig)
    pooled: list[Signal] = []
    for symbol, sigs in by_symbol.items():
        vectors = []
        for s in sigs:
            if s.has_probs:
                vectors.append((s.p_long or 0.0, s.p_short or 0.0, s.p_flat or 0.0))
            else:
                third = (1.0 - s.confidence) / 2.0
                onehot = {
                    Direction.LONG: (s.confidence, third, third),
                    Direction.SHORT: (third, s.confidence, third),
                    Direction.FLAT: (third, third, s.confidence),
                }
                vectors.append(onehot[s.direction])
        n = len(vectors)
        p = tuple(sum(v[i] for v in vectors) / n for i in range(3))
        directions = (Direction.LONG, Direction.SHORT, Direction.FLAT)
        best = max(range(3), key=lambda i: p[i])
        pooled.append(
            sigs[0].model_copy(
                update={
                    "direction": directions[best],
                    "confidence": p[best],
                    "p_long": p[0],
                    "p_short": p[1],
                    "p_flat": p[2],
                }
            )
        )
    return pooled


def _signals_table(round_id: int, signals: list[Signal]) -> Table:
    table = Table(title=f"Round {round_id} signals")
    table.add_column("Agent", style="bold")
    table.add_column("Symbol")
    table.add_column("Direction")
    table.add_column("Conf", justify="right")
    table.add_column("Price @ signal", justify="right")
    table.add_column("Rationale", overflow="ellipsis", max_width=48)
    for sig in signals:
        table.add_row(
            sig.agent,
            sig.symbol,
            _fmt_direction(sig.direction),
            f"{sig.confidence:.2f}",
            f"{sig.price_at_signal:,.4f}",
            sig.rationale,
        )
    return table


# ---------------------------------------------------------------------------
# run-round
# ---------------------------------------------------------------------------


def cmd_run_round(args: argparse.Namespace) -> int:
    """Fetch a snapshot, collect signals from every agent, store the round."""
    from arena.agents import available_agents
    from arena.agents.base import AgentError
    from arena.data import fetch_top_coins
    from arena.engine import consensus_signals
    from arena.store import Store

    console = _console()
    err = _err_console()
    cfg = _load_config(args)
    settings = cfg.arena

    snapshot = fetch_top_coins(settings.top_n_coins, offline=args.offline)
    if not args.offline:
        try:
            from arena.data.perp import enrich_snapshot

            snapshot = enrich_snapshot(snapshot)
        except Exception as exc:  # enrichment is best-effort, never blocking
            err.print(f"[dim]perp enrichment skipped: {exc}[/dim]")
        for mod_name in ("micro", "macro", "onchain", "news"):
            try:
                import importlib

                mod = importlib.import_module(f"arena.data.{mod_name}")
                snapshot = getattr(mod, f"enrich_{mod_name}")(snapshot)
            except Exception as exc:
                err.print(f"[dim]{mod_name} enrichment skipped: {exc}[/dim]")
    agents = available_agents(cfg.agents, mock=args.mock)
    if not agents:
        err.print("[red]error:[/red] no agents available (missing API keys? try --mock).")
        return 1

    # --- Wave-2 agent intelligence: debate brief + per-agent preambles -----
    regime = "chop"
    debate_brief = ""
    try:
        if cfg.debate.enabled and agents:
            from arena.brain import debate_context

            debate_brief = debate_context(snapshot, agents[0].ask)
    except Exception as exc:
        err.print(f"[dim]debate skipped: {exc}[/dim]")
    try:
        from arena.agents.personas import persona_preamble
        from arena.agents.promptkit import (
            chart_pattern_notes,
            detect_regime,
            fact_subjectivity_split,
            fincot_blueprint,
            regime_demos,
        )
        from arena.brain import AgentMemory, PromptVariantBook

        regime = detect_regime(snapshot)
        shared = "\n\n".join(
            part
            for part in (
                fincot_blueprint(snapshot),
                fact_subjectivity_split(),
                chart_pattern_notes(snapshot),
                regime_demos(regime),
                debate_brief,
            )
            if part
        )
        book = PromptVariantBook(cfg.memory.path)
        for agent in agents:
            parts = [
                persona_preamble(getattr(agent.spec, "role", "generalist")),
                book.preamble_for(agent.name),
                shared,
            ]
            if cfg.memory.enabled:
                lessons = AgentMemory(cfg.memory.path, agent.name).lessons()
                if lessons:
                    parts.append("LESSONS FROM PAST ROUNDS:\n- " + "\n- ".join(lessons))
            agent.prompt_preamble = "\n\n".join(p for p in parts if p) + "\n\n"
    except Exception as exc:
        err.print(f"[dim]prompt enhancement skipped: {exc}[/dim]")

    # Query all fighters concurrently; one slow or broken agent must neither
    # stall nor crash the round (it just sits this one out).
    signals_by_agent: dict[str, list[Signal]] = {}
    with ThreadPoolExecutor(max_workers=max(len(agents), 1)) as pool:
        futures = [(agent, pool.submit(agent.generate_signals, snapshot)) for agent in agents]
        for agent, future in futures:
            try:
                sigs = future.result(timeout=AGENT_TIMEOUT_S)
            except AgentError as exc:
                err.print(
                    f"[yellow]warning:[/yellow] agent '{agent.name}' sits this round out: {exc}"
                )
                continue
            except FutureTimeoutError:
                err.print(
                    f"[yellow]warning:[/yellow] agent '{agent.name}' timed out after "
                    f"{AGENT_TIMEOUT_S:g}s; sits this round out."
                )
                continue
            except Exception as exc:  # defensive: never let one agent kill the round
                err.print(
                    f"[yellow]warning:[/yellow] agent '{agent.name}' failed unexpectedly: {exc}"
                )
                continue
            if sigs:
                signals_by_agent[agent.name] = sigs
            else:
                err.print(f"[yellow]warning:[/yellow] agent '{agent.name}' produced no signals.")

    if len(signals_by_agent) < 2:
        err.print(
            "[red]error:[/red] fewer than 2 agents produced signals; "
            "round aborted (an arena needs at least two fighters)."
        )
        return 1

    # Self-consistency (improvement #13): re-sample low-conviction agents and
    # pool the probability vectors across samples.
    try:
        threshold = cfg.debate.self_consistency_below
        if threshold > 0:
            for agent in agents:
                sigs = signals_by_agent.get(agent.name)
                if not sigs:
                    continue
                mean_conf = sum(s.confidence for s in sigs) / len(sigs)
                if mean_conf >= threshold:
                    continue
                samples = [sigs]
                for _ in range(2):
                    try:
                        samples.append(agent.generate_signals(snapshot))
                    except Exception:
                        break
                if len(samples) > 1:
                    signals_by_agent[agent.name] = _pool_signal_samples(samples)
                    err.print(
                        f"[dim]{agent.name}: low conviction "
                        f"({mean_conf:.2f} < {threshold}); pooled "
                        f"{len(samples)} samples.[/dim]"
                    )
    except Exception as exc:
        err.print(f"[dim]self-consistency skipped: {exc}[/dim]")

    # Deterministic critic pass: attenuate rationale-vs-data contradictions.
    if cfg.debate.critic:
        try:
            from arena.brain import critic_review

            signals_by_agent = {
                name: critic_review(sigs, snapshot)
                for name, sigs in signals_by_agent.items()
            }
        except Exception as exc:
            err.print(f"[dim]critic skipped: {exc}[/dim]")

    db_path = _db_path(args, cfg)
    store = Store(db_path)
    try:
        # Per-agent probability calibration + early-rounds damping.
        if cfg.calibration.enabled:
            try:
                from arena.engine.calibration import CalibrationLog, Calibrator, early_damping

                rounds_seen = store.evaluated_rounds_count()
                clog = CalibrationLog(db_path)
                for name, sigs in signals_by_agent.items():
                    cal = Calibrator(cfg.calibration.method)
                    cal.fit(clog.history(name, window=200))
                    signals_by_agent[name] = [
                        early_damping(
                            cal.calibrate_signal(s),
                            rounds_seen,
                            threshold=cfg.calibration.early_damping_rounds,
                        )
                        for s in sigs
                    ]
                clog.close()
            except Exception as exc:
                err.print(f"[dim]calibration skipped: {exc}[/dim]")

        weights = store.get_weights(sorted(signals_by_agent))
        consensus: list[Signal]
        try:
            from arena.engine.consensus2 import consensus_signals_v2, no_trade_gate

            per_asset = None
            if cfg.weights.per_asset:
                from arena.engine.weights2 import regime_key
                from arena.store.weights_ext import WeightMatrixStore

                book = regime_key(regime) if cfg.weights.regime_conditional else "global"
                wms = WeightMatrixStore(db_path)
                per_asset = wms.get(
                    book, [c.symbol for c in snapshot.coins], sorted(signals_by_agent)
                )
                wms.close()
            consensus, gated = no_trade_gate(
                consensus_signals_v2(
                    signals_by_agent,
                    weights,
                    settings=cfg.consensus,
                    per_asset_weights=per_asset,
                ),
                cfg.consensus.no_trade_min_confidence,
            )
            if gated:
                err.print(
                    "[yellow]no-trade gate:[/yellow] aggregate conviction below "
                    "threshold; round stored all-FLAT."
                )
        except Exception as exc:
            err.print(f"[dim]consensus v2 unavailable ({exc}); using v1.[/dim]")
            consensus = consensus_signals(signals_by_agent, weights)

        round_id = store.create_round(snapshot.as_of, settings.horizon_hours)
        all_signals: list[Signal] = [
            sig for name in sorted(signals_by_agent) for sig in signals_by_agent[name]
        ]
        all_signals.extend(consensus)
        store.add_signals(round_id, all_signals)

        # Journal: exact snapshot + raw replies + token usage (audit/replay).
        if cfg.ops.journal:
            try:
                from arena.ops import config_hash
                from arena.store.journal import Journal

                journal = Journal(db_path)
                journal.record_snapshot(round_id, snapshot, config_hash(args.config))
                for agent in agents:
                    raw = getattr(agent, "last_raw", "")
                    usage = getattr(agent, "last_usage", {}) or {}
                    if raw:
                        journal.record_reply(
                            round_id,
                            agent.name,
                            raw,
                            int(usage.get("input_tokens", 0) or 0),
                            int(usage.get("output_tokens", 0) or 0),
                        )
                journal.close()
            except Exception as exc:
                err.print(f"[dim]journal skipped: {exc}[/dim]")
    finally:
        store.close()

    try:
        from arena.ops import send_alert

        send_alert(
            f"Arena round {round_id}: {len(signals_by_agent)} agents, regime={regime}",
            ops=cfg.ops,
        )
    except Exception:
        pass

    console.print(_signals_table(round_id, all_signals))
    console.print(
        f"Stored round [bold]{round_id}[/bold] "
        f"({len(signals_by_agent)} agents + consensus, "
        f"{len(all_signals)} signals, horizon {settings.horizon_hours:g}h). "
        f"Run [bold]arena evaluate[/bold] once the horizon has passed."
    )
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score every due round, update decision power and paper equity."""
    from arena.data import fetch_prices
    from arena.engine import adaptive_eta, score_signal, update_weights
    from arena.store import Store, apply_round_to_equity

    console = _console()
    err = _err_console()
    cfg = _load_config(args)
    settings = cfg.arena
    force = bool(getattr(args, "force", False))

    store = Store(_db_path(args, cfg))
    try:
        due = store.pending_rounds(datetime.now(timezone.utc), force=force)
        if not due:
            console.print("No rounds due for evaluation. (Use --force to evaluate early.)")
            return 0

        for rnd in due:
            round_id = int(rnd["id"])
            signals = store.signals_for_round(round_id)
            if not signals:
                err.print(f"[yellow]warning:[/yellow] round {round_id} has no signals; skipping.")
                continue

            symbols = sorted({sig.symbol for sig in signals})
            prices = fetch_prices(symbols, offline=args.offline)

            scored: list[ScoredSignal] = []
            for sig in signals:
                price_now = prices.get(sig.symbol)
                if price_now is None:
                    err.print(
                        f"[yellow]warning:[/yellow] no current price for "
                        f"{sig.symbol}; skipping that signal."
                    )
                    continue
                scored.append(
                    ScoredSignal(
                        **sig.model_dump(),
                        price_at_eval=price_now,
                        score=score_signal(
                            sig, price_now, flat_threshold_pct=settings.flat_threshold_pct
                        ),
                    )
                )

            by_agent: dict[str, list[ScoredSignal]] = {}
            for sc in scored:
                by_agent.setdefault(sc.agent, []).append(sc)
            round_scores: dict[str, float] = {
                name: sum(s.score for s in sigs) / len(sigs) for name, sigs in by_agent.items()
            }
            store.record_scores(round_id, scored, round_scores)
            db_path = _db_path(args, cfg)

            # Calibration log: stated confidence vs realized outcome.
            if cfg.calibration.enabled:
                try:
                    from arena.engine.calibration import CalibrationLog
                    from arena.engine.scoring import realized_class, realized_return_pct

                    clog = CalibrationLog(db_path)
                    for sc in scored:
                        if sc.agent == "consensus":
                            continue
                        r = realized_return_pct(sc.price_at_signal, sc.price_at_eval)
                        hit = sc.direction == realized_class(r, settings.flat_threshold_pct)
                        clog.add(sc.agent, sc.confidence, hit, round_id)
                    clog.close()
                except Exception as exc:
                    err.print(f"[dim]calibration log skipped: {exc}[/dim]")

            # Consensus is derived: it never competes for decision power.
            competitor_scores = {k: v for k, v in round_scores.items() if k != "consensus"}
            old_weights = store.get_weights(sorted(competitor_scores))
            if settings.adaptive_eta:
                # record_scores already marked this round EVALUATED, so the
                # count is the 1-based index of the current round.
                eta = adaptive_eta(store.evaluated_rounds_count(), len(competitor_scores))
            else:
                eta = settings.eta
            new_weights = update_weights(
                old_weights,
                competitor_scores,
                eta=eta,
                min_weight=settings.min_weight,
                max_weight=settings.max_weight,
            )
            # Statistical guards: damp young weight moves, watch diversity.
            try:
                from arena.engine.weights2 import (
                    effective_sample_size,
                    significance_damping,
                )

                new_weights = significance_damping(
                    new_weights,
                    old_weights,
                    store.evaluated_rounds_count(),
                    cfg.weights.significance_rounds,
                )
                ess = effective_sample_size(new_weights)
                if ess < cfg.weights.ess_floor:
                    err.print(
                        f"[yellow]warning:[/yellow] ensemble diversity collapsed "
                        f"(ESS {ess:.2f} < {cfg.weights.ess_floor}); one brain dominates."
                    )
            except Exception as exc:
                err.print(f"[dim]weight guards skipped: {exc}[/dim]")
            store.set_weights(new_weights)

            # Per-asset (and regime) decision-power books.
            if cfg.weights.per_asset:
                try:
                    from arena.engine.weights2 import regime_key, update_weight_matrix
                    from arena.store.weights_ext import WeightMatrixStore

                    coin_scores: dict[str, dict[str, float]] = {}
                    for sc in scored:
                        if sc.agent != "consensus" and sc.score is not None:
                            coin_scores.setdefault(sc.symbol, {})[sc.agent] = sc.score
                    books = ["global"]
                    if cfg.weights.regime_conditional:
                        try:
                            from arena.agents.promptkit import detect_regime
                            from arena.store.journal import Journal

                            journal = Journal(db_path)
                            snap = journal.snapshot_for_round(round_id)
                            journal.close()
                            if snap is not None:
                                books.append(regime_key(detect_regime(snap)))
                        except Exception:
                            pass
                    agents_list = sorted(competitor_scores)
                    coins = sorted(coin_scores)
                    wms = WeightMatrixStore(db_path)
                    for book in books:
                        matrix = wms.get(book, coins, agents_list)
                        matrix = update_weight_matrix(
                            matrix,
                            coin_scores,
                            eta=eta,
                            min_weight=settings.min_weight,
                            max_weight=settings.max_weight,
                            clip_low=cfg.weights.clip_factor_low,
                            clip_high=cfg.weights.clip_factor_high,
                        )
                        wms.set(book, matrix)
                    wms.close()
                except Exception as exc:
                    err.print(f"[dim]per-asset weights skipped: {exc}[/dim]")

            # Funding rates from the journaled snapshot (funding-aware PnL).
            funding_by_symbol = None
            if cfg.risk.funding_in_pnl and cfg.ops.journal:
                try:
                    from arena.store.journal import Journal

                    journal = Journal(db_path)
                    snap = journal.snapshot_for_round(round_id)
                    journal.close()
                    if snap is not None:
                        funding_by_symbol = {
                            c.symbol: c.funding_rate_pct
                            for c in snap.coins
                            if c.funding_rate_pct is not None
                        }
                except Exception:
                    pass

            # Paper equity: every agent *including* consensus trades its book,
            # with Kelly sizing, fees, slippage, funding and stops (v2);
            # legacy equal-stake fallback keeps the round alive on any error.
            fee_rate = settings.fee_rate_bps / 10_000.0
            score_history: dict[str, list[float]] = {}
            for row in store.history(limit=500):
                score_history.setdefault(str(row.get("agent")), []).append(
                    float(row.get("round_score", 0.0))
                )
            new_equity: dict[str, float] = {}
            for name, sigs in by_agent.items():
                equity = store.get_equity(name, start_equity=settings.start_equity)
                try:
                    from arena.risk import apply_round_to_equity_v2, position_fractions

                    peak = max(equity, settings.start_equity)
                    drawdown_pct = max(0.0, 100.0 * (peak - equity) / peak)
                    fractions = position_fractions(
                        {name: score_history.get(name, [])},
                        list(sigs),
                        equity,
                        cfg.risk,
                        drawdown_pct=drawdown_pct,
                    )
                    result = apply_round_to_equity_v2(
                        equity,
                        sigs,
                        fractions=fractions,
                        fee_rate=fee_rate,
                        settings=cfg.risk,
                        funding_by_symbol=funding_by_symbol,
                        horizon_hours=float(rnd.get("horizon_hours", 24.0)),
                    )
                    new_equity[name] = result["equity"]
                    if result["stopped"]:
                        err.print(
                            f"[dim]{name}: stops hit on {', '.join(result['stopped'])}[/dim]"
                        )
                except Exception as exc:
                    err.print(f"[dim]risk pipeline fallback for {name}: {exc}[/dim]")
                    new_equity[name] = apply_round_to_equity(equity, sigs, fee_rate=fee_rate)
                store.set_equity(name, new_equity[name])

            # Memory, reflection and prompt-variant feedback.
            if cfg.memory.enabled:
                try:
                    from arena.brain import AgentMemory, PromptVariantBook

                    book = PromptVariantBook(cfg.memory.path)
                    for name, score in round_scores.items():
                        if name == "consensus":
                            continue
                        mem = AgentMemory(cfg.memory.path, name)
                        mem.record_round(
                            round_id, score, mem.reflect(score, by_agent.get(name, []))
                        )
                        book.record_result(name, score)
                except Exception as exc:
                    err.print(f"[dim]memory update skipped: {exc}[/dim]")

            table = Table(title=f"Round {round_id} evaluated ({rnd.get('as_of', '')})")
            table.add_column("Agent", style="bold")
            table.add_column("Round score", justify="right")
            table.add_column("Decision power", justify="right")
            table.add_column("Paper equity", justify="right")
            for name in sorted(round_scores, key=round_scores.get, reverse=True):
                if name == "consensus":
                    weight_cell = "[dim]derived[/dim]"
                else:
                    old_w = old_weights.get(name, 0.0)
                    new_w = new_weights.get(name, old_w)
                    arrow = "green" if new_w >= old_w else "red"
                    weight_cell = f"{old_w:.1%} -> [{arrow}]{new_w:.1%}[/{arrow}]"
                score = round_scores[name]
                score_style = "green" if score >= 0 else "red"
                table.add_row(
                    name,
                    f"[{score_style}]{score:+.4f}[/{score_style}]",
                    weight_cell,
                    f"${new_equity.get(name, 0.0):,.2f}",
                )
            console.print(table)

        console.print(f"Evaluated {len(due)} round(s).")
        # Ops tail: rolling DB backup + round digest alert (both best-effort).
        try:
            from arena.ops import alert_round_summary, backup_db

            if cfg.ops.backup_keep > 0:
                backup_db(_db_path(args, cfg), keep=cfg.ops.backup_keep)
            alert_round_summary(round_scores, new_weights, ops=cfg.ops)
        except Exception:
            pass
    finally:
        store.close()
    return 0


# ---------------------------------------------------------------------------
# leaderboard / history
# ---------------------------------------------------------------------------


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Render the all-time leaderboard."""
    from arena.store import Store

    console = _console()
    cfg = _load_config(args)
    store = Store(_db_path(args, cfg))
    try:
        rows = store.leaderboard()
    finally:
        store.close()

    if not rows:
        console.print("Leaderboard is empty - run [bold]arena run-round[/bold] first.")
        return 0

    table = Table(title="Crypto LLM Arena - leaderboard")
    table.add_column("#", justify="right")
    table.add_column("Agent", style="bold")
    table.add_column("Decision power", justify="right")
    table.add_column("Rounds", justify="right")
    table.add_column("Avg score", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Paper equity", justify="right")
    for rank, row in enumerate(rows, start=1):
        avg = row.get("avg_score")
        table.add_row(
            str(rank),
            str(row.get("agent", "")),
            f"{float(row.get('weight', 0.0)):.1%}",
            str(row.get("rounds", 0)),
            f"{avg:+.4f}" if avg is not None else "-",
            str(row.get("wins", 0)),
            f"${float(row.get('equity', 0.0)):,.2f}",
        )
    console.print(table)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Render recent evaluated rounds."""
    from arena.store import Store

    console = _console()
    cfg = _load_config(args)
    store = Store(_db_path(args, cfg))
    try:
        rows = store.history()
    finally:
        store.close()

    if not rows:
        console.print("No evaluated rounds yet - run [bold]arena evaluate[/bold] after a round.")
        return 0

    table = Table(title="Recent rounds")
    table.add_column("Round", justify="right")
    table.add_column("As of")
    table.add_column("Agent", style="bold")
    table.add_column("Round score", justify="right")
    for row in rows:
        score = float(row.get("round_score", 0.0))
        style = "green" if score >= 0 else "red"
        as_of = row.get("as_of", "")
        if isinstance(as_of, datetime):
            as_of = as_of.strftime("%Y-%m-%d %H:%M UTC")
        table.add_row(
            str(row.get("round_id", "")),
            str(as_of),
            str(row.get("agent", "")),
            f"[{style}]{score:+.4f}[/{style}]",
        )
    console.print(table)
    return 0


# ---------------------------------------------------------------------------
# backtest / report / costs
# ---------------------------------------------------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    """Replay-backtest the arena on synthetic or journaled snapshots."""
    from arena.agents import available_agents
    from arena.backtest import ReplayBacktester, synthetic_snapshots

    console = _console()
    err = _err_console()
    cfg = _load_config(args)
    settings = cfg.arena

    if args.from_journal:
        from arena.store.journal import Journal

        journal = Journal(_db_path(args, cfg))
        snapshots = journal.snapshots()
        journal.close()
        if len(snapshots) < 2:
            err.print("[red]error:[/red] need >= 2 journaled rounds; run more rounds first.")
            return 1
    else:
        snapshots = synthetic_snapshots(args.rounds, args.coins, seed=args.seed)

    # Backtests run deterministic mock agents (LLM replay would re-spend API
    # budget and leak future knowledge through training cutoffs).
    agents = available_agents(cfg.agents, mock=True)
    result = ReplayBacktester(snapshots, agents).run(
        flat_threshold_pct=settings.flat_threshold_pct,
        fee_rate=settings.fee_rate_bps / 10_000.0,
        eta_adaptive=settings.adaptive_eta,
        eta=settings.eta,
        min_weight=settings.min_weight,
        max_weight=settings.max_weight,
        start_equity=settings.start_equity,
    )

    table = Table(title=f"Backtest — {len(snapshots)} snapshots ({len(snapshots) - 1} rounds)")
    table.add_column("Agent", style="bold")
    table.add_column("Weight", justify="right")
    table.add_column("Avg score", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Equity", justify="right")
    table.add_column("Regret", justify="right")
    regret = result.get("regret", {})
    for row in result["leaderboard"]:
        name = row["agent"]
        sharpe_v = row.get("sharpe")
        table.add_row(
            name,
            f"{row['weight']:.1%}",
            f"{row['avg_score']:+.4f}",
            f"{sharpe_v:+.2f}" if sharpe_v is not None else "-",
            f"${row['equity']:,.2f}",
            f"{regret.get(name, 0.0):.3f}",
        )
    console.print(table)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Emit the explainability report for one evaluated round (markdown)."""
    from arena.backtest import round_report_markdown
    from arena.store import Store

    console = _console()
    err = _err_console()
    cfg = _load_config(args)
    store = Store(_db_path(args, cfg))
    try:
        signals = store.signals_for_round(args.round)
        if not signals:
            err.print(f"[red]error:[/red] round {args.round} has no signals.")
            return 1
        round_scores = {
            str(row["agent"]): float(row["round_score"])
            for row in store.history(limit=10_000)
            if int(row.get("round_id", -1)) == args.round
        }
        weights = store.get_weights(sorted({s.agent for s in signals if s.agent != "consensus"}))
    finally:
        store.close()
    # Stored signals lack eval data fields when the round is pending — render
    # what exists; scores appear once `arena evaluate` has run.
    from arena.models import ScoredSignal

    scored = [
        s if isinstance(s, ScoredSignal) else ScoredSignal(
            **s.model_dump(), price_at_eval=s.price_at_signal, score=0.0
        )
        for s in signals
    ]
    console.print(round_report_markdown(args.round, scored, round_scores, weights, weights))
    return 0


def cmd_costs(args: argparse.Namespace) -> int:
    """Per-agent LLM spend from the journal (alpha per dollar matters)."""
    from arena.store.journal import Journal

    console = _console()
    cfg = _load_config(args)
    journal = Journal(_db_path(args, cfg))
    try:
        costs = journal.costs_by_agent()
    finally:
        journal.close()
    if not costs:
        console.print("No journaled replies yet.")
        return 0
    table = Table(title="LLM cost by agent")
    table.add_column("Agent", style="bold")
    table.add_column("Cost (USD)", justify="right")
    for agent, cost in sorted(costs.items(), key=lambda kv: -kv[1]):
        table.add_row(agent, f"${cost:,.4f}")
    console.print(table)
    return 0


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------


def cmd_loop(args: argparse.Namespace) -> int:
    """Forever: evaluate due rounds, run a new round, sleep, repeat."""
    console = _console()
    err = _err_console()
    interval_mins: float = args.interval_mins
    eval_args = argparse.Namespace(
        config=args.config, db=args.db, mock=args.mock, offline=args.offline, force=False
    )

    console.print(
        f"Arena loop started (every {interval_mins:g} min). Press Ctrl-C to stop."
    )
    try:
        while True:
            try:
                cmd_evaluate(eval_args)
            except Exception as exc:  # a flaky evaluation must not kill the loop
                err.print(f"[red]evaluate failed:[/red] {exc}")
            try:
                rc = cmd_run_round(args)
                if rc != 0:
                    err.print("[yellow]run-round did not complete; will retry next cycle.[/yellow]")
            except Exception as exc:  # a flaky round must not kill the loop
                err.print(f"[red]run-round failed:[/red] {exc}")
            console.print(
                f"Sleeping {interval_mins:g} minute(s) until the next round "
                f"(now: {datetime.now(timezone.utc).isoformat(timespec='seconds')})."
            )
            time.sleep(interval_mins * 60)
    except KeyboardInterrupt:
        console.print("\nInterrupted - exiting cleanly. The fighters will rest.")
        return 0


# ---------------------------------------------------------------------------
# parser / entry point
# ---------------------------------------------------------------------------


def _add_global_args(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """Add the global flags. With suppress=True (subparsers), absent flags do
    not override values already parsed at the top level."""

    def d(value: object) -> object:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument(
        "--config", default=d("config.yaml"), metavar="PATH", help="config file (default: config.yaml)"
    )
    parser.add_argument(
        "--db", default=d(None), metavar="PATH", help="SQLite DB path (overrides config)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=d(False),
        help="force deterministic MockAgents (no API keys needed)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=d(False),
        help="use fixture market data instead of the live API",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arena",
        description=(
            "Crypto LLM Arena: LLM agents fight for decision power by calling "
            "LONG/SHORT/FLAT on the top-20 coins."
        ),
    )
    _add_global_args(parser)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_run = sub.add_parser("run-round", help="fetch market data, collect signals, store a round")
    _add_global_args(p_run, suppress=True)
    p_run.set_defaults(func=cmd_run_round)

    p_eval = sub.add_parser("evaluate", help="score due rounds, update weights and paper equity")
    _add_global_args(p_eval, suppress=True)
    p_eval.add_argument(
        "--force", action="store_true", help="evaluate all PENDING rounds even before the horizon"
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_lb = sub.add_parser("leaderboard", help="show decision power, scores, wins and paper equity")
    _add_global_args(p_lb, suppress=True)
    p_lb.set_defaults(func=cmd_leaderboard)

    p_hist = sub.add_parser("history", help="show recent evaluated rounds")
    _add_global_args(p_hist, suppress=True)
    p_hist.set_defaults(func=cmd_history)

    p_bt = sub.add_parser("backtest", help="replay-backtest on synthetic or journaled data")
    _add_global_args(p_bt, suppress=True)
    p_bt.add_argument("--rounds", type=int, default=30, help="synthetic rounds (default 30)")
    p_bt.add_argument("--coins", type=int, default=5, help="synthetic coins (default 5)")
    p_bt.add_argument("--seed", type=int, default=0, help="synthetic data seed")
    p_bt.add_argument(
        "--from-journal", action="store_true", help="replay journaled snapshots instead"
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_rep = sub.add_parser("report", help="explainability report for one round (markdown)")
    _add_global_args(p_rep, suppress=True)
    p_rep.add_argument("--round", type=int, required=True, help="round id")
    p_rep.set_defaults(func=cmd_report)

    p_costs = sub.add_parser("costs", help="per-agent LLM spend from the journal")
    _add_global_args(p_costs, suppress=True)
    p_costs.set_defaults(func=cmd_costs)

    p_loop = sub.add_parser("loop", help="run forever: evaluate, run a round, sleep, repeat")
    _add_global_args(p_loop, suppress=True)
    p_loop.add_argument(
        "--interval-mins",
        type=float,
        default=60.0,
        metavar="N",
        help="minutes between rounds (default: 60)",
    )
    p_loop.set_defaults(func=cmd_loop)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        from arena.config import load_config
        from arena.ops import setup_logging

        setup_logging(json_mode=load_config(args.config).ops.json_logs)
    except Exception:
        pass
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
