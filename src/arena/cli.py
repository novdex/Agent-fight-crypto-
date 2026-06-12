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
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table

from arena.models import ArenaConfig, Direction, ScoredSignal, Signal

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
    agents = available_agents(cfg.agents, mock=args.mock)
    if not agents:
        err.print("[red]error:[/red] no agents available (missing API keys? try --mock).")
        return 1

    signals_by_agent: dict[str, list[Signal]] = {}
    for agent in agents:
        try:
            sigs = agent.generate_signals(snapshot)
        except AgentError as exc:
            err.print(
                f"[yellow]warning:[/yellow] agent '{agent.name}' sits this round out: {exc}"
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

    store = Store(_db_path(args, cfg))
    try:
        weights = store.get_weights(sorted(signals_by_agent))
        consensus = consensus_signals(signals_by_agent, weights)
        round_id = store.create_round(snapshot.as_of, settings.horizon_hours)
        all_signals: list[Signal] = [
            sig for name in sorted(signals_by_agent) for sig in signals_by_agent[name]
        ]
        all_signals.extend(consensus)
        store.add_signals(round_id, all_signals)
    finally:
        store.close()

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
    from arena.engine import score_signal, update_weights
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

            # Consensus is derived: it never competes for decision power.
            competitor_scores = {k: v for k, v in round_scores.items() if k != "consensus"}
            old_weights = store.get_weights(sorted(competitor_scores))
            new_weights = update_weights(
                old_weights,
                competitor_scores,
                eta=settings.eta,
                min_weight=settings.min_weight,
                max_weight=settings.max_weight,
            )
            store.set_weights(new_weights)

            # Paper equity: every agent *including* consensus trades its book.
            new_equity: dict[str, float] = {}
            for name, sigs in by_agent.items():
                equity = store.get_equity(name, start_equity=settings.start_equity)
                new_equity[name] = apply_round_to_equity(equity, sigs)
                store.set_equity(name, new_equity[name])

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
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
