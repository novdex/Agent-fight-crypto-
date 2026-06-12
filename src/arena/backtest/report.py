"""Human-readable markdown reports for backtest/arena rounds."""

from __future__ import annotations

from arena.models import ScoredSignal

__all__ = ["round_report_markdown"]


def _fmt_price(p: float) -> str:
    return f"{p:,.4f}" if p < 10 else f"{p:,.2f}"


def round_report_markdown(
    round_id: int,
    signals: list[ScoredSignal],
    round_scores: dict[str, float],
    weights_before: dict,
    weights_after: dict,
) -> str:
    """Render one round as a readable markdown block.

    Includes a signals table (agent, symbol, direction, confidence, entry/
    eval prices, score), the per-agent round scores, and each agent's
    decision weight before -> after the update.
    """
    lines = [f"## Round {round_id}", "", "### Signals", ""]
    if signals:
        lines += [
            "| Agent | Symbol | Direction | Conf | Price@Signal | Price@Eval | Score |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for s in signals:
            lines.append(
                f"| {s.agent} | {s.symbol} | {s.direction.value} "
                f"| {s.confidence:.2f} | {_fmt_price(s.price_at_signal)} "
                f"| {_fmt_price(s.price_at_eval)} | {s.score:+.4f} |"
            )
    else:
        lines.append("_No signals this round._")

    lines += ["", "### Round scores", ""]
    if round_scores:
        for agent in sorted(round_scores, key=round_scores.get, reverse=True):
            lines.append(f"- **{agent}**: {round_scores[agent]:+.4f}")
    else:
        lines.append("_No scores this round._")

    lines += ["", "### Weights (before -> after)", ""]
    agents = sorted(set(weights_before) | set(weights_after))
    if agents:
        lines += ["| Agent | Before | After | Delta |", "|---|---:|---:|---:|"]
        for agent in agents:
            before = float(weights_before.get(agent, 0.0))
            after = float(weights_after.get(agent, 0.0))
            lines.append(
                f"| {agent} | {before:.4f} | {after:.4f} | {after - before:+.4f} |"
            )
    else:
        lines.append("_No weights recorded._")

    return "\n".join(lines) + "\n"
