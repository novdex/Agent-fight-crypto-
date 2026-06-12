"""Best-effort alerting via Telegram bot API and/or Discord webhook.

Credentials are read from the environment using the variable *names* held in
``OpsSettings``. Missing keys or failed requests never raise — ``send_alert``
just returns ``False`` so the round loop is never interrupted by ops noise.
"""

from __future__ import annotations

import os

import httpx

from arena.models import OpsSettings

_TIMEOUT_S = 5.0
_TELEGRAM_MAX_CHARS = 4000  # API limit is 4096
_DISCORD_MAX_CHARS = 1900  # API limit is 2000


def send_alert(text: str, *, ops: OpsSettings) -> bool:
    """Send ``text`` to every configured channel; True if any delivery worked."""
    sent = False

    token = os.environ.get(ops.telegram_token_env, "")
    chat_id = os.environ.get(ops.telegram_chat_id_env, "")
    if token and chat_id:
        try:
            response = httpx.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:_TELEGRAM_MAX_CHARS]},
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
            sent = True
        except Exception:  # alerting must never crash the caller
            pass

    webhook = os.environ.get(ops.discord_webhook_env, "")
    if webhook:
        try:
            response = httpx.post(
                webhook,
                json={"content": text[:_DISCORD_MAX_CHARS]},
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
            sent = True
        except Exception:
            pass

    return sent


def alert_round_summary(
    round_scores: dict[str, float], weights: dict[str, float], *, ops: OpsSettings
) -> bool:
    """Format a compact round digest (scores + decision power) and send it."""
    lines = ["Arena round summary"]
    if round_scores:
        ranked = sorted(round_scores.items(), key=lambda kv: -kv[1])
        lines.append("scores: " + ", ".join(f"{a} {s:+.3f}" for a, s in ranked))
    if weights:
        ranked_w = sorted(weights.items(), key=lambda kv: -kv[1])
        lines.append("weights: " + ", ".join(f"{a} {w * 100:.0f}%" for a, w in ranked_w))
    return send_alert("\n".join(lines), ops=ops)
