"""Offline tests for ops, journal, and structured outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
import pytest

from arena.models import CoinSnapshot, MarketSnapshot, OpsSettings
from arena.ops import backup_db, config_hash, log_event, send_alert, setup_logging
from arena.store.journal import Journal


def _snap() -> MarketSnapshot:
    return MarketSnapshot(
        as_of=datetime(2026, 6, 12, tzinfo=timezone.utc),
        coins=[CoinSnapshot(symbol="BTC", price_usd=100_000.0)],
        fear_greed=42,
    )


def test_json_logging_shape(capsys) -> None:
    setup_logging(json_mode=True)
    log_event("round_evaluated", round_id=7, agent="claude", score=0.5)
    # records go to stderr/stdout handlers; pull from logging capture instead
    logger = logging.getLogger("arena")
    assert logger.handlers, "setup_logging must attach a handler"


def test_alerts_no_keys_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert send_alert("hello", ops=OpsSettings()) is False


def test_alerts_telegram_posts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class OkResp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

    def fake_post(url, **kwargs):  # noqa: ANN001
        calls.append(url)
        return OkResp()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t0k")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(httpx, "post", fake_post)
    assert send_alert("ping", ops=OpsSettings()) is True
    assert any("api.telegram.org/bott0k" in u or "api.telegram.org/bot" in u for u in calls)


def test_backup_create_prune_missing(tmp_path) -> None:
    db = tmp_path / "a.db"
    db.write_bytes(b"x" * 10)
    paths = [backup_db(str(db), keep=2) for _ in range(3)]
    assert all(p for p in paths)
    backups = list(tmp_path.glob("a.db.bak.*"))
    assert len(backups) <= 2  # pruned
    assert backup_db(str(tmp_path / "missing.db")) is None
    assert len(config_hash(str(db))) == 12
    assert config_hash(str(tmp_path / "nope.yaml")) == ""


def test_journal_round_trip(tmp_path) -> None:
    j = Journal(str(tmp_path / "j.db"))
    snap = _snap()
    j.record_snapshot(3, snap, config_hash="abc123")
    j.record_reply(3, "claude", '{"signals": []}', 1000, 200, 0.0123)
    j.record_reply(3, "gpt", "raw", 500, 100, 0.0040)
    j.record_reply(4, "claude", "raw2", 1, 1, 0.0007)
    loaded = j.snapshot_for_round(3)
    assert loaded is not None and loaded.coins[0].symbol == "BTC"
    assert loaded.fear_greed == 42
    assert j.snapshot_for_round(99) is None
    costs = j.costs_by_agent()
    assert costs["claude"] == pytest.approx(0.013)
    assert costs["gpt"] == pytest.approx(0.004)
    assert len(j.snapshots()) == 1
    j.close()


def test_openai_compat_sends_response_format_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arena.agents.openai_compat import OpenAICompatAgent
    from arena.models import AgentSpec

    bodies: list[dict] = []
    ok_payload = {
        "choices": [{"message": {"content": json.dumps({"signals": [
            {"symbol": "BTC", "p_long": 0.5, "p_short": 0.3, "p_flat": 0.2,
             "rationale": "x"}]})}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }

    class Resp:
        def __init__(self, status: int):
            self.status_code = status

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("bad", request=None, response=None)  # type: ignore[arg-type]

        def json(self) -> dict:
            return ok_payload

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002,ANN001
        bodies.append(dict(json))
        # Reject the structured-output body, accept the plain one.
        return Resp(400) if "response_format" in json else Resp(200)

    import arena.agents.openai_compat as oc

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    agent = OpenAICompatAgent(
        AgentSpec(name="g", provider="openai_compat", base_url="https://x", model="m")
    )
    sigs = agent.generate_signals(_snap())
    assert "response_format" in bodies[0]  # tried structured first
    assert "response_format" not in bodies[-1]  # fell back
    assert sigs[0].p_long == pytest.approx(0.5)
    assert agent.last_usage == {"input_tokens": 11, "output_tokens": 7}
    assert agent.last_raw.startswith("{")


def test_logprob_probs_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """#22: probabilities derived from token logprobs via per-coin classification."""
    import math

    import arena.agents.openai_compat as oc
    from arena.agents.openai_compat import OpenAICompatAgent
    from arena.models import AgentSpec

    main_payload = {
        "choices": [{"message": {"content": json.dumps({"signals": [
            {"symbol": "BTC", "p_long": 0.34, "p_short": 0.33, "p_flat": 0.33,
             "rationale": "meh"}]})}}],
        "usage": {},
    }
    lp_payload = {
        "choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "LONG", "logprob": -0.2},
            {"token": "SHORT", "logprob": -2.0},
            {"token": "FLAT", "logprob": -3.0},
        ]}]}}]
    }

    class Resp:
        def __init__(self, payload):  # noqa: ANN001
            self._p = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self):  # noqa: ANN201
            return self._p

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002,ANN001
        return Resp(lp_payload) if "logprobs" in json else Resp(main_payload)

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    agent = OpenAICompatAgent(AgentSpec(
        name="g", provider="openai_compat", base_url="https://x", model="m",
        logprob_probs=True,
    ))
    sigs = agent.generate_signals(_snap())
    btc = {s.symbol: s for s in sigs}["BTC"]
    z = math.exp(-0.2) + math.exp(-2.0) + math.exp(-3.0)
    assert btc.p_long == pytest.approx(math.exp(-0.2) / z)
    assert btc.direction.value == "LONG"
    assert btc.confidence == pytest.approx(btc.p_long)
