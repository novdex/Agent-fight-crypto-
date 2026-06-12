"""Offline tests for Unit 2 — LLM agents. No network, no API keys."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from arena.agents import available_agents, create_agent
from arena.agents.anthropic_agent import AnthropicAgent
from arena.agents.base import AgentError, BaseAgent
from arena.agents.mock import MockAgent
from arena.agents.openai_compat import OpenAICompatAgent
from arena.agents.prompts import build_prompt, parse_signals
from arena.models import AgentSpec, CoinSnapshot, Direction, MarketSnapshot, Signal


def make_snapshot(as_of: datetime | None = None) -> MarketSnapshot:
    return MarketSnapshot(
        as_of=as_of or datetime(2026, 6, 12, 0, 0, tzinfo=timezone.utc),
        coins=[
            CoinSnapshot(
                symbol="BTC",
                name="Bitcoin",
                price_usd=60000.0,
                change_1h_pct=0.1,
                change_24h_pct=-1.2,
                change_7d_pct=3.4,
                rsi_14=55.0,
                ema_20_dist_pct=0.8,
                volatility_24h_pct=1.1,
            ),
            CoinSnapshot(symbol="ETH", name="Ethereum", price_usd=3000.0),
            CoinSnapshot(symbol="SOL", name="Solana", price_usd=150.0),
        ],
    )


# ---------------------------------------------------------------- prompts


def test_build_prompt_contains_every_symbol() -> None:
    snap = make_snapshot()
    prompt = build_prompt(snap)
    for symbol in snap.symbols:
        assert symbol in prompt
    assert "LONG" in prompt and "SHORT" in prompt and "FLAT" in prompt
    assert "JSON" in prompt


def test_parse_signals_clean_json() -> None:
    snap = make_snapshot()
    text = json.dumps(
        {
            "signals": [
                {"symbol": "BTC", "direction": "LONG", "confidence": 0.8, "rationale": "up"},
                {"symbol": "ETH", "direction": "SHORT", "confidence": 0.6, "rationale": "down"},
                {"symbol": "SOL", "direction": "FLAT", "confidence": 0.5, "rationale": "meh"},
            ]
        }
    )
    sigs = parse_signals(text, "tester", snap)
    assert [s.symbol for s in sigs] == ["BTC", "ETH", "SOL"]
    by_sym = {s.symbol: s for s in sigs}
    assert by_sym["BTC"].direction is Direction.LONG
    assert by_sym["BTC"].confidence == 0.8
    assert by_sym["BTC"].price_at_signal == 60000.0
    assert by_sym["ETH"].direction is Direction.SHORT
    assert by_sym["SOL"].direction is Direction.FLAT
    assert all(s.agent == "tester" for s in sigs)


def test_parse_signals_fenced_json() -> None:
    snap = make_snapshot()
    inner = json.dumps(
        {"signals": [{"symbol": "BTC", "direction": "LONG", "confidence": 0.7, "rationale": "x"}]}
    )
    text = f"```json\n{inner}\n```"
    sigs = parse_signals(text, "tester", snap)
    by_sym = {s.symbol: s for s in sigs}
    assert by_sym["BTC"].direction is Direction.LONG
    assert by_sym["BTC"].confidence == 0.7
    # Missing symbols backfilled FLAT/0.0
    assert by_sym["ETH"].direction is Direction.FLAT
    assert by_sym["ETH"].confidence == 0.0
    assert by_sym["SOL"].direction is Direction.FLAT


def test_parse_signals_junk_wrapped_json() -> None:
    snap = make_snapshot()
    inner = json.dumps(
        {"signals": [{"symbol": "SOL", "direction": "SHORT", "confidence": 0.4, "rationale": "y"}]}
    )
    text = f"Sure! Here is my analysis of the market:\n{inner}\nHope that helps!"
    sigs = parse_signals(text, "tester", snap)
    by_sym = {s.symbol: s for s in sigs}
    assert by_sym["SOL"].direction is Direction.SHORT
    assert by_sym["SOL"].confidence == 0.4
    assert by_sym["BTC"].direction is Direction.FLAT


def test_parse_signals_garbage_all_flat_backfill() -> None:
    snap = make_snapshot()
    for garbage in ("", "no json here at all", "{broken json", "[1, 2, 3]"):
        sigs = parse_signals(garbage, "tester", snap)
        assert len(sigs) == len(snap.coins)
        assert all(s.direction is Direction.FLAT for s in sigs)
        assert all(s.confidence == 0.0 for s in sigs)
        assert [s.symbol for s in sigs] == snap.symbols
        assert all(
            s.price_at_signal == snap.coin(s.symbol).price_usd for s in sigs  # type: ignore[union-attr]
        )


def test_parse_signals_confidence_clamping() -> None:
    snap = make_snapshot()
    text = json.dumps(
        {
            "signals": [
                {"symbol": "BTC", "direction": "LONG", "confidence": 7.5, "rationale": ""},
                {"symbol": "ETH", "direction": "SHORT", "confidence": -3, "rationale": ""},
                {"symbol": "SOL", "direction": "FLAT", "confidence": "not-a-number"},
            ]
        }
    )
    by_sym = {s.symbol: s for s in parse_signals(text, "tester", snap)}
    assert by_sym["BTC"].confidence == 1.0
    assert by_sym["ETH"].confidence == 0.0
    assert by_sym["SOL"].confidence == 0.0


def test_parse_signals_drops_unknown_symbols() -> None:
    snap = make_snapshot()
    text = json.dumps(
        {
            "signals": [
                {"symbol": "DOGE", "direction": "LONG", "confidence": 0.9, "rationale": "wow"},
                {"symbol": "BTC", "direction": "LONG", "confidence": 0.5, "rationale": ""},
            ]
        }
    )
    sigs = parse_signals(text, "tester", snap)
    assert sorted(s.symbol for s in sigs) == ["BTC", "ETH", "SOL"]
    assert all(s.symbol != "DOGE" for s in sigs)


# ---------------------------------------------------------------- MockAgent


def test_mock_agent_deterministic_same_snapshot() -> None:
    snap = make_snapshot()
    agent = MockAgent(AgentSpec(name="m1", provider="mock", seed=42))
    first = agent.generate_signals(snap)
    second = agent.generate_signals(snap)
    assert first == second
    assert len(first) == len(snap.coins)
    assert all(isinstance(s, Signal) for s in first)
    assert all(0.3 <= s.confidence <= 0.95 for s in first)
    assert all(s.price_at_signal == snap.coin(s.symbol).price_usd for s in first)  # type: ignore[union-attr]


def test_mock_agent_different_seeds_differ() -> None:
    snap = make_snapshot()
    a = MockAgent(AgentSpec(name="a", provider="mock", seed=1))
    b = MockAgent(AgentSpec(name="b", provider="mock", seed=2))
    sigs_a = a.generate_signals(snap)
    sigs_b = b.generate_signals(snap)
    assert [(s.direction, s.confidence) for s in sigs_a] != [
        (s.direction, s.confidence) for s in sigs_b
    ]


def test_mock_agent_different_as_of_differs() -> None:
    agent = MockAgent(AgentSpec(name="m", provider="mock", seed=7))
    snap1 = make_snapshot(datetime(2026, 6, 12, tzinfo=timezone.utc))
    snap2 = make_snapshot(datetime(2026, 6, 13, tzinfo=timezone.utc))
    out1 = [(s.direction, s.confidence) for s in agent.generate_signals(snap1)]
    out2 = [(s.direction, s.confidence) for s in agent.generate_signals(snap2)]
    assert out1 != out2


# ---------------------------------------------------------------- create_agent


def test_create_agent_dispatch() -> None:
    anth = create_agent(
        AgentSpec(name="claude", provider="anthropic", model="claude-opus-4-8")
    )
    assert isinstance(anth, AnthropicAgent)
    assert isinstance(anth, BaseAgent)
    assert anth.name == "claude"

    oai = create_agent(
        AgentSpec(
            name="gpt",
            provider="openai_compat",
            model="gpt-5",
            base_url="https://api.openai.com/v1",
        )
    )
    assert isinstance(oai, OpenAICompatAgent)
    assert oai.name == "gpt"

    mock = create_agent(AgentSpec(name="m", provider="mock", seed=3))
    assert isinstance(mock, MockAgent)


def test_create_agent_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        create_agent(AgentSpec(name="x", provider="banana"))


def test_base_agent_generate_signals_not_implemented() -> None:
    agent = BaseAgent(AgentSpec(name="base", provider="mock"))
    with pytest.raises(NotImplementedError):
        agent.generate_signals(make_snapshot())


# ---------------------------------------------------------------- available_agents


def _specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            name="claude",
            provider="anthropic",
            model="claude-opus-4-8",
            api_key_env="TEST_ANTHROPIC_KEY",
        ),
        AgentSpec(
            name="gpt",
            provider="openai_compat",
            model="gpt-5",
            base_url="https://api.openai.com/v1",
            api_key_env="TEST_OPENAI_KEY",
        ),
    ]


def test_available_agents_skips_missing_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    agents = available_agents(_specs())
    assert [a.name for a in agents] == ["gpt"]
    assert isinstance(agents[0], OpenAICompatAgent)
    err = capsys.readouterr().err
    assert "claude" in err
    assert "TEST_ANTHROPIC_KEY" in err


def test_available_agents_all_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test")
    agents = available_agents(_specs())
    assert [a.name for a in agents] == ["claude", "gpt"]
    assert isinstance(agents[0], AnthropicAgent)


def test_available_agents_mock_forcing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keys absent — mock=True must still build every agent, no warnings needed.
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    agents = available_agents(_specs(), mock=True)
    assert [a.name for a in agents] == ["claude", "gpt"]
    assert all(isinstance(a, MockAgent) for a in agents)
    # Seeds are a stable hash of the name: rebuilding gives identical signals.
    snap = make_snapshot()
    again = available_agents(_specs(), mock=True)
    for a, b in zip(agents, again):
        assert a.spec.seed == b.spec.seed
        assert a.generate_signals(snap) == b.generate_signals(snap)
    # Different names produce different seeds.
    assert agents[0].spec.seed != agents[1].spec.seed


def test_mock_agents_produce_valid_signals() -> None:
    snap = make_snapshot()
    agents = available_agents(_specs(), mock=True)
    for agent in agents:
        sigs = agent.generate_signals(snap)
        assert len(sigs) == len(snap.coins)
        assert all(0.0 <= s.confidence <= 1.0 for s in sigs)
        assert all(s.agent == agent.name for s in sigs)


# ----------------------------------------------- provider parse logic (offline)


def test_openai_compat_wraps_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    import arena.agents.openai_compat as oc

    def boom(*args: object, **kwargs: object) -> object:
        raise httpx.ConnectError("no network in tests")

    monkeypatch.setattr(oc.httpx, "post", boom)
    agent = OpenAICompatAgent(
        AgentSpec(name="gpt", provider="openai_compat", model="gpt-5", base_url="https://x/v1")
    )
    with pytest.raises(AgentError):
        agent.generate_signals(make_snapshot())


def test_anthropic_agent_parses_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction + parse logic only — no live API call."""

    class FakeBlock:
        def __init__(self, type_: str, text: str = "") -> None:
            self.type = type_
            self.text = text

    class FakeResponse:
        content = [
            FakeBlock("thinking"),
            FakeBlock(
                "text",
                json.dumps(
                    {
                        "signals": [
                            {
                                "symbol": "BTC",
                                "direction": "LONG",
                                "confidence": 0.9,
                                "rationale": "r",
                            }
                        ]
                    }
                ),
            ),
        ]

    agent = AnthropicAgent(
        AgentSpec(name="claude", provider="anthropic", model="claude-opus-4-8")
    )

    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(agent._client.messages, "create", fake_create)
    sigs = agent.generate_signals(make_snapshot())
    by_sym = {s.symbol: s for s in sigs}
    assert by_sym["BTC"].direction is Direction.LONG
    assert by_sym["BTC"].confidence == 0.9
    assert by_sym["ETH"].direction is Direction.FLAT
    assert captured["model"] == "claude-opus-4-8"
    assert captured["max_tokens"] == 16000
    assert captured["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured


# ---------------------------------------------------------------------------
# Probability-vector parsing (Brier upgrade)
# ---------------------------------------------------------------------------


def test_parse_signals_probability_vector() -> None:
    snap = make_snapshot()
    text = json.dumps(
        {
            "signals": [
                {"symbol": "BTC", "p_long": 0.7, "p_short": 0.2, "p_flat": 0.1, "rationale": "up"},
                {"symbol": "ETH", "p_long": 0.1, "p_short": 0.6, "p_flat": 0.3, "rationale": "dn"},
            ]
        }
    )
    by_sym = {s.symbol: s for s in parse_signals(text, "a", snap)}
    btc = by_sym["BTC"]
    assert btc.direction == Direction.LONG
    assert btc.confidence == pytest.approx(0.7)
    assert (btc.p_long, btc.p_short, btc.p_flat) == (
        pytest.approx(0.7), pytest.approx(0.2), pytest.approx(0.1),
    )
    assert by_sym["ETH"].direction == Direction.SHORT
    # Backfilled coin has no probability vector
    assert by_sym["SOL"].p_long is None


def test_parse_signals_unnormalized_probs_renormalized() -> None:
    snap = make_snapshot()
    text = json.dumps(
        {"signals": [{"symbol": "BTC", "p_long": 0.8, "p_short": 0.4, "p_flat": 0.4}]}
    )
    sig = {s.symbol: s for s in parse_signals(text, "a", snap)}["BTC"]
    assert sig.p_long == pytest.approx(0.5)  # 0.8 / 1.6
    assert sig.p_long + sig.p_short + sig.p_flat == pytest.approx(1.0)
    assert sig.direction == Direction.LONG


def test_mock_agent_probs_consistent_with_direction() -> None:
    from arena.agents.mock import MockAgent

    agent = MockAgent(AgentSpec(name="m", provider="mock", seed=7))
    for sig in agent.generate_signals(make_snapshot()):
        assert sig.has_probs
        probs = {
            Direction.LONG: sig.p_long,
            Direction.SHORT: sig.p_short,
            Direction.FLAT: sig.p_flat,
        }
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
        assert max(probs, key=probs.get) == sig.direction


def test_openai_compat_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 then a clean response: the agent retries instead of benching."""
    import arena.agents.openai_compat as oc

    calls: list[int] = []

    class FakeResponse:
        def __init__(self, status: int, payload: dict | None = None) -> None:
            self.status_code = status
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

        def json(self) -> dict:
            return self._payload

    ok_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "signals": [
                                {"symbol": "BTC", "p_long": 0.6, "p_short": 0.3, "p_flat": 0.1}
                            ]
                        }
                    )
                }
            }
        ]
    }

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        calls.append(1)
        return FakeResponse(503) if len(calls) == 1 else FakeResponse(200, ok_payload)

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)

    agent = OpenAICompatAgent(
        AgentSpec(name="g", provider="openai_compat", base_url="https://x", model="m")
    )
    sigs = agent.generate_signals(make_snapshot())
    assert len(calls) == 2  # one retry
    assert {s.symbol: s.direction for s in sigs}["BTC"] == Direction.LONG


def test_openai_compat_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    import arena.agents.openai_compat as oc

    calls: list[int] = []

    class Always503:
        status_code = 503

        def raise_for_status(self) -> None:  # pragma: no cover - not reached
            pass

        def json(self) -> dict:  # pragma: no cover - not reached
            return {}

    def fake_post(url: str, **kwargs: object) -> Always503:
        calls.append(1)
        return Always503()

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)

    agent = OpenAICompatAgent(
        AgentSpec(name="g", provider="openai_compat", base_url="https://x", model="m")
    )
    with pytest.raises(AgentError):
        agent.generate_signals(make_snapshot())
    assert len(calls) == 3
