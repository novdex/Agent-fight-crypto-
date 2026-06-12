# Frozen Module Interfaces

Every module is built by an independent worker against this contract.
**Do not deviate from these signatures.** Shared types live in `src/arena/models.py`
(`Direction`, `CoinSnapshot`, `MarketSnapshot`, `Signal`, `ScoredSignal`,
`AgentSpec`, `RoundStatus`, `ArenaSettings`, `ArenaConfig`) and
`src/arena/config.py` (`load_config`, `api_key_for`).

Rules for all modules:

- Python 3.10+, pydantic v2, full type hints.
- No dependencies beyond those in `pyproject.toml`.
- A module's tests may import **only** its own module plus `arena.models` /
  `arena.config`. Tests must pass offline (no network, no API keys).
- Each module owns only its listed files; never edit another module's files.

---

## Unit 1 — Market data: `src/arena/data/market.py` (+ `src/arena/data/__init__.py`)

Data source: CoinGecko free API (`https://api.coingecko.com/api/v3`), no key needed.

```python
def fetch_top_coins(n: int = 20, *, offline: bool = False) -> MarketSnapshot: ...
def fetch_prices(symbols: list[str], *, offline: bool = False) -> dict[str, float]: ...
```

- `fetch_top_coins`: GET `/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={n}&page=1&sparkline=true&price_change_percentage=1h,24h,7d`.
  Exclude stablecoins (symbol in {USDT, USDC, DAI, FDUSD, USDE, TUSD, PYUSD, USDS, BUSD} — fetch extra rows so n real coins remain).
  Compute from the 7d hourly sparkline: `rsi_14` (hourly RSI), `ema_20_dist_pct`
  (% distance of last price from the 20-period EMA), `volatility_24h_pct`
  (stdev of last 24 hourly returns × 100). Indicators are `None` if sparkline missing.
- `fetch_prices`: current USD price per symbol (same endpoint, filtered).
- `offline=True`: read `tests/fixtures/market_snapshot.json` (a serialized
  `MarketSnapshot`) and `tests/fixtures/prices_later.json`
  (`{"SYMBOL": price, ...}` — prices ~24h "later", with realistic drift vs the
  snapshot so scores are nonzero). Unit 1 creates both fixtures (hand-written or
  recorded) with ≥20 coins.
- Network errors raise `RuntimeError` with a helpful message.

Files owned: `src/arena/data/*`, `tests/fixtures/market_snapshot.json`,
`tests/fixtures/prices_later.json`, `tests/test_market.py`.

---

## Unit 2 — LLM agents: `src/arena/agents/`

```python
# src/arena/agents/base.py
class BaseAgent:
    def __init__(self, spec: AgentSpec): self.spec = spec
    @property
    def name(self) -> str: return self.spec.name
    def generate_signals(self, snapshot: MarketSnapshot) -> list[Signal]:
        raise NotImplementedError

# src/arena/agents/__init__.py
def create_agent(spec: AgentSpec) -> BaseAgent: ...   # dispatch on spec.provider
def available_agents(specs: list[AgentSpec], *, mock: bool = False) -> list[BaseAgent]: ...
```

- `create_agent` providers: `"anthropic"`, `"openai_compat"`, `"mock"`. Unknown → `ValueError`.
- `available_agents`: builds an agent per spec; if `mock=True`, every spec is
  forced to a `MockAgent` (same name, seed = stable hash of name). If a spec's
  API key env var is unset (`arena.config.api_key_for`), skip it with a warning
  on stderr rather than failing.
- `anthropic` provider: official `anthropic` SDK, `client.messages.create`,
  `max_tokens=16000`, `thinking={"type": "adaptive"}`. Do NOT pass `temperature`.
- `openai_compat` provider: `httpx.post(f"{base_url}/chat/completions", ...)`,
  bearer auth, `messages=[{"role":"user","content":prompt}]`. 120s timeout.
- `prompts.py`: `build_prompt(snapshot: MarketSnapshot) -> str` — presents each
  coin's stats and demands STRICT JSON:
  `{"signals":[{"symbol":"BTC","direction":"LONG|SHORT|FLAT","confidence":0.0-1.0,"rationale":"..."}, ...]}`
  with exactly one entry per coin. `parse_signals(text: str, agent_name: str, snapshot: MarketSnapshot) -> list[Signal]` —
  robust extraction (strip code fences, find outermost `{...}`), clamps
  confidence, drops unknown symbols, fills `price_at_signal` from the snapshot,
  and fills any *missing* symbols with a FLAT/0.0 signal so every agent always
  covers every coin. A completely unparseable reply → all-FLAT signals.
- `MockAgent` (`mock.py`): deterministic, offline. Direction/confidence derived
  from `sha256(f"{seed}:{symbol}:{snapshot.as_of.isoformat()}")` — no randomness
  module state, no network.
- Per-agent API failures must not crash a round: raise `AgentError` (define in
  `base.py`) and let callers catch it.

Files owned: `src/arena/agents/*`, `tests/test_agents.py`.

---

## Unit 3 — Engine: `src/arena/engine/`

```python
# scoring.py
def realized_return_pct(price_then: float, price_now: float) -> float: ...
def score_signal(sig: Signal, price_now: float, *, flat_threshold_pct: float = 1.0) -> float: ...

# weights.py
def update_weights(weights: dict[str, float], agent_scores: dict[str, float],
                   *, eta: float, min_weight: float, max_weight: float) -> dict[str, float]: ...
def equal_weights(names: list[str]) -> dict[str, float]: ...

# consensus.py
def consensus_signals(signals_by_agent: dict[str, list[Signal]],
                      weights: dict[str, float]) -> list[Signal]: ...
```

Scoring (deterministic, bounded to [-1, 1]):

```
r = 100 * (price_now - price_then) / price_then        # realized % move
raw = r          if direction == LONG
raw = -r         if direction == SHORT
raw = flat_threshold_pct - abs(r)   if direction == FLAT
score = confidence * tanh(raw / 5.0)
```

An agent's **round score** = mean of its per-signal scores.

Weight update (multiplicative-weights, the "decision power" mechanic):

```
w[a] *= exp(eta * round_score[a])    # winners gain, losers lose
normalize to sum 1
clamp each to [min_weight, max_weight], then renormalize once
```

Agents present in `weights` but missing from `agent_scores` keep their raw
weight through the update (score treated as 0). `equal_weights` returns
`1/len(names)` each.

Consensus: per symbol, mass(direction) = Σ weights[a] × confidence(a, symbol)
over agents voting that direction. Winning direction = argmax mass (tie →
FLAT). Confidence = winning_mass / total_mass (0 if total 0). `agent` field =
`"consensus"`, `rationale` = short vote summary, `price_at_signal` taken from
any contributing signal for that symbol.

Files owned: `src/arena/engine/*`, `tests/test_engine.py`.

---

## Unit 4 — Store & paper trading: `src/arena/store/`

```python
# db.py
class Store:
    def __init__(self, path: str = "arena.db"): ...   # creates schema if absent
    def close(self) -> None: ...

    # rounds & signals
    def create_round(self, as_of: datetime, horizon_hours: float) -> int: ...
    def add_signals(self, round_id: int, signals: list[Signal]) -> None: ...
    def pending_rounds(self, now: datetime, *, force: bool = False) -> list[dict]: ...
        # rounds with status PENDING whose (as_of + horizon) <= now; force=True returns all PENDING
        # each dict: {"id", "as_of": datetime, "horizon_hours": float}
    def signals_for_round(self, round_id: int) -> list[Signal]: ...
    def record_scores(self, round_id: int, scored: list[ScoredSignal],
                      round_scores: dict[str, float]) -> None: ...  # also marks round EVALUATED

    # weights ("decision power")
    def get_weights(self, agent_names: list[str]) -> dict[str, float]: ...
        # returns stored weights; initializes to equal weights for unseen names
    def set_weights(self, weights: dict[str, float]) -> None: ...

    # paper portfolios
    def get_equity(self, agent: str, *, start_equity: float = 10_000.0) -> float: ...
    def set_equity(self, agent: str, equity: float) -> None: ...

    # reporting
    def leaderboard(self) -> list[dict]: ...
        # one dict per agent incl. "consensus", sorted by weight desc:
        # {"agent", "weight", "rounds", "avg_score", "wins", "equity"}
        # "wins" = number of evaluated rounds where the agent had the top round_score
    def history(self, limit: int = 20) -> list[dict]: ...
        # recent evaluated rounds: {"round_id", "as_of", "agent", "round_score"}
```

SQLite via stdlib `sqlite3`, one file, `check_same_thread=False` not required.
Store datetimes as ISO strings (UTC). Weights for `"consensus"` are NOT stored
in the weights table (it's derived), but its equity IS tracked.

Paper trading (`paper.py`):

```python
def apply_round_to_equity(equity: float, scored: list[ScoredSignal],
                          *, stake_fraction: float = 0.5) -> float: ...
```

Equal-weight stake of `equity * stake_fraction` across the agent's non-FLAT
signals; each position's PnL = stake_per_position × (r/100) × (+1 LONG / −1
SHORT), where r is the realized % move. FLAT signals hold cash. Returns new equity.

Files owned: `src/arena/store/*`, `tests/test_store.py`.

---

## Unit 5 — CLI & docs: `src/arena/cli.py`, `README.md`

`argparse` CLI, entry point `arena` (see `pyproject.toml`). Global flags:
`--config PATH` (default `config.yaml`), `--db PATH` (overrides settings),
`--mock` (force MockAgents), `--offline` (fixture market data).

Subcommands:

- `arena run-round` — fetch snapshot (`fetch_top_coins`), build agents
  (`available_agents`), collect each agent's signals (catch `AgentError` →
  that agent sits the round out, warn), compute + store the **consensus**
  signals too, persist via `Store.create_round`/`add_signals`, print a rich
  table of all signals. Exits nonzero if fewer than 2 agents produced signals.
- `arena evaluate [--force]` — for each due pending round (`pending_rounds`),
  fetch `fetch_prices`, score every signal (`score_signal`), compute per-agent
  round scores (consensus excluded from weight updates), `record_scores`,
  update weights (`update_weights` with settings' eta/min/max), update each
  agent's and consensus's paper equity (`apply_round_to_equity`), print results.
- `arena leaderboard` — rich table from `Store.leaderboard()`: rank, agent,
  decision power (weight as %), rounds, avg score, wins, paper equity.
- `arena history` — recent round results.
- `arena loop --interval-mins N` — forever: evaluate due rounds, run a round,
  sleep N minutes (default 60). Ctrl-C exits cleanly.

CLI imports all other modules **per the signatures in this document** — they
are developed in parallel, so code strictly to the contract. CLI tests
(`tests/test_cli.py`) must not import the other modules at runtime: inject
stubs via `sys.modules` / monkeypatch and test argument parsing + orchestration
logic only.

README.md (owned by unit 5): project intro ("LLMs fight for decision power"),
architecture diagram (ASCII), quickstart (install, .env, mock demo commands,
real run), how scoring/weights work, config reference, disclaimer that this is
research/paper-trading software, not financial advice.

Files owned: `src/arena/cli.py`, `tests/test_cli.py`, `README.md`.
