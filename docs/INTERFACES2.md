# Frozen Module Interfaces — Wave 2 (the 100-improvements build-out)

Ten units are built in parallel worktrees against this contract plus
`src/arena/models.py` (read both first — models has the settings blocks:
`DebateSettings`, `MemorySettings`, `CalibrationSettings`, `WeightsSettings`,
`ConsensusSettings`, `RiskSettings`, `OpsSettings`, `ExchangeSettings`, and the
`extras`/`headlines` extension points on snapshots).

Rules for every unit:
- Own ONLY your listed files. Never edit `cli.py`, `prompts.py`, `db.py`,
  `config.yaml`, `README.md`, or another unit's files — the coordinator wires
  integration afterwards.
- Import only `arena.models`, `arena.config`, stdlib, `httpx`/`pydantic`/
  `rich`, and your own module. Optional heavy deps (`ccxt`,
  `prometheus_client`, telegram) must be imported lazily inside functions and
  the feature must degrade gracefully when missing.
- All tests offline (no network, no keys), `tests/test_<unit>.py` only.
- Every public entry point must be callable with plain models from
  `arena.models` — no global state except files/tables you own.
- SQLite: you may CREATE your own tables in the shared DB (namespaced
  `<unit>_*`), via your own helper class taking a `sqlite3.Connection` or path.

---

## U1 `src/arena/brain/` — debate, critic, reflection, memory, prompt optimizer

```python
# brain/debate.py
def debate_context(snapshot: MarketSnapshot, ask) -> str
    # ask: Callable[[str], str] — calls one LLM with a prompt, returns text.
    # Runs a bull-case pass and a bear-case pass over the snapshot (two ask()
    # calls), returns a compact "DEBATE BRIEF" text block to prepend to agent
    # prompts. Must truncate to <= ~2000 chars.
def critic_review(signals: list[Signal], snapshot: MarketSnapshot) -> list[Signal]
    # Deterministic fact-check (no LLM): flags/attenuates signals whose
    # rationale contradicts snapshot data (e.g. claims "oversold" while
    # RSI > 70 → halve confidence, note in rationale). Never drops coverage.

# brain/memory.py
class AgentMemory:
    def __init__(self, root: str, agent: str): ...
    def lessons(self, max_items: int = 8) -> list[str]      # for prompt injection
    def record_round(self, round_id: int, round_score: float, reflection: str) -> None
    def reflect(self, round_score: float, scored: list[ScoredSignal]) -> str
        # Deterministic template reflection (what worked / failed / regime note).

# brain/optimizer.py
class PromptVariantBook:  # champion/challenger prompt-preamble store (JSON file)
    def __init__(self, root: str): ...
    def preamble_for(self, agent: str) -> str
    def record_result(self, agent: str, round_score: float) -> None
        # OPRO-lite: tracks per-variant mean score, promotes the champion.
```
Files: `src/arena/brain/__init__.py` (re-exports), `debate.py`, `memory.py`,
`optimizer.py`, `tests/test_brain.py`.

## U2 `src/arena/agents/personas.py` + `promptkit.py` — roles & prompt upgrades

```python
# personas.py
ROLES: dict[str, str]  # name -> system-style persona preamble. Must include:
# "technical", "sentiment", "onchain", "macro", "risk", "contrarian",
# "synthesizer", and "generalist" (default).
def persona_preamble(role: str) -> str  # unknown role -> "generalist"

# promptkit.py
def fincot_blueprint(snapshot: MarketSnapshot) -> str
    # Expert-workflow scaffold: regime -> technicals -> positioning ->
    # sentiment -> cross-check -> probabilities. Returns a text block.
def fact_subjectivity_split() -> str
    # Instruction block demanding separate FACTS vs NARRATIVE analysis.
def chart_pattern_notes(snapshot: MarketSnapshot) -> str
    # Deterministic pattern annotations from numeric fields (RSI divergence
    # proxy, BB squeeze via extras, EMA posture...). "" when nothing notable.
def regime_demos(regime: str) -> str
    # 2-3 hardcoded few-shot exemplars per regime ("bull"|"bear"|"chop").
def detect_regime(snapshot: MarketSnapshot) -> str  # "bull"|"bear"|"chop"
    # From mean 24h/7d change + ADX/extras; deterministic.
def build_enhanced_prompt(snapshot: MarketSnapshot, *, role: str = "generalist",
                          memory_lessons: list[str] = [], debate_brief: str = "",
                          preamble: str = "") -> str
    # Composes: persona + preamble + fincot + fact/subjectivity + base market
    # data rendering (re-implement coin lines locally; do NOT import
    # arena.agents.prompts) + pattern notes + regime demos + memory + debate
    # brief + the SAME strict-JSON probability output contract as
    # arena/agents/prompts.py (p_long/p_short/p_flat per coin, one entry per
    # symbol, Brier note). Output parses with arena.agents.prompts.parse_signals.
```
Files: `src/arena/agents/personas.py`, `src/arena/agents/promptkit.py`,
`tests/test_promptkit.py`.

## U3 `src/arena/engine/calibration.py`

```python
class Calibrator:
    def __init__(self, method: str = "platt"): ...
    def fit(self, predictions: list[tuple[float, bool]]) -> None
        # (stated probability of the realized class... i.e. (p, hit)) pairs.
    def apply(self, p: float) -> float          # calibrated probability
    def calibrate_signal(self, sig: Signal) -> Signal
        # Applies to the prob vector (renormalized) when has_probs.

def early_damping(sig: Signal, rounds_seen: int, *, threshold: int) -> Signal
    # p' = 0.7 p + 0.3 (1/3) while rounds_seen < threshold.
def calibration_table(history: list[tuple[float, bool]], bins: int = 10) -> list[dict]
    # per-bin {p_mean, hit_rate, n}; for drift flags & leaderboard display.
def drift_flag(table: list[dict], gap: float = 0.15) -> bool
def overconfidence_flag(history: list[tuple[float, bool]]) -> bool
    # mean p > 0.6 and hit rate < 0.5 on the window.
def log_score(sig: Signal, outcome: Direction) -> float
    # log(P[outcome]) clamped to >= -6; alternative scoring rule.
```
Plus a tiny sqlite-backed history helper `CalibrationLog` (own table
`calib_predictions(agent, p, hit, round_id)`) with `add(...)`/`history(agent,
window)`.
Files: `src/arena/engine/calibration.py`, `tests/test_calibration.py`.

## U4 `src/arena/engine/weights2.py` + `src/arena/store/weights_ext.py`

```python
# weights2.py
def clip_update_factor(factor: float, low: float, high: float) -> float
def effective_sample_size(weights: dict[str, float]) -> float
def significance_damping(new_w, old_w, rounds_seen: int, threshold: int) -> dict
    # 50/50 blend while rounds_seen < threshold.
def decorrelation_adjust(weights, prediction_history) -> dict
    # prediction_history: dict[agent, list[tuple[symbol, Direction]]] recent;
    # penalize |corr|>0.8 pairs (haircut weaker), small KL-dissent bonus.
def update_weight_matrix(matrix: dict[str, dict[str, float]],
                         scores: dict[str, dict[str, float]],  # [coin][agent]
                         *, eta: float, min_weight: float, max_weight: float,
                         clip_low: float, clip_high: float) -> dict
    # Per-asset EXP4-style update; reuses the global update semantics per coin.
def regime_key(regime: str) -> str  # "bull"|"bear"|"chop" -> weight-book key
def bootstrap_skill_pvalue(scores: list[float], n_perm: int = 1000,
                           seed: int = 0) -> float
    # Sign-permutation test of mean score > 0; deterministic via seed.

# store/weights_ext.py
class WeightMatrixStore:  # own tables: pa_weights(book, coin, agent, weight)
    def __init__(self, path: str): ...
    def get(self, book: str, coins: list[str], agents: list[str]) -> dict[str, dict[str, float]]
    def set(self, book: str, matrix: dict[str, dict[str, float]]) -> None
    def close(self) -> None
```
Files above + `tests/test_weights2.py`.

## U5 `src/arena/engine/consensus2.py`

```python
def extremize(p: float, lam: float) -> float
def consensus_signals_v2(signals_by_agent, weights, *, settings: ConsensusSettings,
                         per_asset_weights: dict | None = None) -> list[Signal]
    # Weighted prob-vector pooling per symbol (mean of weighted normalized
    # vectors), then extremize the winning prob; herding haircut when all
    # agents agree on direction; per-asset weights override global when given;
    # trimmed pooling when trim_fraction > 0; "lmsr" mechanism option (LMSR
    # market over the three outcomes, agents bet weight*confidence).
    # Output Signals must carry the pooled prob vector (p_long/short/flat).
def no_trade_gate(consensus: list[Signal], min_confidence: float) -> tuple[list[Signal], bool]
    # When mean top-prob < threshold -> force all-FLAT, return (signals, gated=True).
def consensus_interval(signals_by_agent, symbol) -> tuple[float, float]
    # Crude 5th/95th percentile of agents' p(winning direction).
```
Files: `src/arena/engine/consensus2.py`, `tests/test_consensus2.py`.

## U6 `src/arena/data/` extensions — micro, onchain, macro, technicals, news

All fetchers: best-effort, single stderr warn per source, fill
`CoinSnapshot.extras` / `MarketSnapshot.extras` / `headlines`; never raise.
Pure computation helpers must be importable and unit-tested offline.

```python
# technicals.py (pure; from sparkline closes already on the snapshot pipeline
# or 1h klines passed in)
def hurst_exponent(closes: list[float]) -> float | None
def macd_histogram(closes: list[float]) -> float | None
def stoch_rsi(closes: list[float], period: int = 14) -> float | None
def bollinger_width_pct(closes: list[float], period: int = 20) -> float | None
def ema_ribbon_slope(closes: list[float]) -> float | None   # +1..-1-ish score
def enrich_technicals(snapshot: MarketSnapshot,
                      closes_by_symbol: dict[str, list[float]]) -> MarketSnapshot
    # writes extras: hurst, macd_hist, stoch_rsi, bb_width_pct, ema_ribbon

# micro.py (Binance fapi/spot public; canned-JSON parsers separate from fetch)
def parse_depth_imbalance(depth: dict, levels: int = 5) -> float | None
def parse_taker_ratio(rows: list[dict]) -> float | None
def parse_basis_pct(mark: float, spot: float) -> float
def enrich_micro(snapshot: MarketSnapshot) -> MarketSnapshot
    # extras: ob_imbalance, taker_buy_ratio, basis_pct, liq_24h_musd (from
    # forceOrders if reachable), dvol via Deribit for BTC/ETH (extras: iv_dvol)

# macro.py
def enrich_macro(snapshot: MarketSnapshot) -> MarketSnapshot
    # MarketSnapshot.extras: btc_dominance_pct (CoinGecko /global),
    # eth_btc_ratio; best-effort dxy/spx via stooq.com free CSV.

# onchain.py
def enrich_onchain(snapshot: MarketSnapshot) -> MarketSnapshot
    # extras (when free keys present in env GLASSNODE_API_KEY etc., else skip
    # silently): exch_netflow_musd, stablecoin_supply_chg_pct, mvrv, sopr.

# news.py
def fetch_headlines(limit: int = 8) -> list[str]
    # CryptoPanic free endpoint or CoinGecko status_updates or RSS; on failure [].
def enrich_news(snapshot: MarketSnapshot) -> MarketSnapshot
def detect_events(headlines: list[str]) -> list[str]
    # keyword scan (ETF, SEC, hack, FOMC, unlock...) -> event tags.
```
Files: those five + `tests/test_data2.py`.

## U7 `src/arena/risk/` — sizing, controls, portfolio

```python
# sizing.py
def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float  # clamped >= 0
def position_fractions(scored_history: dict[str, list[float]],  # agent -> past scores (proxy)
                       signals: list[Signal], equity: float,
                       settings: RiskSettings,
                       snapshot: MarketSnapshot | None = None) -> dict[str, float]
    # symbol -> stake fraction of equity. Applies: fractional Kelly base,
    # per-coin cap, gross-leverage cap, vol-targeting scalar (when extras
    # carry vol), liquidity screen (volume/mcap), drawdown scaling hook via
    # `drawdown_pct` param... keep signature: extra kw-only args allowed with
    # defaults.

# controls.py
def check_stops(entry: float, price_now: float, direction: Direction,
                *, stop_loss_pct: float, take_profit_pct: float) -> str
    # "open" | "stopped" | "took_profit" — exit price semantics documented.
def circuit_breaker_state(daily_pnl_pct: float, drawdown_pct: float,
                          settings: RiskSettings) -> str
    # "ok" | "daily_halt" | "drawdown_halt"
def pre_trade_checks(equity: float, stake_usd: float, gross_usd: float,
                     settings: RiskSettings) -> list[str]  # [] = pass, else reasons

# portfolio.py
def apply_round_to_equity_v2(equity: float, scored: list[ScoredSignal], *,
                             fractions: dict[str, float],
                             fee_rate: float, settings: RiskSettings,
                             funding_by_symbol: dict[str, float] | None = None,
                             horizon_hours: float = 24.0) -> dict
    # Returns {"equity": float, "fees": float, "funding": float,
    #          "slippage": float, "stopped": list[str]} — per-position stake
    # from `fractions`, slippage model (base_bps scaled by |move|), funding
    # accrual (rate%/8h * horizon/8 ... sign by direction), intra-round
    # SL/TP via check_stops using price path approximation (entry->eval).
def market_neutral_book(signals: list[Signal], top_k: int = 5) -> list[Signal]
    # strongest longs + weakest as shorts, equal legs; FLAT the rest.
```
Files: `src/arena/risk/__init__.py` (re-exports), three modules,
`tests/test_risk.py`.

## U8 `src/arena/backtest/` — replay, metrics, report

```python
# metrics.py (all pure, take a list of per-round returns or equity curve)
def sharpe(returns: list[float], periods_per_year: float = 365.0) -> float | None
def sortino(...) -> float | None
def calmar(equity_curve: list[float], periods_per_year: float = 365.0) -> float | None
def max_drawdown_pct(equity_curve: list[float]) -> float
def hit_rate(scores: list[float]) -> float | None
def profit_factor(returns: list[float]) -> float | None
def deflated_sharpe(sr: float, n_obs: int, n_trials: int = 4) -> float | None
def regime_split(rounds: list[dict]) -> dict[str, dict]
    # rounds: {"regime": str, "agent": str, "round_score": float} aggregation.

# replay.py
class ReplayBacktester:
    def __init__(self, snapshots: list[MarketSnapshot], agents): ...
    def run(self, *, flat_threshold_pct: float = 1.0, fee_rate: float = 0.0005,
            eta_adaptive: bool = True) -> dict
        # Replays snapshot[t] -> signals -> score vs snapshot[t+1] prices ->
        # weights; returns {"leaderboard": [...], "equity_curves": {...},
        # "regret": {...}}. Walk-forward by construction. Uses
        # arena.engine.score_signal / update_weights / adaptive_eta.
def synthetic_snapshots(n_rounds: int, n_coins: int = 5, seed: int = 0) -> list[MarketSnapshot]
    # Deterministic GBM-ish price paths for tests/demo.

# report.py
def round_report_markdown(round_id: int, signals: list[ScoredSignal],
                          round_scores: dict[str, float],
                          weights_before: dict, weights_after: dict) -> str
```
Files: three modules + `__init__.py` re-exports + `tests/test_backtest.py`.

## U9 `src/arena/exchange/` — connector abstraction

```python
class Order(BaseModel-like dataclass): symbol, side ("buy"|"sell"), qty,
    type ("limit"|"market"), price: float | None, status, fill_price | None
class ExchangeConnector(Protocol/ABC):
    def get_price(self, symbol: str) -> float
    def place_order(self, order: Order) -> Order
    def cancel_order(self, order_id: str) -> bool
    def get_balance(self) -> float
class PaperConnector(ExchangeConnector):  # in-memory, deterministic fills at
    # given prices + configurable slippage; used by tests and dry runs.
class CcxtConnector(ExchangeConnector):  # lazy `import ccxt`; binanceusdm |
    # bybit; testnet via set_sandbox_mode; raises a clear error if ccxt missing.
def execute_with_fallback(conn, order: Order, *, limit_timeout_s: float) -> Order
    # limit at price, on timeout cancel + market. PaperConnector simulates
    # timeout via a hook for tests.
def twap_orders(symbol: str, side: str, total_qty: float, slices: int) -> list[Order]
```
Files: `src/arena/exchange/__init__.py`, `connector.py`,
`tests/test_exchange.py`. Add `ccxt` ONLY as optional extra — do not edit
pyproject (coordinator adds `[project.optional-dependencies] live = ["ccxt"]`).

## U10 `src/arena/ops/` + `src/arena/store/journal.py`

```python
# ops/logging.py
def setup_logging(json_mode: bool = False) -> None  # std logging, JSON lines fmt
def log_event(event: str, **fields) -> None

# ops/alerts.py
def send_alert(text: str, *, ops: OpsSettings) -> bool
    # Telegram sendMessage and/or Discord webhook via httpx when env keys
    # present; returns False (no raise) otherwise. 5s timeout.
def alert_round_summary(round_scores: dict, weights: dict, *, ops) -> bool

# ops/backup.py
def backup_db(db_path: str, *, keep: int = 14) -> str | None  # timestamped copy
def config_hash(config_path: str) -> str  # sha256[:12] of file bytes ("" if absent)

# store/journal.py  (own tables: journal_rounds(round_id, snapshot_json,
# config_hash, created_at), journal_replies(round_id, agent, raw_text,
# input_tokens, output_tokens, cost_usd))
class Journal:
    def __init__(self, path: str): ...
    def record_snapshot(self, round_id: int, snapshot: MarketSnapshot,
                        config_hash: str = "") -> None
    def record_reply(self, round_id: int, agent: str, raw_text: str,
                     input_tokens: int = 0, output_tokens: int = 0,
                     cost_usd: float = 0.0) -> None
    def snapshot_for_round(self, round_id: int) -> MarketSnapshot | None
    def costs_by_agent(self) -> dict[str, float]
    def close(self) -> None
```
U10 ALSO owns `src/arena/agents/anthropic_agent.py` and
`src/arena/agents/openai_compat.py` for structured outputs (item 92):
Anthropic `output_config={"format": {"type": "json_schema", "schema": ...}}`
and OpenAI-compatible `response_format={"type": "json_object"}` (graceful:
on a 400 mentioning the parameter, retry once without it). Both must return
`(text, usage)` internally but keep `generate_signals` signature; add
`self.last_usage: dict` attribute (raw reply via `self.last_raw: str`).
Files: ops trio + journal + the two agent provider files +
`tests/test_ops.py`.

---

Coordinator wiring (NOT for workers): cli.py composes the pipeline, prompts.py
gains promptkit hooks, config.yaml/README document the new blocks, pyproject
gains optional extras.
