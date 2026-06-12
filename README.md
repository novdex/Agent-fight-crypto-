# Crypto LLM Arena

**LLMs fight for decision power over crypto signals.**

Three to four large language models — Claude, GPT, Gemini and Grok — compete to
produce the most accurate LONG / SHORT / FLAT calls on the top-20
cryptocurrencies (by market cap, stablecoins excluded). Every round, each
fighter reads the same market snapshot and commits to a signal per coin.
Twenty-four hours later the market grades them: accurate agents *gain* weight
("decision power"), inaccurate ones *lose* it. Those weights drive a
**weighted consensus signal** — the arena's collective call — and every agent
(plus the consensus itself) runs a paper-trading portfolio so you can watch
equity curves diverge over time.

> **This is research / paper-trading software, not financial advice.**
> See the [Disclaimer](#disclaimer).

## How the arena works

1. **Round** — `arena run-round` fetches a snapshot of the top-20 coins from
   CoinGecko (price, 1h/24h/7d change, RSI-14, EMA-20 distance, 24h
   volatility) and asks every agent for one signal per coin:
   direction (`LONG` / `SHORT` / `FLAT`) plus a confidence in `[0, 1]`.
   If one LLM errors out or is rate-limited, it simply sits the round out —
   the fight goes on (as long as at least 2 agents answered).
2. **Consensus** — agents' signals are combined per coin: each direction's
   "mass" is the sum of `weight(agent) × confidence` over agents voting that
   way; the heaviest direction wins (ties go FLAT). The consensus is stored as
   a fifth fighter named `consensus`.
3. **Evaluation (24h later)** — `arena evaluate` fetches current prices and
   scores every signal of every due round. The exact scoring formula
   (deterministic, bounded to `[-1, 1]`):

   ```
   r = 100 * (price_now - price_then) / price_then        # realized % move
   raw = r          if direction == LONG
   raw = -r         if direction == SHORT
   raw = flat_threshold_pct - abs(r)   if direction == FLAT
   score = confidence * tanh(raw / 5.0)
   ```

   An agent's **round score** is the mean of its per-signal scores. Calling a
   big move correctly with high confidence scores near `+1`; calling it wrong
   scores near `-1`; a FLAT call is rewarded only if the move stayed inside
   `flat_threshold_pct`.
4. **Multiplicative weight update** — the "decision power" mechanic:

   ```
   w[a] *= exp(eta * round_score[a])    # winners gain, losers lose
   normalize to sum 1
   project onto [min_weight, max_weight] so the bounds hold exactly
   and the weights still sum to 1
   ```

   The consensus is *excluded* from weight updates (it is derived, it doesn't
   compete for power), but its paper equity *is* tracked alongside the agents'.
5. **Paper trading** — after each evaluated round, every agent (and the
   consensus) stakes `stake_fraction` (50%) of its equity equally across its
   non-FLAT signals; PnL follows the realized moves. Everyone starts at
   $10,000 of strictly imaginary money.

## Architecture

```
                       +--------------------+
                       |  CoinGecko API     |
                       |  (top-20 snapshot, |
                       |   later prices)    |
                       +---------+----------+
                                 |
                          arena.data (Unit 1)
                                 |
                                 v  MarketSnapshot
   +------------+---+------------+---+------------+---+------------+
   |   claude   |   |    gpt     |   |   gemini   |   |    grok    |
   | (anthropic)|   | (openai_   |   | (openai_   |   | (openai_   |
   |            |   |  compat)   |   |  compat)   |   |  compat)   |
   +-----+------+---+-----+------+---+-----+------+---+-----+------+
         |                |               |               |
         +--------+-------+-------+-------+-------+-------+
                  |  signals      |  weights      |
                  v               v               |
             arena.engine (Unit 3)                |
             scoring | multiplicative weights | consensus
                  |                               |
                  v                               v
             arena.store (Unit 4) -- SQLite: rounds, signals,
             scores, weights ("decision power"), paper equity
                  |
                  v
             arena.cli (Unit 5)
             run-round | evaluate | leaderboard | history | loop
```

(The agents live in `arena.agents`, Unit 2 — one `Signal` per coin per agent.)

## Quickstart

```bash
git clone <this-repo> && cd <this-repo>
pip install -e .
```

### Demo without any API keys

Mock agents (deterministic) plus offline fixture market data — no network,
no keys:

```bash
arena run-round --mock --offline && arena evaluate --offline --force && arena leaderboard
```

### Real usage

1. Copy `.env.example` → `.env` and add keys for the agents you want to field
   (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`).
   Agents with a missing key are skipped with a warning.
2. Run one round, come back ~24h later and evaluate:

   ```bash
   arena run-round
   # ... 24 hours pass ...
   arena evaluate
   arena leaderboard
   ```

3. Or let it fight unattended (evaluates due rounds, starts a new round,
   sleeps, repeats; Ctrl-C exits cleanly):

   ```bash
   arena loop --interval-mins 60
   ```

### Commands & global flags

| Command | What it does |
|---|---|
| `arena run-round` | Fetch snapshot, collect signals from every agent + consensus, store the round, print the signals table. Exits nonzero if fewer than 2 agents produced signals. |
| `arena evaluate [--force]` | Score every due round, update decision power and paper equity, print per-agent results (`--force` evaluates pending rounds before the 24h horizon). |
| `arena leaderboard` | Rank, agent, decision power %, rounds, avg score, wins, paper equity. |
| `arena history` | Recent evaluated rounds and per-agent round scores. |
| `arena loop --interval-mins N` | Evaluate → run-round → sleep N minutes, forever (default 60). |

Global flags (accepted before or after the subcommand):

| Flag | Meaning |
|---|---|
| `--config PATH` | Config file (default `config.yaml`). |
| `--db PATH` | SQLite database path (overrides the config's `db_path`). |
| `--mock` | Force deterministic MockAgents — no API keys needed. |
| `--offline` | Use the bundled fixture market data instead of the live API. |

## `config.yaml` reference

| Key | Default | Meaning |
|---|---|---|
| `arena.top_n_coins` | `20` | Coins per round (top by market cap, stablecoins excluded). |
| `arena.horizon_hours` | `24` | Hours between a round's signals and their evaluation. |
| `arena.eta` | `0.35` | Weight-update learning rate — higher means power shifts faster. |
| `arena.min_weight` | `0.05` | No agent's decision power ever drops below this share. |
| `arena.max_weight` | `0.60` | ...or rises above this share. |
| `arena.flat_threshold_pct` | `1.0` | Moves smaller than this (in %) count as "flat" and reward FLAT calls. |
| `arena.db_path` | `arena.db` | SQLite file holding rounds, signals, weights and equity. |
| `arena.start_equity` | `10000` | Paper-trading starting equity per agent. |
| `agents` | 4 fighters | List of agent specs — see below. |

## The fighters

The default roster in `config.yaml`:

| Name | Provider | Model | API key env var |
|---|---|---|---|
| `claude` | `anthropic` | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| `gpt` | `openai_compat` | `gpt-5` | `OPENAI_API_KEY` |
| `gemini` | `openai_compat` | `gemini-2.5-pro` | `GEMINI_API_KEY` |
| `grok` | `openai_compat` | `grok-4` | `XAI_API_KEY` |

**Swapping models** is a config edit: any provider with an OpenAI-compatible
`/chat/completions` endpoint works via `openai_compat` — set `name`, `model`,
`base_url` and `api_key_env`. For example, to field a local model:

```yaml
agents:
  - name: local-llama
    provider: openai_compat
    base_url: http://localhost:11434/v1
    model: llama3.1:70b
    api_key_env: OLLAMA_API_KEY   # any non-empty env var will do
```

There is also a `mock` provider (deterministic, offline) used by `--mock`.
You can field 3, 4 or more fighters; the arena only requires that at least 2
produce signals each round.

## FAQ

**Do I need API keys to try it?**
No — `arena run-round --mock --offline` runs the full pipeline with
deterministic mock agents and fixture market data.

**Why did an agent miss a round?**
Per-agent API failures (rate limits, timeouts, unparseable replies) make that
agent sit the round out with a warning; the round proceeds if at least 2
agents answered.

**What does "decision power" mean concretely?**
It is the agent's normalized weight in the consensus vote. Weights move
multiplicatively with round scores (`w *= exp(eta * score)`), are clamped to
`[min_weight, max_weight]`, and always sum to 1.

**Why can't an agent's power hit 0 or 100%?**
The clamp keeps the arena competitive: a slumping agent can stage a comeback,
and no single agent can fully dictate the consensus.

**Is the consensus an agent?**
Half of one: its signals are stored and paper-traded under the name
`consensus` and it appears on the leaderboard — but it never receives a
weight, since it is derived from the others.

**Where is everything stored?**
A single SQLite file (`arena.db` by default): rounds, signals, scores,
weights and paper equity. Delete it to restart the tournament.

**Does it place real trades?**
No. The arena only fetches public market data and tracks imaginary paper
portfolios. There is no exchange connectivity of any kind.

**Is 24h-ahead crypto prediction by LLMs actually a good idea?**
That is exactly the research question. Expect a lot of noise; the fun is in
the relative comparison and the weight dynamics, not in getting rich.

## Disclaimer

This project is **research and paper-trading software**. It exists to study
how LLMs behave in a competitive forecasting game. **Nothing it outputs is
financial advice.** Cryptocurrency markets are extremely volatile; LLM signals
can be confidently and persistently wrong. **Never wire this software to real
funds** or trade based on its signals without fully understanding the risks —
if you do so anyway, you alone bear the consequences.

## Deploy on a VPS

A one-shot installer for Debian/Ubuntu lives in `deploy/`:

```bash
# on the server, as root:
git clone <this-repo> /tmp/arena-src
sudo /tmp/arena-src/deploy/deploy.sh <this-repo-git-url> <branch>
# add your API keys:
sudo nano /opt/crypto-llm-arena/.env
sudo systemctl restart arena
# watch the fight:
journalctl -u arena -f
sudo -u arena /opt/crypto-llm-arena/.venv/bin/arena leaderboard
```

This installs to `/opt/crypto-llm-arena` under a dedicated non-login `arena`
user and runs `arena loop --interval-mins 60` as a systemd service
(auto-restarts on failure, survives reboots via `systemctl enable`).
