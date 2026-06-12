# 100 Improvements — Road to a Top-1% Crypto Perp Bot

> Research-backed roadmap for making the Crypto LLM Arena smarter. Sources are
> cited inline; items are grouped by category and tagged with the module they
> touch. DRAFT — being assembled; sections A–D come from a literature/industry
> survey, section E from codebase analysis.

## E. LLM & engineering layer (from codebase analysis)

E1. **Parallelize agent calls** — `run-round` queries agents sequentially; 4 LLM
    calls × ~30-60s each. Use `concurrent.futures.ThreadPoolExecutor` (or async
    clients) to collect all signals simultaneously. (`arena/cli.py`)
E2. **Structured outputs instead of JSON-by-prompt** — use Anthropic
    `output_config.format` (json_schema) and OpenAI-compatible
    `response_format`/strict JSON modes so signals can never be malformed,
    removing the all-FLAT fallback failure mode. (`arena/agents/*`)
E3. **Prompt caching** — keep the static instruction block byte-identical and
    put the volatile snapshot last; enable Anthropic `cache_control` to cut
    input cost ~90% on repeated round prompts. (`arena/agents/prompts.py`)
E4. **Retry with exponential backoff per agent** — one transient 429 currently
    benches an agent for the whole round; retry 2-3 times before raising
    `AgentError`. (`arena/agents/*`)
E5. **Per-agent timeout budget** — a hung provider stalls the entire round;
    enforce a hard wall-clock budget per agent and bench late responders.
E6. **Log raw LLM replies** — persist each agent's full response text per round
    (table or files) so bad parses and reasoning quality can be audited later.
E7. **Token/cost accounting** — record input/output tokens and $ cost per agent
    per round; add a `cost` column to the leaderboard (alpha per dollar matters).
E8. **Round idempotency & crash recovery** — if the process dies mid-round,
    signals collected so far are lost; write signals as they arrive and make
    `run-round` resumable.
E9. **Config-driven prompt variants** — allow per-agent system-prompt overrides
    in `config.yaml` so prompt A/B tests can run as different "fighters".
E10. **Snapshot hash in DB** — store a hash + full JSON of the exact snapshot
     each round saw, for reproducibility and later feature-importance analysis.
