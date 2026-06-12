# 100 Improvements — Road to a Top-1% Crypto Perp Bot

Research-backed roadmap for making the Crypto LLM Arena smarter. Synthesized
from four research streams: (A) the 2024–2026 LLM-trading-agent literature,
(B) forecasting / ensemble / online-learning theory, (C) professional crypto
perpetual-futures signals and data sources, and (D) quantitative trading
engineering and risk practice — plus (E) analysis of this codebase. Sources
are cited inline. Items are deduplicated and grouped; the priority roadmap is
at the end.

---

## I. Smarter agents — prompting, debate, memory (1–18)

1. **Add bull/bear debate before signal commitment** — for each coin, have one
   pass argue the bullish case and one the bearish case, then let each agent
   commit. Structured adversarial debate reduces confirmation bias and improved
   risk-adjusted returns in TradingAgents. [arXiv:2412.20138]
2. **Split reasoning into fact vs. subjectivity pathways** — separate factual
   analysis (price, on-chain, funding) from subjective analysis (sentiment,
   narrative), then combine; crypto-specific ablations show subjectivity helps
   in bulls, facts protect in bears. [FS-ReasoningAgent, arXiv:2410.12464]
3. **Give agents layered memory across rounds** — short-term (last 5 rounds),
   pattern memory (last 50), and meta-memory of each agent's own strengths;
   inject distilled lessons into the next prompt. [FinMem, arXiv:2311.13743]
4. **Add post-round self-reflection** — after each evaluation, each agent writes
   a short structured reflection (what worked / what failed / regime shift) that
   is fed into its next round prompt. [TradingGroup, arXiv:2508.17565]
5. **Add team-level reflection every ~10 rounds** — synthesize cross-agent
   insights (which signal types are working) and share them; dual-level
   reflection sped market adaptation in FinAgent. [arXiv:2402.18485]
6. **Use expert-blueprint chain-of-thought (FinCoT)** — structure the prompt as
   a fixed expert workflow (regime → technicals → sentiment → cross-validation →
   signal) instead of free-form "predict"; +17–20% accuracy in FinCoT.
   [arXiv:2506.16123]
7. **Specialize agent roles** — move from 4 generalists toward analyst roles
   (technical, sentiment/news, on-chain, macro, risk, contrarian, synthesizer),
   each fed role-tailored data; multi-role frameworks dominate single-prompt
   agents across the survey literature. [arXiv:2412.20138; arXiv:2408.06361]
8. **Route models to the roles they're best at** — assign each LLM the role it
   empirically wins at (e.g. strongest reasoner = synthesizer); learned routing
   beat equal weighting by 15–25% in TradExpert/LLMoE. [arXiv:2411.00782;
   arXiv:2501.09636]
9. **Add a critic/validation agent before signals are accepted** — an
   independent pass that fact-checks each rationale against the snapshot
   (e.g. claimed "RSI 72" must match), flags hallucinations, and can demand a
   revision. [arXiv:2412.20138; MIT hallucination-mitigation thesis 2025]
10. **Inject retrieved real-time context (RAG)** — top-k recent headlines,
    regulatory events, and on-chain anomalies retrieved per coin into the
    prompt; grounding cut financial-QA error dramatically in FinDER.
    [arXiv:2504.15800]
11. **Run event-driven special rounds** — detect major events (ETF flows, FOMC,
    unlocks, hacks) and trigger a dedicated reasoning round about that event's
    impact; event-driven framing improved direction accuracy 17.5% in Janus-Q.
    [arXiv:2602.19919]
12. **Use regime-matched in-context examples** — keep pools of past
    (snapshot → signal → outcome) demos per regime (bull/bear/chop) and prepend
    3–5 matching demos; regime-matched ICL strongly beat random demos.
    [arXiv:2603.10299]
13. **Self-consistency voting on low-confidence signals** — when an agent's
    confidence < ~0.65, sample 3 independent reasoning paths and majority-vote;
    selective self-consistency buys accuracy where it matters at 3x cost only
    on the uncertain slice. [arXiv:2305.11860]
14. **Add a "to trade or not to trade" gate** — before publishing consensus,
    skip the round (all-FLAT) when aggregate confidence and regime certainty
    are below threshold; decision gates improved Sharpe 12–15%.
    [arXiv:2507.08584]
15. **Penalize herding in consensus** — when all agents agree, haircut the
    consensus confidence (×~0.85); LLM ensembles show correlated-consensus
    bubbles. [arXiv:2509.04537]
16. **Describe chart patterns in text** — add pattern annotations ("bullish
    divergence", "double top forming") and a compact ASCII price sketch to the
    snapshot; multimodal-style inputs improved predictions ~36% in FinAgent.
    [arXiv:2402.18485]
17. **Adaptive prompt optimization** — periodically rewrite prompt wording
    based on win/loss feedback (OPRO-style), keeping the champion prompt per
    agent; ATLAS reports 15–25% Sharpe gains over fixed prompts.
    [arXiv:2510.15949]
18. **Add a fast multi-timeframe technical sub-team** — 1h/4h indicator, trend
    and pattern agents whose outputs feed the 24h prompt as extra features,
    QuantAgent-style. [arXiv:2509.09995]

## II. Scoring, calibration, weights & consensus (19–42)

19. **Elicit full probability vectors** — replace direction+confidence with
    P(LONG), P(SHORT), P(FLAT) summing to 1; prerequisite for proper scoring.
    [Gneiting & Raftery 2007]
20. **Score with the Brier score (strictly proper)** — score = 1 − Σ(Pᵢ −
    outcomeᵢ)²; the current confidence×tanh score is not incentive-compatible
    (overconfidence can pay), Brier forces honest probabilities. [Gneiting &
    Raftery 2007, doi:10.1198/016214506000001437]
21. **Offer log-score mode** — log P(realized outcome) punishes confident
    misses exponentially and converges faster; make the rule configurable.
    [Murphy & Winkler 1992]
22. **Use token logprobs where the provider exposes them** — derive P from
    output logits (OpenAI-compatible `logprobs`) instead of parsing verbalized
    numbers, which are 10–30% miscalibrated. [Kadavath et al., arXiv:2207.05221]
23. **Calibrate each agent with Platt scaling, refit every ~20 rounds** —
    sigmoid(a·P+b) fit on the agent's history converts stated probability into
    empirical probability. [Niculescu-Mizil & Caruana 2005]
24. **Upgrade to isotonic regression once history is long enough** — LLM
    calibration curves are nonlinear; isotonic is distribution-free and beats
    Platt with ≥~30 rounds of data. [arXiv:2309.01173]
25. **Damp confidence in the first ~20 rounds** — P_damped = 0.7·P + 0.3·(1/3)
    until calibration data exists; LLMs are worst-calibrated out-of-domain.
    [arXiv:2406.16366]
26. **Track per-agent calibration curves and flag drift** — bucket predictions
    into deciles vs realized win rate; flag |gap| > 0.15 and surface Brier /
    log-loss / slope on the leaderboard. [Gneiting & Ranjan 2012]
27. **Penalize systematic overconfidence** — if rolling mean P > 0.6 while win
    rate < 0.5, haircut that agent's weight ×0.9 until calibration recovers.
    [Murphy & Winkler 1987]
28. **Use a time-decaying learning rate** — replace fixed η=0.35 with
    η(t)=√(ln N / t) (anytime-optimal Hedge); big early moves, fine-grained
    adjustments later. [Cesa-Bianchi & Lugosi 2006; Abernethy et al. 2008]
29. **Treat absent agents as sleeping experts** — an agent that errors out or
    abstains keeps its weight untouched (already partially true — formalize and
    test it); preserves regret bounds under partial participation.
    [Freund & Schapire 1997]
30. **Clip the per-round weight multiplier to [0.5, 2.0]** — prevents one
    extreme round from whipsawing the power structure. [Cesa-Bianchi & Lugosi
    2006, ch. 2]
31. **Maintain a per-asset weight matrix W[coin][agent]** — agents have
    asset-specific skill (good on BTC, bad on alts); update and use per-coin
    weights for consensus (EXP4-style contextual experts). [Auer et al. 2002;
    Pesaran & Timmermann 2007]
32. **Add regime-conditional weights** — keep W_bull/W_bear/W_chop per agent
    and select by detected regime; expert skill is regime-dependent.
    [Gu, Kelly & Xiu 2023]
33. **Extremize the consensus probability** — p^λ/(p^λ+(1−p)^λ) with λ≈1.3–1.7
    counteracts the aggregate's pull toward 0.5; standard Good Judgment Project
    practice. [Tetlock & Gardner, *Superforecasting*]
34. **Use robust aggregation under high disagreement** — when weight entropy is
    high, fall back to median/trimmed vote instead of the weighted mean.
    [Jose et al. 2014]
35. **Publish a confidence interval with the consensus** — Dirichlet-based
    [p_low, p_high] alongside the point estimate so position sizing can respond
    to uncertainty. [Gneiting & Ranjan 2012]
36. **Weight for diversity, not just accuracy** — penalize highly correlated
    agent pairs (|ρ|>0.8 → haircut the weaker) and/or use inverse-covariance
    weighting; clones add no information. [Kuncheva & Whitaker 2003; Krogh &
    Vedelsby 1995]
37. **Grant a dissent bonus** — small weight bonus proportional to
    KL(P_agent ‖ P_consensus) when the dissenter is later proven right;
    rewards principled disagreement. [Brown et al. 2005]
38. **Monitor effective sample size of the ensemble** — ESS = (Σw)²/Σw²; if
    ESS < ~1.5 of 4, the arena has collapsed onto one brain — raise the floor
    or freeze updates. [Kish 1965]
39. **Damp weight updates until skill is statistically significant** — deflate
    early performance (Bailey–López de Prado deflated Sharpe) and blend 50/50
    with old weights for the first ~30 rounds. [SSRN 2460551]
40. **Run a bootstrap reality check every ~25 rounds** — permutation-test each
    agent's score stream against luck; freeze weight movement for agents whose
    edge isn't distinguishable from noise. [White 2000 "Reality Check"]
41. **Track empirical regret vs the O(√(T ln N)) bound** — log cumulative
    best-agent-minus-consensus gap; a blowout signals broken scoring or
    correlated failures. [Cesa-Bianchi & Lugosi 2006]
42. **Experiment with an LMSR prediction market as the consensus mechanism** —
    agents "bet" weight-scaled stakes with a logarithmic market-scoring-rule
    market maker; incentive-compatible with O(log T) information aggregation,
    and market depth doubles as a confidence readout. [Hanson 2007; Othman &
    Sandholm 2010]

## III. Market data & perp-specific signals (43–64)

43. **Add funding rate level + 7d history per coin** — extreme funding
    (>±0.05%/8h) marks crowded leverage and is a classic contrarian reversal
    setup. Binance `/fapi/v1/fundingRate`, free. 
44. **Add funding-rate momentum** — the derivative of funding over 8h/24h;
    accelerating positive funding flags squeeze risk before price turns.
45. **Add open interest and ΔOI 24h** — OI expansion confirms trends; OI
    rising while price stalls warns of reversal. Binance `/fapi/v1/openInterest`.
46. **Add OI/market-cap leverage ratio** — >~1% marks extreme system leverage
    and crash risk; combine existing CoinGecko caps with Binance OI.
47. **Add long/short account ratio** — >70% longs is a contrarian liquidation
    setup. Binance `/fapi/v1/globalLongShortAccountRatio` (free).
48. **Add liquidation data and level clustering** — cluster recent liquidations
    by price to map cascade zones near current price. Binance liquidation
    endpoints/stream (free Coinglass alternative).
49. **Add basis (perp mark vs spot) and its trend** — basis compression/blowout
    reveals structural bull/bear positioning and arb unwinds. Binance
    `/fapi/v1/markPrice` + spot.
50. **Add implied volatility (Deribit DVOL) for BTC/ETH** — IV vs realized vol
    spread shifts the expected-move distribution agents reason over. Deribit
    public API, no key.
51. **Add orderbook depth imbalance** — bid/ask volume ratio over top levels;
    persistent ≥1.5× imbalance has measurable short-horizon directional edge.
    Binance `/api/v3/depth`.
52. **Add taker buy/sell ratio (CVD)** — classify aggressor side from recent
    trades; CVD-vs-price divergence is a reversal warning.
    Binance `/fapi/v1/aggTrades` or `takerlongshortRatio`.
53. **Add exchange net in/outflows (7d trend)** — sustained outflows =
    accumulation, inflow spikes = distribution. Glassnode/CryptoQuant free
    tiers.
54. **Add stablecoin supply change** — net USDT/USDC minting is dry powder;
    contraction is risk-off. CryptoQuant free tier / DefiLlama stablecoins API.
55. **Add whale-transfer detection** — large transfers to/from exchanges lead
    regime changes by hours. CryptoQuant free tier or chain RPCs.
56. **Add MVRV and SOPR for BTC/ETH** — MVRV >2 historically precedes negative
    24–48h returns ~70% of the time; SOPR <1 marks capitulation. Glassnode free
    tier.
57. **Add Fear & Greed index + 7d trend** — extremes (<20, >80) mean-revert
    60–70% of the time within 1–2 days. alternative.me API, free, no key.
58. **Add BTC dominance and ETH/BTC** — the season indicator: rising dominance
    = de-risking into BTC; falling = altseason. CoinGecko `/global` (already
    integrated source).
59. **Add macro context flags** — DXY and S&P/Nasdaq daily direction; crypto
    trades risk-on/risk-off with them. yfinance or FMP, free.
60. **Add social volume anomaly detection** — social-volume spike with price
    divergence flags retail FOMO/panic 12–24h early. Santiment free tier.
61. **Add ADX trend-strength regime filter** — ADX>25 → trust momentum signals;
    ADX<20 → trust mean-reversion; tell the agents which regime they're in.
    Computed from existing klines.
62. **Add Hurst exponent for trend-vs-chop classification** — H>0.5 momentum
    regime, H<0.5 mean-reverting; drives regime-conditional weights (item 32).
63. **Add richer derived technicals** — MACD histogram momentum, Stoch-RSI,
    Bollinger-squeeze width, EMA-ribbon slope, ATR-relative range, RSI
    *divergence* (level + price direction), pivot/Fib levels — all computable
    from free klines and strictly more informative than the current 3
    indicators.
64. **Adopt a tiered free data stack with fallbacks** — Tier 1 no-key (Binance
    futures, CoinGecko, alternative.me, Deribit), Tier 2 free-key (Glassnode,
    CryptoQuant, Santiment); multi-source redundancy both de-risks outages and
    enables cross-source confirmation.

## IV. Trading realism, risk & portfolio (65–82)

65. **Model taker/maker fees in paper PnL** — deduct ~4–5bps per side
    (configurable) in `apply_round_to_equity`; fee-free backtests flatter
    every strategy. [Binance fee schedule]
66. **Accrue funding payments on held positions** — 8h funding transfers are
    first-order PnL for perps; fetch historical rates and charge/credit
    positions. [Binance funding API]
67. **Add a slippage model** — slippage as a function of position size vs
    daily volume and volatility, deducted from entry/exit. [Kissell & Glantz]
68. **Track liquidation price and margin ratio per position** — model leverage
    properly: force-close at liquidation price if crossed intra-round, alert on
    thin margin.
69. **Stress-test with intrabar wicks** — inject 2–5% adverse wicks in high-vol
    rounds to catch positions that survive on closes but die intraday.
70. **Use fractional Kelly position sizing** — size = ¼–½ Kelly from each
    agent's rolling win rate and win/loss ratio instead of the flat 50% stake;
    full Kelly is ruinous under estimation error. [Thorp; MacLean et al. 2010]
71. **Add volatility targeting** — scale gross exposure to hit a target
    portfolio vol (e.g. 20% annualized) so calm and wild markets carry equal
    risk. [vol-targeting literature]
72. **Cap per-coin and total exposure** — max position % per coin, max
    concurrent positions, and max gross leverage as config limits enforced
    before fills.
73. **Make sizing correlation-aware** — top-20 alts are ~0.8-correlated to BTC;
    shrink size when the new position correlates with the existing book, and
    cap portfolio BTC-beta. [Markowitz/MPT]
74. **Scale down in drawdown** — halve new position sizes when equity drawdown
    exceeds a threshold (e.g. 10%) until recovery; convexity against ruin.
75. **Add per-position stop-loss / take-profit** — configurable SL/TP checked
    against price paths between rounds, not just round boundaries.
76. **Add daily-loss circuit breaker and kill switch** — stop opening positions
    after a configurable daily loss (e.g. 2%); require manual reset above a
    max-drawdown threshold (e.g. 15%) with an alert.
77. **Pre-trade risk checks** — validate balance, post-trade margin, leverage
    and liquidation distance before any (paper or live) order; reject loudly.
78. **Screen out illiquid coins** — skip signals where volume/market-cap <1%
    or spread >50bps; the bottom of the top-20 is sometimes untradeable size.
79. **Build a rank-based market-neutral mode** — long the strongest-conviction
    longs, short the weakest, BTC-beta-neutral; isolates coin-selection alpha
    from the (dominant) market factor.
80. **Add a CCXT exchange layer** — `ExchangeConnector` abstraction (orderbook,
    balance, orders, positions) implemented for Binance USDM + Bybit; this is
    the bridge from paper to live. [docs.ccxt.com]
81. **Execute limit-first with market fallback and TWAP for size** — post at
    mid±1 tick, escalate to market on timeout; slice large orders over time.
    [Almgren-Chriss]
82. **Add an exchange-testnet mode** — run the full loop against Binance/Bybit
    futures testnets to validate execution mechanics with zero capital at risk.

## V. Backtesting & evaluation (83–90)

83. **Build a historical replay backtester** — replay archived snapshots
    through the full agent→consensus→PnL pipeline so changes can be evaluated
    on months of data in minutes, with walk-forward (expanding train/test)
    windows. [López de Prado, *Advances in Financial ML*]
84. **Audit for lookahead and information leakage** — enforce that every input
    existed at signal time (embargo news/candles); "Profit Mirage" showed most
    LLM-trading wins evaporate after strict de-biasing. [arXiv:2510.07920]
85. **Handle survivorship in the universe** — record the top-20 list as-of each
    round (already snapshotted — extend to backtests) so delisted/fallen coins
    aren't silently excluded.
86. **Report deflated Sharpe and out-of-sample gaps** — correct for multiple
    testing across agents/configs; flag overfitting when train/test Sharpe gap
    >0.3. [Bailey & López de Prado, SSRN 2460551]
87. **Add the full metric suite** — rolling Sharpe, Sortino, Calmar, hit rate,
    profit factor, turnover, exposure, max drawdown per agent and consensus, on
    the leaderboard and dashboard.
88. **Benchmark against BTC buy-and-hold and naive baselines** — also against
    a momentum baseline and a random-signal baseline; alpha is outperformance,
    not raw return.
89. **Report performance split by regime** — bull/bear/chop attribution per
    agent; reveals which fighter earns its decision power where.
90. **Emit a round-by-round explainability report** — signals, debate outcome,
    consensus, realized result, weight changes — exportable as Markdown/HTML;
    interpretability is how you debug an LLM strategy. [arXiv:2505.24650]

## VI. Engineering, ops & cost (91–100)

91. **Parallelize agent calls** — collect all four agents' signals concurrently
    (threads/async); cuts round latency ~4x. (`arena/cli.py`)
92. **Use structured outputs end-to-end** — Anthropic `output_config` JSON
    schema and OpenAI-compatible `response_format`; malformed-JSON fallbacks
    become impossible. (`arena/agents/*`)
93. **Cache the static prompt prefix** — keep instructions byte-stable, put the
    volatile snapshot last, enable Anthropic prompt caching (~90% input-cost
    cut on repeated rounds); track tokens + $ per agent per round and show
    cost-adjusted alpha on the leaderboard.
94. **Retry transient API failures with backoff, bound by a per-agent time
    budget** — one 429 shouldn't bench a fighter; one hung provider shouldn't
    stall the round.
95. **Persist raw LLM replies and snapshot hashes** — store full reply text and
    the exact snapshot JSON per round for audits, replays, and feature-importance
    analysis.
96. **Make rounds idempotent and crash-safe** — write signals as they arrive,
    resume interrupted rounds, and recover open paper positions after restart
    (graceful SIGTERM handling).
97. **Adopt structured JSON logging + metrics + dashboard** — `logging` with
    JSON lines, Prometheus counters/gauges (equity, weights, scores, latency),
    and a Grafana (or simple web) dashboard with equity curves and weight
    history.
98. **Add Telegram/Discord alerting** — push consensus signals, weight shifts
    >10%, circuit-breaker trips, API failures, and data-quality anomalies to a
    channel; this is also the product surface of the bot.
99. **Version configs and back up the database** — config hash recorded per
    round for reproducibility; timestamped SQLite backups with a `restore`
    command; secrets only via env (never in repo).
100. **A/B test everything through the arena itself** — every improvement above
     can be fielded as a *new fighter* (same model, new prompt/data/scoring)
     that must win decision power from the incumbents before being trusted —
     the arena is its own experiment harness. Add config-defined prompt/data
     variants per agent to make this one-line.

---

## Priority roadmap

**Phase 1 — quick wins (days):** 19–20 (proper scoring), 28 (decaying η),
43–47 (funding/OI/long-short), 57 (Fear & Greed), 61 (ADX regime), 65–66
(fees+funding PnL), 91–94 (parallel, structured outputs, caching, retries).

**Phase 2 — core upgrades (weeks):** 1–7 (debate, fact/subjectivity, memory,
reflection, roles), 23–27 (calibration), 31–33 (per-asset + regime weights,
extremizing), 48–52 (liquidations, basis, IV, orderbook), 70–78 (sizing & risk
controls), 83–88 (backtester + honest metrics), 97–98 (observability, alerts).

**Phase 3 — frontier (months):** 8, 17, 42 (model routing, prompt optimization,
LMSR market), 53–56/60 (on-chain + social), 79–82 (market-neutral mode, CCXT,
testnet → live), 40–41 (statistical skill gates), 100 (arena-as-experiment-
harness culture).

## Key sources

- TradingAgents — https://arxiv.org/abs/2412.20138 · FS-ReasoningAgent —
  https://arxiv.org/abs/2410.12464 · FinMem — https://arxiv.org/abs/2311.13743 ·
  FinAgent — https://arxiv.org/abs/2402.18485 · TradingGroup —
  https://arxiv.org/abs/2508.17565 · FinCoT — https://arxiv.org/abs/2506.16123 ·
  ATLAS — https://arxiv.org/abs/2510.15949 · Profit Mirage —
  https://arxiv.org/abs/2510.07920 · surveys https://arxiv.org/abs/2408.06361,
  https://arxiv.org/abs/2507.01990
- Gneiting & Raftery 2007 (proper scoring) · Cesa-Bianchi & Lugosi 2006
  (*Prediction, Learning, and Games*) · Tetlock & Gardner (*Superforecasting*) ·
  Hanson 2007 (LMSR) · Bailey & López de Prado (deflated Sharpe, SSRN 2460551) ·
  White 2000 (Reality Check)
- Binance Futures API — https://binance-docs.github.io/apidocs/futures/en/ ·
  Deribit — https://docs.deribit.com/ · alternative.me F&G —
  https://alternative.me/crypto/fear-and-greed-index/ · Glassnode/CryptoQuant/
  Santiment free tiers · CCXT — https://docs.ccxt.com/
- López de Prado, *Advances in Financial Machine Learning* · Almgren & Chriss
  (optimal execution) · Thorp / MacLean et al. (Kelly criterion)

---

## Implementation status (post wave-2 build-out)

**Fully implemented & tested (85):**
1–7, 9–17, 19–21, 23–47, 49–52, 54, 57–59, 61–67, 69–72, 74–75, 77–78,
80–84, 86–87, 89–92, 94–100 — plus 29 (absent agents keep weights), 41
(regret in backtest), 42 (LMSR consensus mechanism, config-gated), 56
(MVRV/SOPR, activates with a free Glassnode key).

**Partial or simplified (13):**
- 8 — PPO weight-learning substituted by the multiplicative engine +
  per-asset routing + OPRO-lite prompt optimizer.
- 18 — multi-timeframe features (1h ADX, technicals) instead of a separate
  HF sub-team of agents.
- 48 — basis/IV/orderbook landed; liquidation *clustering* not yet.
- 53, 55, 60 — exchange netflows, whale transfers, social volume: stubs/
  key-gated design in `arena.data.onchain`, fetchers not yet written.
- 68 — margin/leverage pre-trade checks exist; full liquidation-price
  simulation does not.
- 73 — diversity/decorrelation acts on weights; correlation-shrunk *sizing*
  not yet.
- 76 — circuit-breaker states computed and warned; hard halt enforcement in
  `loop` not wired.
- 79 — market-neutral book builder + config flag exist; not yet applied to
  the consensus book in `evaluate`.
- 85 — survivorship handled for journaled replays; no historical top-20
  backfill.
- 88 — backtest benchmarks agents against each other; explicit BTC
  buy-and-hold benchmark column not yet.

**Not applicable by design (2):**
- 22 — token-logprob probabilities don't map onto a 20-coin JSON reply;
  verbalized probabilities + Brier + Platt calibration achieve the goal.
- 93 (caching half) — arena prompts are below Anthropic's minimum cacheable
  prefix, so prompt caching cannot trigger; the cost-tracking half (tokens +
  `arena costs`) is implemented.
