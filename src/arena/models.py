"""Core data models shared by every arena module.

These are FROZEN INTERFACES: all modules (data, agents, engine, store, cli)
code against these types. Do not change field names or semantics without
updating docs/INTERFACES.md and every consumer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class CoinSnapshot(BaseModel):
    """Market state for one coin at snapshot time."""

    symbol: str  # uppercase ticker, e.g. "BTC"
    name: str = ""
    price_usd: float
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    change_1h_pct: Optional[float] = None
    change_24h_pct: Optional[float] = None
    change_7d_pct: Optional[float] = None
    rsi_14: Optional[float] = None  # hourly RSI from 7d sparkline
    ema_20_dist_pct: Optional[float] = None  # % distance of price from 20-period EMA
    volatility_24h_pct: Optional[float] = None  # stdev of hourly returns * 100
    # Perp-specific enrichment (None when the source is unavailable or the
    # coin has no USDT perpetual) — see arena.data.perp.
    funding_rate_pct: Optional[float] = None  # current funding, % per 8h
    funding_7d_avg_pct: Optional[float] = None  # 7d mean funding, % per 8h
    open_interest_usd: Optional[float] = None
    long_short_ratio: Optional[float] = None  # global accounts long/short
    adx_14: Optional[float] = None  # 1h ADX: >25 trending, <20 choppy
    # Open extension point: further numeric signals (orderbook imbalance,
    # CVD, basis, IV, Hurst, extra TA, on-chain flows...). Keys are short
    # snake_case names; prompts render them generically.
    extras: dict[str, float] = {}


class MarketSnapshot(BaseModel):
    as_of: datetime
    coins: list[CoinSnapshot]
    fear_greed: Optional[int] = None  # market-wide Fear & Greed index (0-100)
    # Market-wide numeric extras (btc_dominance_pct, dxy_change_24h_pct, ...).
    extras: dict[str, float] = {}
    # Recent headlines / event notes injected into prompts (RAG-lite).
    headlines: list[str] = []

    def coin(self, symbol: str) -> Optional[CoinSnapshot]:
        sym = symbol.upper()
        for c in self.coins:
            if c.symbol.upper() == sym:
                return c
        return None

    @property
    def symbols(self) -> list[str]:
        return [c.symbol for c in self.coins]


class Signal(BaseModel):
    """One agent's call on one coin for the round's horizon.

    Agents may optionally provide a full probability vector over outcomes
    (``p_long``/``p_short``/``p_flat``, summing to ~1). When present, the
    engine scores the signal with the strictly proper Brier rule instead of
    the legacy confidence×tanh rule, which makes honest probability reporting
    the optimal strategy (Gneiting & Raftery 2007).
    """

    agent: str
    symbol: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    price_at_signal: float
    p_long: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_short: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_flat: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @property
    def has_probs(self) -> bool:
        return None not in (self.p_long, self.p_short, self.p_flat)


class ScoredSignal(Signal):
    price_at_eval: float
    score: float  # in [-1, 1], see docs/INTERFACES.md


class AgentSpec(BaseModel):
    name: str
    provider: str  # "anthropic" | "openai_compat" | "mock"
    model: str = ""
    api_key_env: str = ""
    base_url: str = ""  # required for openai_compat
    seed: int = 0  # used by mock provider for determinism


class RoundStatus(str, Enum):
    PENDING = "PENDING"
    EVALUATED = "EVALUATED"


class ArenaSettings(BaseModel):
    top_n_coins: int = 20
    horizon_hours: float = 24.0
    eta: float = 0.35  # weight-update learning rate (used when adaptive_eta is off)
    adaptive_eta: bool = True  # eta(t) = sqrt(ln N / t), anytime-optimal Hedge
    min_weight: float = 0.05
    max_weight: float = 0.60
    flat_threshold_pct: float = 1.0  # |move| below this rewards FLAT calls
    fee_rate_bps: float = 5.0  # taker fee per side, basis points (paper PnL)
    db_path: str = "arena.db"
    start_equity: float = 10_000.0  # paper-trading starting equity per agent


class DebateSettings(BaseModel):
    enabled: bool = False  # bull/bear debate + critic pass before signals
    critic: bool = True
    self_consistency_below: float = 0.0  # re-sample when max prob < x (0 = off)


class MemorySettings(BaseModel):
    enabled: bool = True
    path: str = "memory"  # directory of per-agent JSON memory files
    short_rounds: int = 5
    pattern_rounds: int = 50


class CalibrationSettings(BaseModel):
    enabled: bool = True
    method: str = "platt"  # "platt" | "isotonic" | "none"
    refit_every: int = 20
    early_damping_rounds: int = 20  # blend toward uniform for first N rounds
    overconfidence_penalty: float = 0.9  # weight haircut on flagged agents
    drift_flag_gap: float = 0.15


class WeightsSettings(BaseModel):
    per_asset: bool = True  # maintain W[coin][agent] alongside global weights
    regime_conditional: bool = True  # separate bull/bear/chop weight books
    clip_factor_low: float = 0.5  # per-round multiplicative clip
    clip_factor_high: float = 2.0
    ess_floor: float = 1.5  # warn/raise floor when ESS collapses
    significance_rounds: int = 30  # damp updates 50/50 for first N rounds
    decorrelation: bool = True
    diversity_bonus: float = 0.05


class ConsensusSettings(BaseModel):
    extremize_lambda: float = 1.3  # 1.0 disables extremizing
    trim_fraction: float = 0.0  # trimmed-mean fraction (0 = off)
    herding_haircut: float = 0.85  # confidence multiplier when all agree
    no_trade_min_confidence: float = 0.0  # skip round below this (0 = off)
    mechanism: str = "vote"  # "vote" | "lmsr"
    lmsr_b: float = 0.1


class RiskSettings(BaseModel):
    kelly_fraction: float = 0.25  # fraction of full Kelly (0 = legacy equal stake)
    vol_target_annual_pct: float = 0.0  # 0 disables volatility targeting
    max_position_pct: float = 10.0  # per-coin cap, % of equity
    max_gross_leverage: float = 1.0
    stop_loss_pct: float = 0.0  # 0 disables
    take_profit_pct: float = 0.0
    daily_loss_limit_pct: float = 2.0  # 0 disables
    max_drawdown_halt_pct: float = 15.0  # 0 disables
    drawdown_scale_threshold_pct: float = 10.0
    min_volume_mcap_ratio: float = 0.01  # liquidity screen (0 = off)
    slippage_base_bps: float = 2.0  # 0 disables slippage model
    funding_in_pnl: bool = True
    market_neutral: bool = False  # rank-based long-short construction


class OpsSettings(BaseModel):
    json_logs: bool = False
    journal: bool = True  # persist snapshots + raw LLM replies + costs
    structured_outputs: bool = True  # provider-native JSON schema enforcement
    telegram_token_env: str = "TELEGRAM_BOT_TOKEN"
    telegram_chat_id_env: str = "TELEGRAM_CHAT_ID"
    discord_webhook_env: str = "DISCORD_WEBHOOK_URL"
    backup_keep: int = 14  # rolling DB backups (0 = off)


class ExchangeSettings(BaseModel):
    connector: str = "paper"  # "paper" | "binanceusdm" | "bybit"
    testnet: bool = True
    limit_timeout_s: float = 60.0
    twap_threshold_usd: float = 50_000.0
    twap_slices: int = 5


class ArenaConfig(BaseModel):
    arena: ArenaSettings = ArenaSettings()
    agents: list[AgentSpec] = []
    debate: DebateSettings = DebateSettings()
    memory: MemorySettings = MemorySettings()
    calibration: CalibrationSettings = CalibrationSettings()
    weights: WeightsSettings = WeightsSettings()
    consensus: ConsensusSettings = ConsensusSettings()
    risk: RiskSettings = RiskSettings()
    ops: OpsSettings = OpsSettings()
    exchange: ExchangeSettings = ExchangeSettings()
