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


class MarketSnapshot(BaseModel):
    as_of: datetime
    coins: list[CoinSnapshot]

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


class ArenaConfig(BaseModel):
    arena: ArenaSettings = ArenaSettings()
    agents: list[AgentSpec] = []
