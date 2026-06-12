"""Arena engine: scoring, decision-power weights, and consensus signals."""

from arena.engine.consensus import consensus_signals
from arena.engine.scoring import realized_class, realized_return_pct, score_signal
from arena.engine.weights import adaptive_eta, equal_weights, update_weights

__all__ = [
    "adaptive_eta",
    "consensus_signals",
    "equal_weights",
    "realized_class",
    "realized_return_pct",
    "score_signal",
    "update_weights",
]
