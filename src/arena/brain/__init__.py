"""Brain unit: debate brief, deterministic critic, memory, prompt optimizer."""

from arena.brain.debate import critic_review, debate_context
from arena.brain.memory import AgentMemory
from arena.brain.optimizer import VARIANTS, PromptVariantBook

__all__ = [
    "AgentMemory",
    "PromptVariantBook",
    "VARIANTS",
    "critic_review",
    "debate_context",
]
