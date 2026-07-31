"""Backward-compatible entry point for DDGR sem-id evolution.

The complete Section 3.6 implementation is in :mod:`train.evolution`.
"""
from train.evolution import (
    EvolutionConfig,
    EvolutionRoundStats,
    SemIDEvolutionEngine,
    resolve_duplicate_sem_ids,
    run_evolution,
)

__all__ = [
    "EvolutionConfig",
    "EvolutionRoundStats",
    "SemIDEvolutionEngine",
    "resolve_duplicate_sem_ids",
    "run_evolution",
]
