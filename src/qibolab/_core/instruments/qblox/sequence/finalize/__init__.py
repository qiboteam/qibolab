from . import move, register, wait
from .components import Pipeline

__all__ = [
    "DEFAULT_PIPELINE",
    "Pipeline",
    "move",
    "register",
    "wait",
]

DEFAULT_PIPELINE = (
    wait.merge_wait,
    (wait.long_wait, move.negative_immediate),
    register.update_nop,
)
"""Pipeline used by default to finalize the Q1ASM code compiled by this driver."""
