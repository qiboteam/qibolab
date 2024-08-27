from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from qm.qua._dsl import _Variable  # for type declaration only

from qibolab.sequence import PulseSequence

from .acquisition import Acquisitions


@dataclass
class Parameters:
    """Container of swept QUA variables."""

    duration: Optional[_Variable] = None
    amplitude: Optional[_Variable] = None
    phase: Optional[_Variable] = None
    pulses: list[tuple[float, str]] = field(default_factory=list)
    interpolated: bool = False


@dataclass
class ExecutionArguments:
    """Container of arguments required to generate the QUA program.

    These are collected in a single class because they are passed to all
    the different sweeper types.
    """

    sequence: PulseSequence
    acquisitions: Acquisitions
    relaxation_time: int = 0
    parameters: dict[str, Parameters] = field(
        default_factory=lambda: defaultdict(Parameters)
    )
