from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from qm.qua._dsl import _Variable  # for type declaration only

from qibolab.pulses import PulseSequence

from .acquisition import Acquisition


@dataclass
class Parameters:
    duration: Optional[_Variable] = None
    amplitude: Optional[_Variable] = None
    phase: Optional[_Variable] = None


@dataclass
class ExecutionArguments:
    sequence: PulseSequence
    acquisitions: dict[tuple[str, str], Acquisition]
    relaxation_time: int = 0
    parameters: dict[str, Parameters] = field(
        default_factory=lambda: defaultdict(Parameters)
    )
