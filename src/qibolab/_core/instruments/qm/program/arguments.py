from collections import defaultdict
from dataclasses import dataclass, field

from qm.qua._dsl import _Variable  # for type declaration only

from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Pulse
from qibolab._core.sequence import PulseSequence

from .acquisition import Acquisitions


@dataclass
class Parameters:
    """Container of QUA variables and other parameters needed for sweeping."""

    amplitude: _Variable | None = None
    amplitude_pulse: Pulse | None = None
    amplitude_op: str | None = None

    phase: _Variable | None = None

    duration: _Variable | None = None
    duration_ops: list[tuple[float, str]] = field(default_factory=list)
    interpolated: bool = False
    interpolated_op: str | None = None

    element: str | None = None
    lo_frequency: int | None = None
    max_offset: float = 0.5
    sampling_rate: int = 1


@dataclass
class ExecutionArguments:
    """Container of arguments required to generate the QUA program.

    These are collected in a single class because they are passed to all
    the different sweeper types.
    """

    sequence: PulseSequence
    acquisitions: Acquisitions
    relaxation_time: int = 0
    parameters: dict[str | ChannelId, Parameters] = field(
        default_factory=lambda: defaultdict(Parameters)
    )
