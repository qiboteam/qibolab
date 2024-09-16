from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

from qm.qua._dsl import _Variable  # for type declaration only

from qibolab.identifier import ChannelId
from qibolab.pulses import Pulse
from qibolab.sequence import PulseSequence

from .acquisition import Acquisitions


@dataclass
class Parameters:
    """Container of QUA variables and other parameters needed for sweeping."""

    amplitude: Optional[_Variable] = None
    amplitude_pulse: Optional[Pulse] = None
    amplitude_op: Optional[str] = None

    phase: Optional[_Variable] = None

    duration: Optional[_Variable] = None
    duration_ops: list[tuple[float, str]] = field(default_factory=list)
    interpolated: bool = False

    element: Optional[str] = None
    lo_frequency: Optional[int] = None


@dataclass
class ExecutionArguments:
    """Container of arguments required to generate the QUA program.

    These are collected in a single class because they are passed to all
    the different sweeper types.
    """

    sequence: PulseSequence
    acquisitions: Acquisitions
    relaxation_time: int = 0
    parameters: dict[Union[str, ChannelId], Parameters] = field(
        default_factory=lambda: defaultdict(Parameters)
    )
