from enum import Enum, auto
from math import prod
from typing import Any, Optional, TypeVar

from pydantic import Field

from qibolab._core.sequence import PulseSequence

from .serialize import Model
from .sweeper import ParallelSweepers

__all__ = ["AcquisitionType", "AveragingMode"]

NS_TO_SEC = 1e-9


class AcquisitionType(Enum):
    """Data acquisition from hardware."""

    DISCRIMINATION = auto()
    """Demodulate, integrate the waveform and discriminate among states based
    on the voltages."""
    INTEGRATION = auto()
    """Demodulate and integrate the waveform."""
    RAW = auto()
    """Acquire the waveform as it is."""
    SPECTROSCOPY = auto()
    """Zurich Integration mode for RO frequency sweeps."""


class AveragingMode(Enum):
    """Data averaging modes from hardware."""

    CYCLIC = auto()
    """Better averaging for short timescale noise."""
    SINGLESHOT = auto()
    """SINGLESHOT: No averaging."""
    SEQUENTIAL = auto()
    """SEQUENTIAL: Worse averaging for noise[Avoid]"""

    @property
    def average(self) -> bool:
        """Whether an average is performed or not."""
        return self is not AveragingMode.SINGLESHOT


Update = dict[str, Any]

ConfigUpdate = dict[str, Update]
"""Update for component configs.

Maps component name to corresponding update, which in turn is a map from
config property name that needs an update to its new value.
"""

# TODO: replace with https://docs.python.org/3/reference/compound_stmts.html#type-params
T = TypeVar("T")


# TODO: lift for general usage in Qibolab
def default(value: Optional[T], default: T) -> T:
    """None replacement shortcut."""
    return value if value is not None else default


class ExecutionParameters(Model):
    """Data structure to deal with execution parameters."""

    nshots: Optional[int] = None
    """Number of shots to sample from the experiment.

    Default is the runcard value.
    """
    relaxation_time: Optional[int] = None
    """Time to wait for the qubit to relax to its ground Sample between shots
    in ns.

    Default is the runcard value.
    """
    fast_reset: bool = False
    """Enable or disable fast reset."""
    acquisition_type: AcquisitionType = AcquisitionType.DISCRIMINATION
    """Data acquisition type."""
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT
    """Data averaging mode."""
    updates: list[ConfigUpdate] = Field(default_factory=list)
    """List of updates for component configs.

    Later entries in the list take precedence over earlier ones (if they
    happen to update the same thing). These updates will be applied on
    top of platform defaults.
    """

    def bins(self, sweepers: list[ParallelSweepers]) -> tuple[int, ...]:
        assert self.nshots is not None
        shots = (
            (self.nshots,) if self.averaging_mode is AveragingMode.SINGLESHOT else ()
        )
        sweeps = tuple(
            min(len(sweep.values) for sweep in parsweeps) for parsweeps in sweepers
        )
        return shots + sweeps

    def results_shape(
        self, sweepers: list[ParallelSweepers], samples: Optional[int] = None
    ) -> tuple[int, ...]:
        """Compute the expected shape for collected data."""

        inner = {
            AcquisitionType.DISCRIMINATION: (),
            AcquisitionType.INTEGRATION: (2,),
            AcquisitionType.RAW: (samples, 2),
        }[self.acquisition_type]
        return self.bins(sweepers) + inner

    def estimate_duration(
        self,
        sequences: list[PulseSequence],
        sweepers: list[ParallelSweepers],
    ) -> float:
        """Estimate experiment duration."""
        duration = sum(seq.duration for seq in sequences)
        relaxation = default(self.relaxation_time, 0)
        nshots = default(self.nshots, 0)
        return (
            (duration + len(sequences) * relaxation)
            * nshots
            * NS_TO_SEC
            * prod(len(s[0]) for s in sweepers)
        )
