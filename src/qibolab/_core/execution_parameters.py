from enum import Enum, auto
from typing import Any, Optional

from .serialize import Model
from .sweeper import ParallelSweepers

__all__ = ["AcquisitionType", "AveragingMode", "ExecutionParameters"]


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


ConfigUpdate = dict[str, dict[str, Any]]
"""Update for component configs.

Maps component name to corresponding update, which in turn is a map from
config property name that needs an update to its new value.
"""


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
    updates: list[ConfigUpdate] = []
    """List of updates for component configs.

    Later entries in the list take precedence over earlier ones (if they
    happen to update the same thing). These updates will be applied on
    top of platform defaults.
    """

    def results_shape(
        self, sweepers: list[ParallelSweepers], samples: Optional[int] = None
    ) -> tuple[int, ...]:
        """Compute the expected shape for collected data."""

        shots = (
            (self.nshots,) if self.averaging_mode is AveragingMode.SINGLESHOT else ()
        )
        sweeps = tuple(
            min(len(sweep.values) for sweep in parsweeps) for parsweeps in sweepers
        )
        inner = {
            AcquisitionType.DISCRIMINATION: (),
            AcquisitionType.INTEGRATION: (2,),
            AcquisitionType.RAW: (samples, 2),
        }[self.acquisition_type]
        return shots + sweeps + inner
