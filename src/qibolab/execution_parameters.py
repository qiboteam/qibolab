from enum import Enum, auto
from typing import Optional

from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedSampleResults,
    IntegratedResults,
    RawWaveformResults,
    SampleResults,
)
from qibolab.serialize_ import Model


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


RESULTS_TYPE = {
    AveragingMode.CYCLIC: {
        AcquisitionType.INTEGRATION: AveragedIntegratedResults,
        AcquisitionType.RAW: AveragedRawWaveformResults,
        AcquisitionType.DISCRIMINATION: AveragedSampleResults,
    },
    AveragingMode.SINGLESHOT: {
        AcquisitionType.INTEGRATION: IntegratedResults,
        AcquisitionType.RAW: RawWaveformResults,
        AcquisitionType.DISCRIMINATION: SampleResults,
    },
}


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

    @property
    def results_type(self):
        """Returns corresponding results class."""
        return RESULTS_TYPE[self.averaging_mode][self.acquisition_type]
