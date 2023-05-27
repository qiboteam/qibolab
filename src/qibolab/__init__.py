import importlib.metadata as im
import importlib.util
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from qibo.config import raise_error

from qibolab.platform import Platform
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedRawWaveformResults,
    AveragedSampleResults,
    IntegratedResults,
    RawWaveformResults,
    SampleResults,
)

__version__ = im.version(__package__)

PLATFORMS = "QIBOLAB_PLATFORMS"


def get_platforms_path():
    """Get path to repository containing the platforms.

    Path is specified using the environment variable QIBOLAB_PLATFORMS.
    """
    profiles = os.environ.get(PLATFORMS)
    if profiles is None or not os.path.exists(profiles):
        raise_error(RuntimeError, f"Profile directory {profiles} does not exist.")
    return Path(profiles)


def create_platform(name, runcard=None):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
    Returns:
        The plaform class.
    """
    if name == "dummy":
        from qibolab.paths import qibolab_folder
        from qibolab.platform import create_dummy

        return create_dummy(qibolab_folder / "runcards" / "dummy.yml")

    platform = get_platforms_path() / f"{name}.py"
    if not platform.exists():
        raise_error(ValueError, f"Platform {name} does not exist.")

    spec = importlib.util.spec_from_file_location("platform", platform)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if runcard is None:
        return module.create()
    return module.create(runcard)


class AcquisitionType(Enum):
    """Data acquisition from hardware"""

    DISCRIMINATION = auto()
    """Demodulate, integrate the waveform and discriminate among states based on the voltages"""
    INTEGRATION = auto()
    """Demodulate and integrate the waveform"""
    RAW = auto()
    """Acquire the waveform as it is"""
    SPECTROSCOPY = auto()
    """Zurich Integration mode for RO frequency sweeps"""


class AveragingMode(Enum):
    """Data averaging modes from hardware"""

    CYCLIC = auto()
    """Better averaging for short timescale noise"""
    SINGLESHOT = auto()
    """SINGLESHOT: No averaging"""
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


@dataclass(frozen=True)
class ExecutionParameters:
    """Data structure to deal with execution parameters"""

    nshots: Optional[int] = None
    """Number of shots to sample from the experiment. Default is the runcard value."""
    relaxation_time: Optional[int] = None
    """Time to wait for the qubit to relax to its ground Sample between shots in ns. Default is the runcard value."""
    fast_reset: bool = False
    """Enable or disable fast reset"""
    acquisition_type: AcquisitionType = AcquisitionType.DISCRIMINATION
    """Data acquisition type"""
    averaging_mode: AveragingMode = AveragingMode.SINGLESHOT
    """Data averaging mode"""

    @property
    def results_type(self):
        """Returns corresponding results class"""
        return RESULTS_TYPE[self.averaging_mode][self.acquisition_type]
