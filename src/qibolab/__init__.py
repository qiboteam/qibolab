import importlib.metadata as im
import importlib.util
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from dataclasses import dataclass
from enum import Enum, auto
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


class Profile:
    envvar = "QIBOLAB_PLATFORMS"
    filename = "platforms.toml"

    def __init__(self, path: Path):
        profile = tomllib.loads((path / self.filename).read_text(encoding="utf-8"))

        paths = {}
        for name, p in profile["paths"].items():
            paths[name] = path / Path(p)

        self.paths = paths


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

    profiles = Path(os.environ.get(Profile.envvar))
    if not os.path.exists(profiles):
        raise_error(RuntimeError, f"Profile file {profiles} does not exist.")

    platform = Profile(profiles).paths[name]

    spec = importlib.util.spec_from_file_location("platform", platform)
    if spec is None:
        raise_error(ModuleNotFoundError, f"Platform {platform} does not exist.")
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
