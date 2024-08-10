from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

import numpy.typing as npt

from qibolab.components import Config
from qibolab.execution_parameters import ExecutionParameters
from qibolab.pulses.sequence import PulseSequence
from qibolab.sweeper import ParallelSweepers
from qibolab.unrolling import Bounds

InstrumentId = str


@dataclass
class InstrumentSettings:
    """Container of settings that are dumped in the platform runcard json."""

    def dump(self):
        """Dictionary containing the settings.

        Useful when dumping the instruments to the runcard JSON.
        """
        return asdict(self)


class Instrument(ABC):
    """Parent class for all the instruments connected via TCPIP.

    Args:
        name (str): Instrument name.
        address (str): Instrument network address.
    """

    def __init__(self, name, address):
        self.name: InstrumentId = name
        self.address: str = address
        self.is_connected: bool = False
        self.settings: Optional[InstrumentSettings] = None

    @property
    def signature(self):
        return f"{type(self).__name__}@{self.address}"

    @abstractmethod
    def connect(self):
        """Establish connection to the physical instrument."""

    @abstractmethod
    def disconnect(self):
        """Close connection to the physical instrument."""

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Set instrument settings."""


class Controller(Instrument):
    """Instrument that can play pulses (using waveform generator)."""

    def __init__(self, name, address):
        super().__init__(name, address)
        self.bounds: Bounds = Bounds(0, 0, 0)
        """Estimated limitations of the device memory."""

    def setup(self, bounds):
        """Set unrolling batch bounds."""
        self.bounds = Bounds(**bounds)

    def dump(self):
        """Dump unrolling batch bounds."""
        return {"bounds": asdict(self.bounds)}

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""

    @abstractmethod
    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, npt.NDArray]:
        """Play a pulse sequence and retrieve feedback.

        If :cls:`qibolab.sweeper.Sweeper` objects are passed as arguments, they are
        executed in real-time. If not possible, an error is raised.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """


class InstrumentException(Exception):
    def __init__(self, instrument: Instrument, message: str):
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
