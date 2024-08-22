from abc import ABC, abstractmethod
from typing import Optional

import numpy.typing as npt
from pydantic import ConfigDict, Field

from qibolab.components import Config
from qibolab.components.channels import Channel
from qibolab.execution_parameters import ExecutionParameters
from qibolab.identifier import ChannelId
from qibolab.sequence import PulseSequence
from qibolab.serialize import Model
from qibolab.sweeper import ParallelSweepers

InstrumentId = str


class InstrumentSettings(Model):
    """Container of settings that are dumped in the platform runcard json."""

    model_config = ConfigDict(frozen=False)


class Instrument(Model, ABC):
    """Parent class for all the instruments connected via TCPIP.

    Args:
        name (str): Instrument name.
        address (str): Instrument network address.
    """

    model_config = ConfigDict(frozen=False)

    name: InstrumentId
    address: str
    is_connected: bool = False
    settings: Optional[InstrumentSettings] = None

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

    bounds: str
    """Estimated limitations of the device memory."""
    channels: dict[ChannelId, Channel] = Field(default_factory=dict)

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

        If :class:`qibolab.sweeper.Sweeper` objects are passed as arguments, they are
        executed in real-time. If not possible, an error is raised.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """


class InstrumentException(Exception):
    def __init__(self, instrument: Instrument, message: str):
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
