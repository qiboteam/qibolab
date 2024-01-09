from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

from qibolab.instruments.port import Port

InstrumentId = str


@dataclass
class InstrumentSettings:
    """Container of settings that are dumped in the platform runcard yaml."""

    def dump(self):
        """Dictionary containing the settings.

        Useful when dumping the instruments to the runcard YAML.
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

    PortType = Port
    """Class used by the instrument to instantiate ports."""

    def __init__(self, name, address):
        super().__init__(name, address)
        self._ports = {}

    @property
    @abstractmethod
    def sampling_rate(self):
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""

    def ports(self, port_name, *args, **kwargs):
        """Get ports associated to this controller.

        Args:
            port_name: Identifier for the port. The type of the identifier
                depends on the specialized port defined for each instrument.

        Returns:
            :class:`qibolab.instruments.port.Port` object providing the interface
            for setting instrument parameters.
        """
        if port_name not in self._ports:
            self._ports[port_name] = self.PortType(port_name)
        return self._ports[port_name]

    @abstractmethod
    def play(self, *args, **kwargs):
        """Play a pulse sequence and retrieve feedback.

        Returns:
            (Dict[ResultType]) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """

    def split_batches(self, sequences):  # pragma: no cover
        """Split sequences to batches each of which can be unrolled and played
        as a single sequence.

        Args:
            sequence (list): List of :class:`qibolab.pulses.PulseSequence` to be played.

        Returns:
            Iterator of batches that can be unrolled in a single one.
            Default will return all sequences as a single batch.
        """
        raise RuntimeError(
            f"Instrument of type {type(self)} does not support "
            "batch splitting for sequence unrolling."
        )

    @abstractmethod
    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters.

        Returns:
            (dict) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """


class InstrumentException(Exception):
    def __init__(self, instrument: Instrument, message: str):
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
