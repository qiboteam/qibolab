import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from qibolab.instruments.port import Port

InstrumentId = str
INSTRUMENTS_DATA_FOLDER = Path.home() / ".qibolab" / "instruments" / "data"


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
        self.signature: str = f"{type(self).__name__}@{address}"
        self.settings: Optional[InstrumentSettings] = None
        # create local storage folder
        instruments_data_folder = INSTRUMENTS_DATA_FOLDER
        instruments_data_folder.mkdir(parents=True, exist_ok=True)
        # create temporary directory
        self.tmp_folder = tempfile.TemporaryDirectory(dir=instruments_data_folder)
        self.data_folder = Path(self.tmp_folder.name)

    @abstractmethod
    def connect(self):
        """Establish connection to the physical instrument."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Upload settings to the physical instrument."""
        raise NotImplementedError

    @abstractmethod
    def start(self):
        """Turn on the physical instrument."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Turn off the physical instrument."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        """Close connection to the physical instrument."""
        raise NotImplementedError


class Controller(Instrument):
    """Instrument that can play pulses (using waveform generator)."""

    PortType = Port
    """Class used by the instrument to instantiate ports."""

    def __init__(self, name, address):
        super().__init__(name, address)
        self._ports = {}

    def __getitem__(self, port_name):
        return self.ports(port_name)

    def ports(self, port_name):
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

    @abstractmethod
    def play_sequences(self, *args, **kwargs):
        """Play pulses sequences by unrolling and retrieve feedback.

        Returns:
            (Dict[List[ResultType]) mapping the serial of the readout pulses used to a list of
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """

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
