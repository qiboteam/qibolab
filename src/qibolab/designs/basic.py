from dataclasses import dataclass, field
from typing import Any, List, Optional

from qibolab.instruments.abstract import AbstractInstrument


@dataclass
class Channel:
    """Representation of physical wire connection (channel).

    Name is used as a unique identifier for channels.
    Channel objects are instantiated by :class:`qibolab.platforms.platform.Platform`,
    but their attributes are modified and used by instrument designs.

    Args:
        name (str): Name of the channel as given in the platform runcard.

    Attributes:
        ports (list): List of tuples (controller (`str`), port (`int`))
            specifying the QM (I, Q) ports that the channel is connected.
        qubits (list): List of tuples (:class:`qibolab.platforms.utils.Qubit`, str)
            for the qubit connected to this channel and the role of the channel.
        Optional arguments holding local oscillators and related parameters.
        These are relevant only for mixer-based insturment designs.
    """

    name: str

    qubit: Optional["Qubit"] = field(default=None, init=False, repr=False)
    ports: List[tuple] = field(default_factory=list, init=False)
    local_oscillator: Any = field(default=None, init=False)
    lo_frequency: float = field(default=0, init=False)
    lo_power: float = field(default=0, init=False)
    _offset: Optional[float] = field(default=None, init=False)
    _filter: Optional[dict] = field(default=None, init=False)

    @property
    def offset(self):
        if self._offset is None:
            # operate qubits at their sweetspot unless otherwise stated
            self._offset = self.qubit.sweetspot
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def filter(self):
        if self._filter is None:
            self._filter = self.qubit.filter
        return self._filter

    @filter.setter
    def filter(self, filter):
        self._filter = filter


@dataclass
class BasicInstrumentDesign:
    """Instrument design that uses a single controller.

    Attributes:
        controller (:class:`qibolab.instruments.abstract.AbstractInstrument`): Instrument used for sending pulses and retrieving feedback.
        is_connected (bool): Boolean that shows whether instruments are connected.
    """

    controller: AbstractInstrument
    channels: dict = field(default_factory=dict)
    is_connected: bool = field(default=False, init=False)

    def connect(self):
        """Connect to all instruments."""
        if not self.is_connected:
            self.controller.connect()
        self.is_connected = True

    def setup(self, qubits, *args, **kwargs):
        """Load settings to instruments."""
        self.controller.setup(qubits, *args, **kwargs)

    def start(self):
        """Start all instruments."""
        self.controller.start()

    def stop(self):
        """Stop all instruments."""
        self.controller.stop()

    def disconnect(self):
        """Disconnect all instruments."""
        self.controller.disconnect()
        self.is_connected = False

    def play(self, *args, **kwargs):
        """Play a pulse sequence and retrieve feedback."""
        return self.controller.play(*args, **kwargs)

    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters."""
        return self.controller.sweep(*args, **kwargs)
