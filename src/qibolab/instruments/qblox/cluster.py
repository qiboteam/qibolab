"""Qblox instruments driver.

Supports the following Instruments:

-Cluster
-Cluster QRM-RF
-Cluster QCM-RF
-Cluster QCM

Compatible with qblox-instruments driver 0.9.0 (28/2/2023).
It supports:

- multiplexed readout of up to 6 qubits
- hardware modulation, demodulation, and classification
- software modulation, with support for arbitrary pulses
- software demodulation
- binned acquisition
- real-time sweepers of
    - pulse frequency (requires hardware modulation)
    - pulse relative phase (requires hardware modulation)
    - pulse amplitude
    - pulse start
    - pulse duration
    - port gain
    - port offset

- multiple readouts for the same qubit (sequence unrolling)
- max iq pulse length 8_192ns
- waveforms cache, uses additional free sequencers if the memory of one sequencer (16384) is exhausted
- instrument parameters cache
- safe disconnection of offsets on termination

The operation of multiple clusters simultaneously is not supported yet.
https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/
"""

from dataclasses import dataclass
from enum import Enum

from qblox_instruments.qcodes_drivers.cluster import Cluster as QbloxCluster
from qibo.config import log

from qibolab.instruments.abstract import Instrument, InstrumentException


class ReferenceClockSource(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class Cluster_Settings:
    """Settings of the Cluster instrument."""

    reference_clock_source: ReferenceClockSource = ReferenceClockSource.INTERNAL
    """Instruct the cluster to use the internal clock source or an external
    source."""


class Cluster(Instrument):
    """A class to extend the functionality of qblox_instruments Cluster.

    The class exposes the attribute `reference_clock_source` to enable
    the selection of an internal or external clock source.

    The class inherits from

    :class: `qibolab.instruments.abstract.Instrument` and implements its
        interface methods: __init__() connect() setup() start() stop()
        disconnect()
    """

    def __init__(
        self, name: str, address: str, settings: Cluster_Settings = Cluster_Settings()
    ):
        """Initialises the instrument storing its name, address and
        settings."""
        super().__init__(name, address)
        self.device: QbloxCluster = None
        """Reference to the underlying
        `qblox_instruments.qcodes_drivers.cluster.Cluster` object."""
        self.settings: Cluster_Settings = settings
        """Instrument settings."""

    @property
    def reference_clock_source(self) -> ReferenceClockSource:
        if self.is_connected:
            _reference_clock_source = self.device.get("reference_source")
            if _reference_clock_source == "internal":
                self.settings.reference_clock_source = ReferenceClockSource.INTERNAL
            elif _reference_clock_source == "external":
                self.settings.reference_clock_source = ReferenceClockSource.EXTERNAL
        return self.settings.reference_clock_source

    @reference_clock_source.setter
    def reference_clock_source(self, value: ReferenceClockSource):
        self.settings.reference_clock_source = value
        if self.is_connected:
            self.device.set(
                "reference_source", self.settings.reference_clock_source.value
            )

    def connect(self):
        """Connects to the cluster.

        If the connection is successful, it resets the instrument and
        configures it with the stored settings. A reference to the
        underlying object is saved in the attribute `device`.
        """
        if not self.is_connected:
            for attempt in range(3):
                try:
                    QbloxCluster.close_all()
                    self.device = QbloxCluster(self.name, self.address)
                    self.device.reset()
                    self.is_connected = True
                    break
                except Exception as exc:
                    log.info(f"Unable to connect:\n{str(exc)}\nRetrying...")
                # TODO: if not able to connect after 3 attempts, check for ping response and reboot
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")

        # apply stored settings
        # self._setup()

    # def _setup(self):
    #     if self.is_connected:
    # self.device.set("reference_source", self.settings.reference_clock_source.value)

    def setup(self):
        """Configures the instrument with the stored settings."""
        # self._setup()

    def start(self):
        """Empty method to comply with Instrument interface."""

    def stop(self):
        """Empty method to comply with Instrument interface."""

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
        self.device.close()
        # TODO: set modules is_connected to False
