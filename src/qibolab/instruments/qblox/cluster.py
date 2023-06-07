""" Qblox instruments driver.

Supports the following Instruments:
    Cluster
    Cluster QRM-RF
    Cluster QCM-RF
    Cluster QCM
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

from qblox_instruments.qcodes_drivers.cluster import Cluster as QbloxCluster
from qibo.config import log

from qibolab.instruments.abstract import Instrument, InstrumentException


class Cluster(Instrument):
    """A class to extend the functionality of qblox_instruments Cluster.

    The class exposes the attribute `reference_clock_source` to enable the selection of an internal or external clock
    source.

    The class inherits from Instrument and implements its interface methods:
        __init__()
        connect()
        setup()
        start()
        stop()
        disconnect()

    Attributes:
        device (QbloxCluster): A reference to the underlying `qblox_instruments.qcodes_drivers.cluster.Cluster` object.
            It can be used to access other features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/cluster.html
        reference_clock_source (str): ('internal', 'external') Instructs the cluster to use the internal clock source
            or an external source.
    """

    def __init__(self, name: str, address: str, settings: dict):
        """Initialises the instrument storing its name and address."""
        super().__init__(name, address)
        self.settings: dict = settings
        self.reference_clock_source: str
        self.device: QbloxCluster = None

        self._device_parameters = {}

    @property
    def reference_clock_source(self) -> str:
        if self.is_connected:
            return self.device.get("reference_source")
        else:
            raise Exception(
                f"Parameter reference_source cannot be accessed, there is no connection to cluster {self.name}."
            )

    @reference_clock_source.setter
    def reference_clock_source(self, value: str):
        if self.is_connected:
            self._set_device_parameter(self.device, "reference_source", value=value)
        else:
            raise Exception(
                f"Parameter reference_source cannot be set up, there is no connection to cluster {self.name}."
            )

    def connect(self):
        """Connects to the cluster.

        If the connection is successful, it saves a reference to the underlying object in the attribute `device`.
        The instrument is reset on each connection.
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
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
                # TODO: if not able to connect after 3 attempts, check for ping response and reboot
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")
        else:
            pass

    def _set_device_parameter(self, target, *parameters, value):
        """Sets a parameter of the instrument if its value changed from the last value stored in the cache.

        Args:
            target = an instance of qblox_instruments.qcodes_drivers.qcm_qrm.QcmQrm or
                                    qblox_instruments.qcodes_drivers.sequencer.Sequencer
            *parameters (list): A list of parameters to be cached and set.
            value = The value to set the paramters.
        Raises:
            Exception = If attempting to set a parameter without a connection to the instrument.
        """
        if self.is_connected:
            key = target.name + "." + parameters[0]
            if not key in self._device_parameters:
                for parameter in parameters:
                    if not hasattr(target, parameter):
                        raise Exception(f"The instrument {self.name} does not have parameters {parameter}")
                    target.set(parameter, value)
                self._device_parameters[key] = value
            elif self._device_parameters[key] != value:
                for parameter in parameters:
                    target.set(parameter, value)
                self._device_parameters[key] = value
        else:
            raise Exception("There is no connection to the instrument {self.name}")

    def _erase_device_parameters_cache(self):
        """Erases the cache of instrument parameters."""
        self._device_parameters = {}

    def setup(self):
        """Configures the instrument with the settings saved in the runcard.

        Args:
            **kwargs: A dictionary with the settings:
                reference_clock_source
        """
        settings = self.settings
        if self.is_connected:
            self.reference_clock_source = settings["reference_clock_source"]
        else:
            raise Exception("The instrument cannot be set up, there is no connection")

    def start(self):
        """Empty method to comply with Instrument interface."""
        pass

    def stop(self):
        """Empty method to comply with Instrument interface."""
        pass

    def disconnect(self):
        """Closes the connection to the instrument."""
        self.is_connected = False
        self.device.close()
        # TODO: set modules is_connected to False
