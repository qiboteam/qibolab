from qibo.config import log
from abc import ABC, abstractmethod
import yaml


class AbstractPlatform(ABC):
    """Abstract platform for controlling quantum devices.

    Args:
        name (str): name of the platform.
        runcard (str): path to the yaml file containing the platform setup.
    """
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False        
        # Load platform settings
        with open(runcard, "r") as file:
            self.settings = yaml.safe_load(file)
            
        self.instruments = {}
        self.instrument_settings = self.settings['instruments']

        for name in self.instrument_settings:
            lib = self.instrument_settings[name]['lib']
            i_class = self.instrument_settings[name]['class']
            ip = self.instrument_settings[name]['ip']
            from importlib import import_module
            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, ip)
            # instance.__dict__.update(self.settings['shared_settings'])
            self.instruments[name] = instance    

    def __getstate__(self):
        return {
            "name": self.name,
            "runcard": self.runcard,
            "settings": self.settings,
            "is_connected": self.is_connected
        }

    def __setstate__(self, data):
        self.name = data.get("name")
        self.runcard = data.get("runcard")
        self.settings = data.get("settings")
        self.is_connected = data.get("is_connected")

    def _check_connected(self):
        if not self.is_connected:
            from qibo.config import raise_error
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        with open(self.runcard, "r") as file:
            self.settings = yaml.safe_load(file)
        self.setup()

    @abstractmethod
    def run_calibration(self, show_plots=False):  # pragma: no cover
        """Executes calibration routines and updates the settings yml file"""
        raise NotImplementedError   

    def connect(self):
        if not self.is_connected:
            try:
                for name in self.instruments:
                    log.info(f"Connecting to {self.name} instruments {name}.")
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                from qibo.config import raise_error
                raise_error(RuntimeError, "Cannot establish connection to "
                            f"{self.name} instruments. "
                            f"Error captured: '{exception}'")

    def setup(self):
        self.__dict__.update(self.settings['shared_settings'])
        self.topology = self.settings['topology']
        self.channels = self.settings['channels']
        self.qubit_channel_map = self.settings['qubit_channel_map']
        
        # Generate qubit_instrument_map from qubit_channel_map and the instruments' channel_port_maps
        self.qubit_instrument_map = {}
        for qubit in self.qubit_channel_map:
            self.qubit_instrument_map[qubit] = [None, None, None]
            for name in self.instruments:
                if 'channel_port_map' in self.instrument_settings[name]['setup']:
                    for channel in self.instrument_settings[name]['setup']['channel_port_map']:
                        if channel in self.qubit_channel_map[qubit]:
                             self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name
        # Generate ro_channel[qubit], qd_channel[qubit], qf_channel[qubit], qrm[qubit], qcm[qubit], lo_qrm[qubit], lo_qcm[qubit]
        self.ro_channel = {}
        self.qd_channel = {}
        self.qf_channel = {}
        self.qrm = {}
        self.lo_qrm = {}
        self.qcm = {}
        self.lo_qcm = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][2]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit]  = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.lo_qrm[qubit] = self.instruments[self.instrument_settings[self.qubit_instrument_map[qubit][0]]['setup']['lo']]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qcm[qubit]  = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.lo_qcm[qubit] = self.instruments[self.instrument_settings[self.qubit_instrument_map[qubit][1]]['setup']['lo']]
            # TODO: implement qf modules


        # Load Native Gates
        self.native_gates = self.settings['native_gates']

        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].setup(**self.settings['shared_settings'], **self.instrument_settings[name]['setup'])

    def start(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].start()

    def stop(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].stop()

    def disconnect(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].disconnect()
            self.is_connected = False

    def __call__(self, sequence, nshots=None):
        return self.execute_pulse_sequence(sequence, nshots)

    @abstractmethod
    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration json will be used.

        Returns:
            Readout results acquired by after execution.
        """
        raise NotImplementedError
