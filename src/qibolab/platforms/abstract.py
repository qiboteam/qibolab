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
        # Load platform settings
        with open(runcard, "r") as file:
            self.settings = yaml.safe_load(file)
    
        self.is_connected = False

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
            log.info(f"Connecting to {self.name} instruments.")
            try:
                for name in self.instruments:
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                from qibo.config import raise_error
                raise_error(RuntimeError, "Cannot establish connection to "
                            f"{self.name} instruments. "
                            f"Error captured: '{exception}'")

    def setup(self):
        self.__dict__.update(self.settings['shared_settings'])
        setattr(self, 'topology', self.settings['topology'])
        setattr(self, 'qubit_channel_map', self.settings['qubit_channel_map'])
        setattr(self, 'channels', self.settings['channels'])

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
