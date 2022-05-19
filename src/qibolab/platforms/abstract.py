from abc import ABC, abstractmethod
import yaml


class AbstractPlatform(ABC):
    """Abstract platform for controlling quantum devices.

    Args:
        name (str): name of the platform.
        runcard (str): path to the yaml file containing the platform setup.
    """
    def __init__(self, name, runcard):
        from qibo.config import log
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

    @abstractmethod
    def connect(self):  # pragma: no cover
        """Connects to lab instruments using the details specified in the calibration settings."""
        raise NotImplementedError

    @abstractmethod
    def setup(self):  # pragma: no cover
        """Configures instruments using the loaded calibration settings."""
        raise NotImplementedError

    @abstractmethod
    def start(self):  # pragma: no cover
        """Turns on the local oscillators."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):  # pragma: no cover
        """Turns off all the lab instruments."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):  # pragma: no cover
        """Disconnects from the lab instruments."""
        raise NotImplementedError


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
