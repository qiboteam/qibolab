from abc import ABC, abstractmethod
import yaml
from qibo.config import raise_error, log

class AbstractPlatform(ABC):
    """Abstract platform for controlling quantum devices.

    Args:
        name (str): name of the platform.
        runcard (str): path to the yaml file containing the platform setup.
    """

    def __init__(self, name, runcard):
        log.info(f"Loading platform {name}")
        log.info(f"Loading runcard {runcard}")
        self.name = name
        self.runcard = runcard
        # Load calibration settings
        with open(runcard, "r") as file:
            self._settings = yaml.safe_load(file)

        # Define references to instruments
        self.is_connected = False

    def _check_connected(self):
        if not self.is_connected:
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        with open(self.runcard, "r") as file:
            self._settings = yaml.safe_load(file)
        self.setup()

    @property
    def settings(self):
        return self._settings

    @property
    def data_folder(self):
        return self._settings.get("settings").get("data_folder")

    @property
    def hardware_avg(self):
        return self._settings.get("settings").get("hardware_avg")

    @property
    def sampling_rate(self):
        return self._settings.get("settings").get("sampling_rate")

    @property
    def software_averages(self):
        return self._settings.get("settings").get("software_averages")

    @software_averages.setter
    def software_averages(self, x):
        self._settings["settings"]["software_averages"] = x

    @property
    def repetition_duration(self):
        return self._settings.get("settings").get("repetition_duration")

    @property
    def resonator_frequency(self):
        return self._settings.get("settings").get("resonator_freq")

    @property
    def qubit_frequency(self):
        return self._settings.get("settings").get("qubit_freq")

    @property
    def pi_pulse_gain(self):
        return self._settings.get("settings").get("pi_pulse_gain")

    @property
    def pi_pulse_amplitude(self):
        return self._settings.get("settings").get("pi_pulse_amplitude")

    @property
    def pi_pulse_duration(self):
        return self._settings.get("settings").get("pi_pulse_duration")

    @property
    def pi_pulse_frequency(self):
        return self._settings.get("settings").get("pi_pulse_frequency")

    @property
    def readout_pulse(self):
        return self._settings.get("settings").get("readout_pulse")

    @property
    def max_readout_voltage(self):
        return self._settings.get("settings").get("resonator_spectroscopy_max_ro_voltage")

    @property
    def min_readout_voltage(self):
        return self._settings.get("settings").get("rabi_oscillations_pi_pulse_min_voltage")

    @property
    def delay_between_pulses(self):
        return self._settings.get("settings").get("delay_between_pulses")

    @property
    def delay_before_readout(self):
        return self._settings.get("settings").get("delay_before_readout")

    @abstractmethod
    def run_calibration(self):  # pragma: no cover
        """Executes calibration routines and updates the settings yml file"""
        raise_error(NotImplementedError)

    def __call__(self, sequence, nshots=None):
        return self.execute(sequence, nshots)

    @abstractmethod
    def connect(self):  # pragma: no cover
        """Connects to lab instruments using the details specified in the calibration settings."""
        raise_error(NotImplementedError)

    @abstractmethod
    def setup(self):  # pragma: no cover
        """Configures instruments using the loaded calibration settings."""
        raise_error(NotImplementedError)

    @abstractmethod
    def start(self):  # pragma: no cover
        """Turns on the local oscillators."""
        raise_error(NotImplementedError)

    @abstractmethod
    def stop(self):  # pragma: no cover
        """Turns off all the lab instruments."""
        raise_error(NotImplementedError)

    @abstractmethod
    def disconnect(self):  # pragma: no cover
        """Disconnects from the lab instruments."""
        raise_error(NotImplementedError)

    @abstractmethod
    def execute(self, sequence, nshots=None):  # pragma: no cover
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration json will be used.

        Returns:
            Readout results acquired by after execution.
        """
        raise_error(NotImplementedError)
