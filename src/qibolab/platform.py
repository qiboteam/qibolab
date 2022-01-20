import os
import pathlib
import yaml
from qibo.config import raise_error, log


class Platform:
    """Platform for controlling quantum devices.

    The path of the calibration json can be provided using the
    ``"CALIBRATION_PATH"`` environment variable.
    If no path is provided the default path (``platforms/tiiq_settings.json``)
    will be used.

    Args:
        name (str): name of the platform stored in a yaml file in qibolab/platforms.
    """

    def __init__(self, name):
        log.info(f"Loading platform {name}.")
        self.name = name
        # Load calibration settings
        self.calibration_path = pathlib.Path(__file__).parent / "platforms" / f"{name}.yml"
        with open(self.calibration_path, "r") as file:
            self._settings = yaml.safe_load(file)

        # Define references to instruments
        self.is_connected = False
        self._qrm = None
        self._qcm = None
        self._LO_qrm = None
        self._LO_qcm = None
        # instruments are connected in :meth:`qibolab.platform.Platform.start`

    def _check_connected(self):
        if not self.is_connected:
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    @property
    def qrm(self):
        """Reference to :class:`qibolab.instruments.qblox.PulsarQRM` instrument."""
        self._check_connected()
        return self._qrm

    @property
    def qcm(self):
        """Reference to :class:`qibolab.instruments.qblox.PulsarQCM` instrument."""
        self._check_connected()
        return self._qcm

    @property
    def LO_qrm(self):
        """Reference to QRM local oscillator (:class:`qibolab.instruments.rohde_schwarz.SGS100A`)."""
        self._check_connected()
        return self._LO_qrm

    @property
    def LO_qcm(self):
        """Reference to QCM local oscillator (:class:`qibolab.instruments.rohde_schwarz.SGS100A`)."""
        self._check_connected()
        return self._LO_qcm

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
        # I don't like that this updates the local dictionary but not the json
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

    def run_calibration(self):
        """Executes calibration routines and updates the settings json."""
        # TODO: Implement calibration routines and update ``self._settings``.

        # update instruments with new calibration settings
        self.setup()
        # save new calibration settings to json
        with open(self.calibration_path, "w") as file:
            yaml.dump(self._settings, file)

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            log.info(f"Connecting to {self.name} instruments.")
            try:
                from qibolab.instruments import PulsarQRM, PulsarQCM, SGS100A
                self._qrm = PulsarQRM(**self._settings.get("QRM_init_settings"))
                self._qcm = PulsarQCM(**self._settings.get("QCM_init_settings"))
                self._LO_qrm = SGS100A(**self._settings.get("LO_QRM_init_settings"))
                self._LO_qcm = SGS100A(**self._settings.get("LO_QCM_init_settings"))
                self.is_connected = True
            except Exception as exception:
                raise_error(RuntimeError, "Cannot establish connection to "
                                         f"{self.name} instruments. "
                                         f"Error captured: '{exception}'")
        self.setup()

    def setup(self):
        """Configures instruments using the loaded calibration settings."""
        if self.is_connected:
            self._qrm.setup(**self._settings.get("QRM_settings"))
            self._qcm.setup(**self._settings.get("QCM_settings"))
            self._LO_qrm.setup(**self._settings.get("LO_QRM_settings"))
            self._LO_qcm.setup(**self._settings.get("LO_QCM_settings"))

    def start(self):
        """Turns on the local oscillators.

        The QBlox insturments are turned on automatically during execution after
        the required pulse sequences are loaded.
        """
        self.connect()
        self._LO_qcm.on()
        self._LO_qrm.on()

    def stop(self):
        """Turns off all the lab instruments."""
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()

    def disconnect(self):
        """Disconnects from the lab instruments."""
        if self.is_connected:
            self._LO_qrm.close()
            self._LO_qcm.close()
            self._qrm.close()
            self._qcm.close()
            self.is_connected = False

    def execute(self, sequence, nshots=None):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration json will be used.

        Returns:
            Readout results acquired by :class:`qibolab.instruments.qblox.PulsarQRM`
            after execution.
        """
        if not self.is_connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg

        # Translate and upload instructions to instruments
        if sequence.qcm_pulses:
            waveforms, program = self._qcm.translate(sequence, nshots)
            self._qcm.upload(waveforms, program, self.data_folder)
        if sequence.qrm_pulses:
            waveforms, program = self._qrm.translate(sequence, nshots)
            self._qrm.upload(waveforms, program, self.data_folder)

        # Execute instructions
        if sequence.qcm_pulses:
            self._qcm.play_sequence()
        if sequence.qrm_pulses:
            # TODO: Find a better way to pass the readout pulse here
            acquisition_results = self._qrm.play_sequence_and_acquire(sequence.qrm_pulses[0])
        else:
            acquisition_results = None

        return acquisition_results

    def __call__(self, sequence, nshots=None):
        return self.execute(sequence, nshots)
