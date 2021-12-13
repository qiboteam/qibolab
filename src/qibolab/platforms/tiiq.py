import os
import json
from qibo.config import raise_error, log


class TIIq:
    """Platform for controlling TII device.

    Controls the PulsarQRM, PulsarQCM and two SGS100A local oscillators.
    Uses calibration parameters provided through a json to setup the instruments.
    The path of the calibration json can be provided using the
    ``"CALIBRATION_PATH"`` environment variable.
    If no path is provided the default path (``platforms/tiiq_settings.json``)
    will be used.
    """

    def __init__(self):
        # TODO: Consider passing ``calibration_path` as environment variable
        self.calibration_path = os.environ.get("CALIBRATION_PATH")
        if self.calibration_path is None:
            # use default file
            import pathlib
            self.calibration_path = pathlib.Path(__file__).parent / "tiiq_settings.json"
        # Load calibration settings
        with open(self.calibration_path, "r") as file:
            self._settings = json.load(file)

        # initialize instruments
        self._connected = False
        self.qrm = None
        self.qcm = None
        self.LO_qrm = None
        self.LO_qcm = None
        try:
            self.connect()
        except: # capture time-out errors when importing outside the lab (bad practice)
            log.warning("Cannot establish connection to TIIq instruments. Skipping...")
        # initial instrument setup
        self.setup()

    @property
    def data_folder(self):
        return self._settings.get("_settings").get("data_folder")

    @property
    def hardware_avg(self):
        return self._settings.get("_settings").get("hardware_avg")

    @property
    def software_averages(self):
        return self._settings.get("_settings").get("software_averages")

    @software_averages.setter
    def software_averages(self, x):
        # I don't like that this updates the local dictionary but not the json
        self._settings["_settings"]["software_averages"] = x

    @property
    def hardware_avg(self):
        return self._settings.get("_settings").get("hardware_avg")

    @property
    def repetition_duration(self):
        return self._settings.get("_settings").get("repetition_duration")

    def run_calibration(self):
        """Executes calibration routines and updates the settings json."""
        # TODO: Implement calibration routines and update ``self._settings``.

        # update instruments with new calibration settings
        self.setup()
        # save new calibration settings to json
        with open(self.calibration_path, "w") as file:
            json.dump(self._settings, file)

    def connect(self):
        """Connects to lab instruments using the details specified in the loaded settings.

        Two QBlox (:class:`qibolab.instruments.qblox.PulsarQRM` and
        :class:`qibolab.instruments.qblox.PulsarQCM`) and two local oscillators
        (:class:`qibolab.instruments.rohde_schwarz.SGS100A`) are used in the
        TIIq configuration.
        """
        from qibolab.instruments import PulsarQRM, PulsarQCM, SGS100A
        self.qrm = PulsarQRM(**self._settings.get("_QRM_init_settings"))
        self.qcm = PulsarQCM(**self._settings.get("_QCM_init_settings"))
        self.LO_qrm = SGS100A(**self._settings.get("_LO_QRM_init_settings"))
        self.LO_qcm = SGS100A(**self._settings.get("_LO_QCM_init_settings"))
        self._connected = True

    def setup(self):
        """Configures instruments using the loaded calibration settings."""
        if self._connected:
            self.qrm.setup(**self._settings.get("_QRM_settings"))
            self.qcm.setup(**self._settings.get("_QCM_settings"))
            self.LO_qrm.setup(**self._settings.get("_LO_QRM_settings"))
            self.LO_qcm.setup(**self._settings.get("_LO_QCM_settings"))

    def start(self):
        """Turns on the local oscillators.

        The QBlox insturments are turned on automatically during execution after
        the required pulse sequences are loaded.
        """
        if self._connected:
            self.LO_qcm.on()
            self.LO_qrm.on()

    def stop(self):
        """Turns off all the lab instruments."""
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()

    def disconnect(self):
        """Disconnects from the lab instruments."""
        if self._connected:
            self.LO_qrm.close()
            self.LO_qcm.close()
            self.qrm.close()
            self.qcm.close()
            self._connected = False

    def execute(self, sequence):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.

        Returns:
            Readout results acquired by :class:`qibolab.instruments.qblox.PulsarQRM`
            after execution.
        """
        if not self._connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")

        # Translate and upload instructions to instruments
        if sequence.qcm_pulses:
            waveforms, program = self.qcm.translate(sequence)
            self.qcm.upload(waveforms, program, self.data_folder)
        if sequence.qrm_pulses:
            waveforms, program = self.qrm.translate(sequence)
            self.qrm.upload(waveforms, program, self.data_folder)

        # Execute instructions
        if sequence.qcm_pulses:
            self.qcm.play_sequence()
        if sequence.qrm_pulses:
            # TODO: Find a better way to pass the readout pulse here
            acquisition_results = self.qrm.play_sequence_and_acquire(sequence.qrm_pulses[0])
        else:
            acquisition_results = None

        return acquisition_results

    def __call__(self, sequence):
        return self.execute(sequence)
