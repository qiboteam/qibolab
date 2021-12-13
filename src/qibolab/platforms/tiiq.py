import json


class TIIq:
    """Platform for controlling TII device.

    Controls the PulsarQRM, PulsarQCM and two SGS100A local oscillators.

    Args:
        settings (dict): Dictionary with calibration data.
            See ``examples/tii_single_qubit/tii_single_qubit_settings.json``
            for an example for this dictionary.
    """

    def __init__(self):
        # load latest calibration settings from json
        settings = self.load_settings()

        # initialize instruments
        self._connected = False
        self.qrm = None
        self.qcm = None
        self.LO_qrm = None
        self.LO_qcm = None
        try:
            self.connect()
        except: # capture time-out errors when importing outside the lab (bad practice)
            from qibo.config import log
            log.warning("Cannot establish connection to TIIq instruments. Skipping...")

        # initial platform setup
        self.data_folder = None
        self.hardware_avg = None
        self.sampling_rate = None
        self.software_averages = None
        self.repetition_duration = None
        self.setup_platform(settings.get("_settings"))
        # initial instrument setup
        self.setup(settings)

    def load_settings(self, filedir=None):
        """Loads json with calibration settings.

        Args:
            filedir (str): Path to the json file to load.
                If ``None`` the default settings file located in ``platforms``
                will be used.

        Returns:
            The ``settings`` dictionary required for instrument setup.
        """
        if filedir is None:
            # use default file
            import pathlib
            filedir = pathlib.Path(__file__).parent / "tiiq_settings.json"

        with open(filedir, "r") as file:
            settings = json.load(file)
        return settings

    def connect(self):
        """Connects to lab instruments using the details specified in the loaded settings.

        Two QBlox (:class:`qibolab.instruments.qblox.PulsarQRM` and
        :class:`qibolab.instruments.qblox.PulsarQCM`) and two local oscillators
        (:class:`qibolab.instruments.rohde_schwarz.SGS100A`) are used in the
        TIIq configuration.
        """
        from qibolab.instruments import PulsarQRM, PulsarQCM, SGS100A
        self.qrm = PulsarQRM(**settings.get("_QRM_init_settings"))
        self.qcm = PulsarQCM(**settings.get("_QCM_init_settings"))
        self.LO_qrm = SGS100A(**settings.get("_LO_QRM_init_settings"))
        self.LO_qcm = SGS100A(**settings.get("_LO_QCM_init_settings"))
        self._connected = True

    def setup(self, settings):
        """Configures instruments using the latest calibration settings.

        Args:
            settings (dict): Dictionary with calibration data.
                See ``examples/tii_single_qubit/tii_single_qubit_settings.json``
                for an example for this dictionary.
        """
        self.qrm.setup(**settings.get("_QRM_settings"))
        self.qcm.setup(**settings.get("_QCM_settings"))
        self.LO_qrm.setup(**settings.get("_LO_QRM_settings"))
        self.LO_qcm.setup(**settings.get("_LO_QCM_settings"))

    def setup_platform(self, settings):
        """Updates the platform parameters.

        Args:
            settings (dict): Dictionary with platform data.
        """
        self.data_folder = settings.get("data_folder", self.data_folder)
        self.hardware_avg = settings.get("hardware_avg", self.hardware_avg)
        self.sampling_rate = settings.get("sampling_rate", self.sampling_rate)
        self.software_averages = settings.get("software_averages", self.software_averages)
        self.repetition_duration = settings.get("repetition_duration", self.repetition_duration)

    def start(self):
        """Turns-on the local oscillators.

        The QBlox insturments are turned-on automatically during execution after
        the required pulse sequences are loaded.
        """
        self.LO_qcm.on()
        self.LO_qrm.on()

    def stop(self):
        """Turns-off all the lab instruments."""
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

    def __del__(self):
        self.disconnect()

    def execute(self, sequence):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.

        Returns:
            Readout results acquired by :class:`qibolab.instruments.qblox.PulsarQRM`
            after execution.
        """
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
