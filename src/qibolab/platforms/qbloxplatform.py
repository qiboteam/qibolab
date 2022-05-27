from qibolab.platforms.abstract import AbstractPlatform


class QBloxPlatform(AbstractPlatform):
    """Platform for controlling quantum devices using QCM and QRM.

    Example:
        .. code-block:: python

            from qibolab import Platform

            platform = Platform("tiiq")

    """

    def __init__(self, name, runcard):
        self._qrm = None
        self._qcm = None
        self._LO_qrm = None
        self._LO_qcm = None
        super().__init__(name, runcard)

        self.last_qcm_pulses = None
        self.last_qrm_pulses = None
        
    def __getstate__(self):
        data = super().__getstate__()
        data.update({
            "last_qcm_pulses": self.last_qcm_pulses,
            "last_qrm_pulses": self.last_qrm_pulses
        })
        return data

    def __setstate__(self, data):
        super().__setstate__(data)
        self.last_qcm_pulses = data.get("last_qcm_pulses")
        self.last_qrm_pulses = data.get("last_qrm_pulses")

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

    def run_calibration(self, show_plots=False):  # pragma: no cover
        """Executes calibration routines and updates the settings yml file"""
        # TODO: Consider testing this
        from qibolab.calibration import calibration
        ac = calibration.Calibration(self, show_plots)
        ac.auto_calibrate_plaform()     

        # update instruments with new calibration settings
        self.setup()

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            from qibo.config import log
            log.info(f"Connecting to {self.name} instruments.")
            try:
                from qibolab.instruments import PulsarQRM, PulsarQCM, SGS100A
                self._qrm = PulsarQRM(**self._settings.get("QRM_init_settings"))
                self._qcm = PulsarQCM(**self._settings.get("QCM_init_settings"))
                self._LO_qrm = SGS100A(**self._settings.get("LO_QRM_init_settings"))
                self._LO_qcm = SGS100A(**self._settings.get("LO_QCM_init_settings"))
                self.is_connected = True
            except Exception as exception:
                from qibo.config import raise_error
                raise_error(RuntimeError, "Cannot establish connection to "
                            f"{self.name} instruments. "
                            f"Error captured: '{exception}'")

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
        """Executes a pulse sequence. Pulses are being cached so that are not reuploaded 
            if they are the same as the ones sent previously. This greatly accelerates 
            some characterization routines that recurrently use the same set of pulses, 
            i.e. qubit and resonator spectroscopy, spin echo, and future circuits based on
            fixed gates.

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
            from qibo.config import raise_error
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg

        # Translate and upload instructions to instruments
        if sequence.qcm_pulses:
            if self.last_qcm_pulses != [pulse.serial() for pulse in sequence.qcm_pulses]:
                waveforms, program = self._qcm.translate(sequence, self.delay_before_readout, nshots)
                self._qcm.upload(waveforms, program, self.data_folder)
        if sequence.qrm_pulses:
            if self.last_qrm_pulses != [pulse.serial() for pulse in sequence.qrm_pulses]:
                waveforms, program = self._qrm.translate(sequence, self.delay_before_readout, nshots)
                self._qrm.upload(waveforms, program, self.data_folder)

        # Execute instructions
        if sequence.qcm_pulses:
            self._qcm.play_sequence()
        if sequence.qrm_pulses:
            # TODO: Find a better way to pass the readout pulse here
            acquisition_results = self._qrm.play_sequence_and_acquire(sequence.qrm_pulses[0])
        else:
            acquisition_results = None

        self.last_qcm_pulses = [pulse.serial() for pulse in sequence.qcm_pulses]
        self.last_qrm_pulses = [pulse.serial() for pulse in sequence.qrm_pulses]


        return acquisition_results

    def execute_pulse_sequence(self):
        from qibo.config import raise_error
        raise_error(NotImplementedError)