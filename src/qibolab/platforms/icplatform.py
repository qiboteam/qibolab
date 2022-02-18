from qibo.config import raise_error, log
from qibolab.platforms.abstract import AbstractPlatform


class ICPlatform(AbstractPlatform):
    """Platform for controlling quantum devices with IC."""

    def __init__(self, name, runcard):
        self._instruments = []
        self._lo = []
        super().__init__(name, runcard)

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            log.info(f"Connecting to {self.name} instruments.")
            try:
                import qibolab.instruments as qi
                instruments = self._settings.get("instruments")
                for params in instruments.values():
                    inst = getattr(qi, params.get("type"))(**params.get("init_settings"))
                    self._instruments.append(inst)
                    if params.get("lo"):
                        self._lo.append(inst)
                    
                self.is_connected = True
            except Exception as exception:
                raise_error(RuntimeError, "Cannot establish connection to "
                            f"{self.name} instruments. "
                            f"Error captured: '{exception}'")
        self.setup()

    def setup(self):
        """Configures instruments using the loaded calibration settings."""
        if self.is_connected:
            instruments = self._settings.get("instruments")
            for inst in self._instruments:
                inst.setup(**instruments.get(inst.name).get("settings"))

    def start(self):
        """Turns on the local oscillators.

        At this point, the pulse sequence have not been uploaded to the DACs, so they will not be started yet.
        """
        for lo in self._lo:
            lo.start()

    def stop(self):
        """Turns off all the lab instruments."""
        for inst in self._instruments:
            inst.stop()

    def disconnect(self):
        """Disconnects from the lab instruments."""
        if self.is_connected:
            for inst in self._instruments:
                inst.close()
            self._instruments = []
            self._lo = []
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
            raise_error(
                RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg

        sobj = {}
        robj = {}
        from qibolab.pulses import ReadoutPulse
        for pulse in sequence.pulses:
            # Assign pulses to each respective waveform generator
            if pulse.device not in sobj.keys():
                sobj[pulse.device] = []
            sobj[pulse.device].append(pulse)

            # Track each readout pulse and frequency to its readout device
            if isinstance(pulse, ReadoutPulse):
                if pulse.adc not in robj.keys():
                    robj[pulse.adc] = []
                robj[pulse.adc].append(pulse.frequency)
        # Translate and upload the pulse for each device
        for device, seq in sobj.items():
            self.ic.translate_and_upload(device, seq, nshots)

        self.ic.arm_adc(nshots)
        # Trigger the experiment
        self.ic.trigger_experiment()

        # Fetch the experiment results
        result = self.ic.result(robj)
        return result
