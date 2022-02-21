from qibo.config import raise_error, log
from qibolab.platforms.abstract import AbstractPlatform


class ICPlatform(AbstractPlatform):
    """Platform for controlling quantum devices with IC."""

    def __init__(self, name, runcard):
        self._instruments = []
        self._lo = []
        self._adc = []
        super().__init__(name, runcard)

    @property
    def qubits(self):
        return self._settings.get("qubits")
    
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

                    if params.get("adc"):
                        self._adc.append(inst)
                    
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
            self._adc = []
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

        from qibolab.pulses import ReadoutPulse

        qubits_to_measure = []
        measurement_results = []
        pulse_mapping = {}

        for pulse in sequence.pulses:
            # Assign pulses to each respective waveform generator
            qubit = self.fetch_qubit(pulse.qubit)
            playback_device = qubit.get("playback")

            # Track each qubit to measure
            if isinstance(pulse, ReadoutPulse):
                qubits_to_measure.append(pulse.qubit)
                playback_device = qubit.get("playback_readout")

            if playback_device not in pulse_mapping.keys():
                pulse_mapping[playback_device] = []
            pulse_mapping[playback_device].append(pulse)
    
        # Translate and upload the pulse for each device
        for device, subsequence in pulse_mapping.items():
            inst = self.fetch_instrument(device)
            inst.upload(inst.translate(subsequence, nshots))
            inst.play_sequence()

        for adc in self._adc:
            adc.arm(nshots)
        
        # Start the experiment sequence
        self.start_experiment()

        # Fetch the experiment results
        for qubit_id in qubits_to_measure:
            qubit = self.fetch_qubit(qubit_id)
            inst = self.fetch_instrument(qubit.get("readout"))
            measurement_results.append(inst.result(qubit.get("readout_frequency")))

        if len(qubits_to_measure) == 1:
            return measurement_results[0]
        return measurement_results

    def fetch_instrument(self, name):
        """
        Fetches for instruemnt from added instruments
        """
        try:
            res = next(inst for inst in self._instruments if inst.name == name)
            return res
        except StopIteration:
            raise_error(Exception, "Instrument not found")

    def fetch_qubit(self, qubit_id=0):
        """
        Fetches the qubit based on the id
        """
        return self.qubits.get("qubit_{}".format(qubit_id))

    def start_experiment(self):
        """
        Starts the instrument to start the experiment sequence
        """
        inst = self.fetch_instrument(self._settings.get("settings").get("experiment_start_instrument"))
        inst.start_experiment()
