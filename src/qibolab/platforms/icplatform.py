import copy
from qibo.config import raise_error, log
from qibolab.platforms.abstract import AbstractPlatform

class Qubit:
    """Describes a single qubit in pulse control and readout extraction.

    Args:
        id (int): Qubit ID.
        pi_pulse (dict): Qubit pi-pulse parameters.
            See qibolab.pulses.Pulse for more information.
        readout_pulse (dict): Qubit readout pulse parameters.
            See qibolab.pulses.ReadoutPulse for more information.
        resonator_spectroscopy_max_ro_voltage (float): Readout voltage corresponding to the ground state of the qubit.
        rabi_oscillations_pi_pulse_min_voltage (float): Readout voltage corresponding to the excited state of the qubit.
        playback (str): Instrument name for playing the qubit XY control pulses.
        playback_readout (str): Instrument name for playing the qubit readout pulse.
        readout_frequency (float): Readout frequency for IQ demodulation.
        readout (str): Instrument name for reading the qubit.
        readout_channels (int, int[]): Channels on the instrument associated to qubit readout.
    """

    def __init__(self, pi_pulse, readout_pulse, readout_frequency, resonator_spectroscopy_max_ro_voltage, rabi_oscillations_pi_pulse_min_voltage,
             playback, playback_readout, readout, readout_channels):

        self.id = id
        self.pi_pulse = pi_pulse
        self.readout_pulse = readout_pulse
        self.readout_frequency = readout_frequency
        self.max_readout_voltage = resonator_spectroscopy_max_ro_voltage
        self.min_readout_voltage = rabi_oscillations_pi_pulse_min_voltage
        self.playback = playback
        self.playback_readout = playback_readout
        self.readout = readout
        self.readout_channels = readout_channels


class ICPlatform(AbstractPlatform):
    """Platform for controlling quantum devices with IC."""

    def __init__(self, name, runcard):
        self._instruments = []
        self._lo = []
        self._adc = []
        super().__init__(name, runcard)
        self.qubits = []
        qubits = self._settings.get("qubits")
        for qubit_dict in qubits.values():
            self.qubits.append(Qubit(**qubit_dict))
    
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
            playback_device = qubit.playback

            # Track each qubit to measure
            if isinstance(pulse, ReadoutPulse):
                qubits_to_measure.append(pulse.qubit)
                playback_device = qubit.playback_readout

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
            inst = self.fetch_instrument(qubit.readout)
            measurement_results.append(inst.result(qubit.readout_frequency))

        if len(qubits_to_measure) == 1:
            return measurement_results[0]
        return measurement_results

    def fetch_instrument(self, name):
        """Returns a reference to an instrument.
        """
        try:
            res = next(inst for inst in self._instruments if inst.name == name)
            return res
        except StopIteration:
            raise_error(Exception, "Instrument not found")

    def fetch_qubit(self, qubit_id=0) -> Qubit:
        """Fetches the qubit based on the id.
        """
        return self.qubits[qubit_id]

    def start_experiment(self):
        """Starts the instrument to start the experiment sequence.
        """
        inst = self.fetch_instrument(self._settings.get("settings").get("experiment_start_instrument"))
        inst.start_experiment()

    def fetch_qubit_pi_pulse(self, qubit_id=0) -> dict:
        """Fetches the qubit pi-pulse.
        """
        # Use copy to avoid mutability
        return copy.copy(self.fetch_qubit(qubit_id).pi_pulse) 

    def fetch_qubit_readout_pulse(self, qubit_id=0) -> dict:
        """Fetches the qubit readout pulse.
        """
        # Use copy to avoid mutability
        return copy.copy(self.fetch_qubit(qubit_id).readout_pulse)
