import copy

from qibolab.platforms.multiqubit import MultiqubitPlatform


class Qubit:
    """Describes a single qubit in pulse control and readout extraction.

    Args:
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

    def __init__(
        self,
        pi_pulse,
        readout_pulse,
        readout_frequency,
        resonator_spectroscopy_max_ro_voltage,
        rabi_oscillations_pi_pulse_min_voltage,
        playback,
        playback_readout,
        readout,
        readout_channels,
    ):
        self.pi_pulse = pi_pulse
        self.readout_pulse = readout_pulse
        self.readout_frequency = readout_frequency
        self.max_readout_voltage = resonator_spectroscopy_max_ro_voltage
        self.min_readout_voltage = rabi_oscillations_pi_pulse_min_voltage
        self.playback = playback
        self.playback_readout = playback_readout
        self.readout = readout
        self.readout_channels = readout_channels


class ICPlatform(MultiqubitPlatform):
    """Platform for controlling quantum devices with IC.

    Example:
        .. code-block:: python

            from qibolab import Platform

            platform = Platform("icarusq")

    """

    def __init__(self, name, runcard):
        self._instruments = []
        self._lo = []
        self._adc = []
        self._last_sequence = None
        super().__init__(name, runcard)
        self.qubits = []
        qubits = self.settings.get("qubits")
        for qubit_dict in qubits.values():
            self.qubits.append(Qubit(**qubit_dict))

    def run_calibration(self):  # pragma: no cover
        from qibo.config import raise_error

        raise_error(NotImplementedError)

    def execute_pulse_sequence(self, sequence, nshots=None):
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
            Readout results acquired by the assigned readout instrument
            after execution.
        """
        if not self.is_connected:
            from qibo.config import raise_error

            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg

        from qibolab.pulses import ReadoutPulse

        qubits_to_measure = []
        measurement_results = []
        pulse_mapping = {}
        seq_serial = {}

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
                seq_serial[playback_device] = []
            # Map the pulse to the associated playback instrument.
            pulse_mapping[playback_device].append(pulse)
            seq_serial[playback_device].append(pulse.serial)

        # Translate and upload the pulse subsequence for each device if needed
        for device, subsequence in pulse_mapping.items():
            inst = self.fetch_instrument(device)
            if self._last_sequence is None or seq_serial[device] != self._last_sequence[device]:
                inst.upload(inst.translate(subsequence, nshots))
            inst.play_sequence()
        self._last_sequence = seq_serial

        for adc in self._adc:
            adc.arm(nshots)

        # Start the experiment sequence
        self.start_experiment()

        # Fetch the experiment results
        for qubit_id in set(qubits_to_measure):
            qubit = self.fetch_qubit(qubit_id)
            inst = self.fetch_instrument(qubit.readout)
            measurement_results.append(inst.result(qubit.readout_frequency, qubit.readout_channels))

        if len(qubits_to_measure) == 1:
            return measurement_results[0]
        return measurement_results

    def fetch_instrument(self, name):
        """Returns a reference to an instrument."""
        try:
            res = next(inst for inst in self._instruments if inst.name == name)
            return res
        except StopIteration:
            from qibo.config import raise_error

            raise_error(Exception, "Instrument not found")

    def fetch_qubit(self, qubit_id=0) -> Qubit:
        """Fetches the qubit based on the id."""
        return self.qubits[qubit_id]

    def start_experiment(self):
        """Starts the instrument to start the experiment sequence."""
        inst = self.fetch_instrument(self.settings.get("settings").get("experiment_start_instrument"))
        inst.start_experiment()

    def fetch_qubit_pi_pulse(self, qubit_id=0) -> dict:
        """Fetches the qubit pi-pulse."""
        # Use copy to avoid mutability
        return copy.copy(self.fetch_qubit(qubit_id).pi_pulse)

    def fetch_qubit_readout_pulse(self, qubit_id=0) -> dict:
        """Fetches the qubit readout pulse."""
        # Use copy to avoid mutability
        return copy.copy(self.fetch_qubit(qubit_id).readout_pulse)
