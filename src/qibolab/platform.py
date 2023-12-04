"""A platform for executing quantum algorithms."""

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import networkx as nx
from qibo.config import log, raise_error
from qibo.transpiler import NativeGates

from qibolab.couplers import Coupler
from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller, Instrument, InstrumentId
from qibolab.pulses import PulseSequence, ReadoutPulse
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.sweeper import Sweeper

InstrumentMap = Dict[InstrumentId, Instrument]
QubitMap = Dict[QubitId, Qubit]
CouplerMap = Dict[QubitId, Coupler]
QubitPairMap = Dict[QubitPairId, QubitPair]

NS_TO_SEC = 1e-9


def unroll_sequences(sequences: List[PulseSequence], relaxation_time: int) -> PulseSequence:
    """Unrolls a list of pulse sequences to a single pulse sequence with multiple measurements.

    Args:
        sequences (list): List of pulse sequences to unroll.
        relaxation_time (int): Time in ns to wait for the qubit to relax between
            playing different sequences.

    Returns:
        total_sequence (:class:`qibolab.pulses.PulseSequence`): Unrolled pulse sequence containing
            multiple measurements.
        readout_map (dict): Map from original readout pulse serials to the unrolled readout pulse
            serials. Required to construct the results dictionary that is returned after execution.
    """
    total_sequence = PulseSequence()
    readout_map = defaultdict(list)
    start = 0
    for sequence in sequences:
        for pulse in sequence:
            new_pulse = pulse.copy()
            new_pulse.start += start
            total_sequence.add(new_pulse)
            if isinstance(pulse, ReadoutPulse):
                readout_map[pulse.serial].append(new_pulse.serial)
        start = total_sequence.finish + relaxation_time
    return total_sequence, readout_map


@dataclass
class Settings:
    """Default execution settings read from the runcard."""

    nshots: int = 1024
    """Default number of repetitions when executing a pulse sequence."""
    sampling_rate: int = int(1e9)
    """Number of waveform samples supported by the instruments per second."""
    relaxation_time: int = int(1e5)
    """Time in ns to wait for the qubit to relax to its ground state between shots."""

    def fill(self, options: ExecutionParameters):
        """Use default values for missing execution options."""
        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        return options


@dataclass
class Platform:
    """Platform for controlling quantum devices."""

    name: str
    """Name of the platform."""
    qubits: QubitMap
    """Dictionary mapping qubit names to :class:`qibolab.qubits.Qubit` objects."""
    pairs: QubitPairMap
    """Dictionary mapping sorted tuples of qubit names to :class:`qibolab.qubits.QubitPair` objects."""
    instruments: InstrumentMap
    """Dictionary mapping instrument names to :class:`qibolab.instruments.abstract.Instrument` objects."""

    settings: Settings = field(default_factory=Settings)
    """Container with default execution settings."""
    resonator_type: Optional[str] = None
    """Type of resonator (2D or 3D) in the used QPU.
    Default is 3D for single-qubit chips and 2D for multi-qubit.
    """

    couplers: CouplerMap = field(default_factory=dict)
    """Dictionary mapping coupler names to :class:`qibolab.couplers.Coupler` objects."""

    is_connected: bool = False
    """Flag for whether we are connected to the physical instruments."""
    two_qubit_native_types: NativeGates = field(default_factory=lambda: NativeGates(0))
    """Types of two qubit native gates. Used by the transpiler."""
    topology: nx.Graph = field(default_factory=nx.Graph)
    """Graph representing the qubit connectivity in the quantum chip."""

    def __post_init__(self):
        log.info("Loading platform %s", self.name)
        if self.resonator_type is None:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

        for pair in self.pairs.values():
            self.two_qubit_native_types |= pair.native_gates.types
        if self.two_qubit_native_types is NativeGates(0):
            # dummy value to avoid transpiler failure for single qubit devices
            self.two_qubit_native_types = NativeGates.CZ

        self.topology.add_nodes_from(self.qubits.keys())
        self.topology.add_edges_from([(pair.qubit1.name, pair.qubit2.name) for pair in self.pairs.values()])

    def __str__(self):
        return self.name

    @property
    def nqubits(self) -> int:
        """Total number of usable qubits in the QPU.."""
        # TODO: Seperate couplers from qubits (PR #508)
        return len([qubit for qubit in self.qubits if not (isinstance(qubit, str) and "c" in qubit)])

    def connect(self):
        """Connect to all instruments."""
        if not self.is_connected:
            for instrument in self.instruments.values():
                try:
                    log.info(f"Connecting to instrument {instrument}.")
                    instrument.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {instrument} instruments. Error captured: '{exception}'",
                    )
        self.is_connected = True

    def setup(self):
        """Prepares instruments to execute experiments.

        Sets flux port offsets to the qubit sweetspots.
        """
        for instrument in self.instruments.values():
            instrument.setup()
        for qubit in self.qubits.values():
            if qubit.flux is not None and qubit.sweetspot != 0:
                qubit.flux.offset = qubit.sweetspot
        for coupler in self.couplers.values():
            if coupler.flux is not None and coupler.sweetspot != 0:
                coupler.flux.offset = coupler.sweetspot

    def start(self):
        """Starts all the instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.start()

    def stop(self):
        """Starts all the instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.stop()

    def disconnect(self):
        """Disconnects from instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.disconnect()
        self.is_connected = False

    def _execute(self, method, sequences, options, **kwargs):
        """Executes the sequences on the controllers"""
        result = {}

        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = getattr(instrument, method)(self.qubits, self.couplers, sequences, options)
                if isinstance(new_result, dict):
                    result.update(new_result)
                elif new_result is not None:
                    # currently the result of QMSim is not a dict
                    result = new_result

        return result

    def execute_pulse_sequence(self, sequence: PulseSequence, options: ExecutionParameters, **kwargs):
        """
        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        options = self.settings.fill(options)

        time = (sequence.duration + options.relaxation_time) * options.nshots * NS_TO_SEC
        log.info(f"Minimal execution time (sequence): {time}")

        return self._execute("play", sequence, options, **kwargs)

    @property
    def _controller(self):
        """Controller instrument used for splitting the unrolled sequences to batches.

        Used only by :meth:`qibolab.platform.Platform.execute_pulse_sequences` (unrolling).
        This method does not support platforms with more than one controller instruments.
        """
        controllers = [instr for instr in self.instruments.values() if isinstance(instr, Controller)]
        assert len(controllers) == 1
        return controllers[0]

    def execute_pulse_sequences(self, sequences: List[PulseSequence], options: ExecutionParameters, **kwargs):
        """
        Args:
            sequence (List[:class:`qibolab.pulses.PulseSequence`]): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        options = self.settings.fill(options)

        duration = sum(seq.duration for seq in sequences)
        time = (duration + len(sequences) * options.relaxation_time) * options.nshots * NS_TO_SEC
        log.info(f"Minimal execution time (unrolling): {time}")

        try:
            return self._execute("play_sequences", sequences, options, **kwargs)
        except NotImplementedError:
            # find readout pulses
            ro_pulses = {pulse.serial: pulse.qubit for sequence in sequences for pulse in sequence.ro_pulses}

            results = defaultdict(list)
            for batch in self._controller.split_batches(sequences):
                sequence, readouts = unroll_sequences(batch, options.relaxation_time)
                result = self._execute("play", sequence, options, **kwargs)
                for serial, new_serials in readouts.items():
                    results[serial].extend(result[ser] for ser in new_serials)

            for serial, qubit in ro_pulses.items():
                results[qubit] = results[serial]

            return results

    def sweep(self, sequence: PulseSequence, options: ExecutionParameters, *sweepers: Sweeper):
        """Executes a pulse sequence for different values of sweeped parameters.

        Useful for performing chip characterization.

        Example:
            .. testcode::

                import numpy as np
                from qibolab.dummy import create_dummy
                from qibolab.sweeper import Sweeper, Parameter
                from qibolab.pulses import PulseSequence
                from qibolab.execution_parameters import ExecutionParameters


                platform = create_dummy()
                sequence = PulseSequence()
                parameter = Parameter.frequency
                pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
                sequence.add(pulse)
                parameter_range = np.random.randint(10, size=10)
                sweeper = Sweeper(parameter, parameter_range, [pulse])
                platform.sweep(sequence, ExecutionParameters(), sweeper)

        Returns:
            Readout results acquired by after execution.
        """
        if options.nshots is None:
            options = replace(options, nshots=self.settings.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.settings.relaxation_time)

        time = (sequence.duration + options.relaxation_time) * options.nshots * NS_TO_SEC
        for sweep in sweepers:
            time *= len(sweep.values)
        log.info(f"Minimal execution time (sweep): {time}")

        result = {}
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.sweep(self.qubits, self.couplers, sequence, options, *sweepers)
                if isinstance(new_result, dict):
                    result.update(new_result)
                elif new_result is not None:
                    # currently the result of QMSim is not a dict
                    result = new_result

        return result

    def __call__(self, sequence, options):
        return self.execute_pulse_sequence(sequence, options)

    def get_qubit(self, qubit):
        """Return the name of the physical qubit corresponding to a logical qubit.

        Temporary fix for the compiler to work for platforms where the qubits
        are not named as 0, 1, 2, ...
        """
        try:
            return self.qubits[qubit].name
        except KeyError:
            return list(self.qubits.keys())[qubit]

    def get_coupler(self, coupler):
        """Return the name of the physical coupler corresponding to a logical coupler.

        Temporary fix for the compiler to work for platforms where the couplers
        are not named as 0, 1, 2, ...
        """
        try:
            return self.couplers[coupler].name
        except KeyError:
            return list(self.couplers.keys())[coupler]

    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)

    def create_RX12_pulse(self, qubit, start=0, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return self.qubits[qubit].native_gates.RX12.pulse(start, relative_phase)

    def create_CZ_pulse_sequence(self, qubits, start=0):
        # Check in the settings if qubits[0]-qubits[1] is a key
        pair = tuple(sorted(self.get_qubit(q) for q in qubits))
        if pair not in self.pairs or self.pairs[pair].native_gates.CZ is None:
            raise_error(
                ValueError,
                f"Calibration for CZ gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.CZ.sequence(start)

    def create_iSWAP_pulse_sequence(self, qubits, start=0):
        # Check in the settings if qubits[0]-qubits[1] is a key
        pair = tuple(sorted(self.get_qubit(q) for q in qubits))
        if pair not in self.pairs or self.pairs[pair].native_gates.iSWAP is None:
            raise_error(
                ValueError,
                f"Calibration for iSWAP gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.iSWAP.sequence(start)

    def create_MZ_pulse(self, qubit, start):
        qubit = self.get_qubit(qubit)
        return self.qubits[qubit].native_gates.MZ.pulse(start)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        qubit = self.get_qubit(qubit)
        pulse = self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)
        pulse.duration = duration
        return pulse

    def create_qubit_readout_pulse(self, qubit, start):
        qubit = self.get_qubit(qubit)
        return self.create_MZ_pulse(qubit, start)

    def create_coupler_pulse(self, coupler, start, duration=None, amplitude=None):
        coupler = self.get_coupler(coupler)
        pulse = self.couplers[coupler].native_pulse.CP.pulse(start)
        if duration is not None:
            pulse.duration = duration
        if amplitude is not None:
            pulse.amplitude = amplitude
        return pulse

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        qubit = self.get_qubit(qubit)
        pulse = self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)
        if beta is not None:
            pulse.shape = "Drag(5," + str(beta) + ")"
        return pulse

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        qubit = self.get_qubit(qubit)
        pulse = self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)
        if beta is not None:
            pulse.shape = "Drag(5," + str(beta) + ")"
        return pulse

    def set_lo_drive_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].drive.lo_frequency = freq

    def get_lo_drive_frequency(self, qubit):
        """Get frequency of the qubit drive local oscillator in Hz."""
        return self.qubits[qubit].drive.lo_frequency

    def set_lo_readout_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].readout.lo_frequency = freq

    def get_lo_readout_frequency(self, qubit):
        """Get frequency of the qubit readout local oscillator in Hz."""
        return self.qubits[qubit].readout.lo_frequency

    def set_lo_twpa_frequency(self, qubit, freq):
        """Set frequency of the local oscillator of the TWPA to which the qubit's feedline is connected to.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].twpa.lo_frequency = freq

    def get_lo_twpa_frequency(self, qubit):
        """Get frequency of the local oscillator of the TWPA to which the qubit's feedline is connected to in Hz."""
        return self.qubits[qubit].twpa.lo_frequency

    def set_lo_twpa_power(self, qubit, power):
        """Set power of the local oscillator of the TWPA to which the qubit's feedline is connected to.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            power (int): new value of the power in dBm.
        """
        self.qubits[qubit].twpa.lo_power = power

    def get_lo_twpa_power(self, qubit):
        """Get power of the local oscillator of the TWPA to which the qubit's feedline is connected to in dBm."""
        return self.qubits[qubit].twpa.lo_power

    def set_attenuation(self, qubit, att):
        """Set attenuation value. Usefeul for calibration routines such as punchout.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            att (int): new value of the attenuation (dB).
        Returns:
            None
        """
        self.qubits[qubit].readout.attenuation = att

    def get_attenuation(self, qubit):
        """Get attenuation value. Usefeul for calibration routines such as punchout."""
        return self.qubits[qubit].readout.attenuation

    def set_gain(self, qubit, gain):
        """Set gain value. Usefeul for calibration routines such as Rabi oscillations.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            gain (int): new value of the gain (dimensionless).
        Returns:
            None
        """
        raise_error(NotImplementedError, f"{self.name} does not support gain.")

    def get_gain(self, qubit):
        """Get gain value. Usefeul for calibration routines such as Rabi oscillations."""
        raise_error(NotImplementedError, f"{self.name} does not support gain.")

    def set_bias(self, qubit, bias):
        """Set bias value. Usefeul for calibration routines involving flux.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            bias (int): new value of the bias (V).
        Returns:
            None
        """
        if self.qubits[qubit].flux is None:
            raise_error(NotImplementedError, f"{self.name} does not have flux.")
        self.qubits[qubit].flux.offset = bias

    def get_bias(self, qubit):
        """Get bias value. Usefeul for calibration routines involving flux."""
        return self.qubits[qubit].flux.offset
