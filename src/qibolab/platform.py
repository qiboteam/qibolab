"""A platform for executing quantum algorithms."""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import networkx as nx
from qibo.config import log, raise_error

from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller, Instrument, InstrumentId
from qibolab.native import NativeType
from qibolab.pulses import PulseSequence, ReadoutPulse
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.sweeper import Sweeper

InstrumentMap = Dict[InstrumentId, Instrument]
QubitMap = Dict[QubitId, Qubit]
QubitPairMap = Dict[QubitPairId, QubitPair]


def unroll_sequences(sequences: List[PulseSequence], relaxation_time: int) -> PulseSequence:
    """Unrolls a list of pulse sequences to a single pulse sequence with multiple measurements.

    Args:
        sequences (list): List of pulse sequences to unroll.
        relaxation_time (int): Time in ns to wait for the qubit to relax between
            playing different sequences.
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

    def default(self, options: ExecutionParameters):
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

    is_connected: bool = False
    """Flag for whether we are connected to the physical instruments."""
    two_qubit_native_types: NativeType = field(default_factory=lambda: NativeType(0))
    """Types of two qubit native gates. Used by the transpiler."""
    topology: nx.Graph = field(default_factory=nx.Graph)
    """Graph representing the qubit connectivity in the quantum chip."""

    def __post_init__(self):
        log.info("Loading platform %s", self.name)
        if self.resonator_type is None:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

        for pair in self.pairs.values():
            self.two_qubit_native_types |= pair.native_gates.types
        if self.two_qubit_native_types is NativeType(0):
            # dummy value to avoid transpiler failure for single qubit devices
            self.two_qubit_native_types = NativeType.CZ

        self.topology.add_nodes_from(self.qubits.keys())
        self.topology.add_edges_from([(pair.qubit1.name, pair.qubit2.name) for pair in self.pairs.values()])

    def __str__(self):
        return self.name

    @property
    def nqubits(self) -> int:
        """Total number of usable qubits in the QPU.."""
        # TODO: Seperate couplers from qubits (PR #508)
        return len([qubit for qubit in self.qubits if not (isinstance(qubit, str) and "c" in qubit)])

    def update(self, updates: dict):
        r"""Updates platform common runcard parameters after calibration actions.

        Args:
            updates (dict): Dictionary containing the parameters to update the runcard.
                            A typical dictionary should be of the following form
                            {`parameter_to_update_in_runcard`: {`qubit0`:`par_value_qubit0`, ..., `qubit_i`:`par_value_qubit_i`, ...}}.
                            The parameters that can be updated by this method are:

                            - readout_frequency (GHz)
                            - readout_attenuation (dimensionless)
                            - bare_resonator_frequency (GHz)
                            - sweetspot(V)
                            - drive_frequency (GHz)
                            - readout_amplitude (dimensionless)
                            - drive_amplitude (dimensionless)
                            - drive_length
                            - t2 (ns)
                            - t2_spin_echo (ns)
                            - t1 (ns)
                            - thresold(V)
                            - iq_angle(deg)
                            - mean_gnd_states(V)
                            - mean_exc_states(V)
                            - beta(dimensionless)

        """
        for par, values in updates.items():
            for qubit, value in values.items():
                # resonator_spectroscopy / resonator_spectroscopy_flux / resonator_punchout_attenuation
                if par == "readout_frequency":
                    freq = int(value * 1e9)
                    mz = self.qubits[qubit].native_gates.MZ
                    mz.frequency = freq
                    if mz.if_frequency is not None:
                        mz.if_frequency = freq - self.get_lo_readout_frequency(qubit)
                    self.qubits[qubit].readout_frequency = freq

                # resonator_punchout_attenuation
                elif par == "readout_attenuation":
                    self.qubits[qubit].readout.attenuation = value

                # resonator_punchout_attenuation
                elif par == "bare_resonator_frequency":
                    freq = int(value * 1e9)
                    self.qubits[qubit].bare_resonator_frequency = freq

                # resonator_spectroscopy_flux / qubit_spectroscopy_flux
                elif par == "sweetspot":
                    sweetspot = float(value)
                    self.qubits[qubit].sweetspot = sweetspot
                    if self.qubits[qubit].flux is not None:
                        # set sweetspot as the flux offset (IS THIS NEEDED?)
                        self.qubits[qubit].flux.offset = sweetspot

                # qubit_spectroscopy / qubit_spectroscopy_flux / ramsey
                elif par == "drive_frequency":
                    freq = int(value * 1e9)
                    self.qubits[qubit].native_gates.RX.frequency = freq
                    self.qubits[qubit].drive_frequency = freq

                elif "amplitude" in par:
                    amplitude = float(value)
                    # resonator_spectroscopy
                    if par == "readout_amplitude" and not math.isnan(amplitude):
                        self.qubits[qubit].native_gates.MZ.amplitude = amplitude

                    # rabi_amplitude / flipping
                    if par == "drive_amplitude" or par == "amplitudes":
                        self.qubits[qubit].native_gates.RX.amplitude = amplitude

                # rabi_duration
                elif par == "drive_length":
                    self.qubits[qubit].native_gates.RX.duration = int(value)

                # ramsey
                elif par == "t2":
                    self.qubits[qubit].T2 = float(value)

                # spin_echo
                elif par == "t2_spin_echo":
                    self.qubits[qubit].T2_spin_echo = float(value)

                # t1
                elif par == "t1":
                    self.qubits[qubit].T1 = float(value)

                # classification
                elif par == "threshold":
                    self.qubits[qubit].threshold = float(value)

                # classification
                elif par == "iq_angle":
                    self.qubits[qubit].iq_angle = float(value)

                # classification
                elif par == "mean_gnd_states":
                    self.qubits[qubit].mean_gnd_states = [float(voltage) for voltage in value]

                # classification
                elif par == "mean_exc_states":
                    self.qubits[qubit].mean_exc_states = [float(voltage) for voltage in value]

                # drag pulse tunning
                elif "beta" in par:
                    rx = self.qubits[qubit].native_gates.RX
                    shape = rx.shape
                    rel_sigma = re.findall(r"[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+", shape)[0]
                    rx.shape = f"Drag({rel_sigma}, {float(value)})"

                elif "length" in par:  # assume only drive length
                    self.qubits[qubit].native_gates.RX.duration = int(value)

                elif par == "classifiers_hpars":
                    self.qubits[qubit].classifiers_hpars = value

                else:
                    raise_error(ValueError, f"Unknown parameter {par} for qubit {qubit}")

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
                new_result = getattr(instrument, method)(self.qubits, sequences, options)
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
        options = self.settings.default(options)

        time = (sequence.duration + options.relaxation_time) * options.nshots * 1e-9
        log.info(f"Minimal execution time (seq): {time}")

        return self._execute("play", sequence, options, **kwargs)

    def execute_pulse_sequences(self, sequences: List[PulseSequence], options: ExecutionParameters, **kwargs):
        """
        Args:
            sequence (List[:class:`qibolab.pulses.PulseSequence`]): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.
        """
        options = self.settings.default(options)

        duration = sum(seq.duration for seq in sequences)
        time = (duration + options.relaxation_time) * options.nshots * 1e-9
        log.info(f"Minimal execution time (seq): {time}")

        try:
            return self._execute("play_sequences", sequences, options, **kwargs)
        except NotImplementedError:
            sequence, readouts = unroll_sequences(sequences, options.relaxation_time)
            result = self._execute("play", sequence, options, **kwargs)
            results = {}
            for sequence in sequences:
                for ro_pulse in sequence.ro_pulses:
                    serial = ro_pulse.serial
                    if serial not in results:
                        results[ro_pulse.qubit] = results[serial] = [result[s] for s in readouts[serial]]
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

        time = (sequence.duration + options.relaxation_time) * options.nshots * 1e-9
        for sweep in sweepers:
            time *= len(sweep.values)
        log.info(f"Minimal execution time (sweep): {time}")

        result = {}
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.sweep(self.qubits, sequence, options, *sweepers)
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

    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        qubit = self.get_qubit(qubit)
        return self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)

    def create_CZ_pulse_sequence(self, qubits, start=0):
        # Check in the settings if qubits[0]-qubits[1] is a key
        pair = tuple(sorted(self.get_qubit(q) for q in qubits))
        if pair not in self.pairs or self.pairs[pair].native_gates.CZ is None:
            raise_error(
                ValueError,
                f"Calibration for CZ gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.CZ.sequence(start)

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
        self.qubits[qubit].flux.bias = bias

    def get_bias(self, qubit):
        """Get bias value. Usefeul for calibration routines involving flux."""
        return self.qubits[qubit].flux.bias
