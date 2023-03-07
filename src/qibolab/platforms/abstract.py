from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import yaml
from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit

from qibolab.designs.channels import Channel
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse
from qibolab.transpilers import can_execute, transpile


@dataclass
class Qubit:
    """Representation of a physical qubit.

    Qubit objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.

    Args:
        name (int, str): Qubit number or name.
        readout (:class:`qibolab.platforms.utils.Channel`): Channel used to
            readout pulses to the qubit.
        feedback (:class:`qibolab.platforms.utils.Channel`): Channel used to
            get readout feedback from the qubit.
        drive (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send drive pulses to the qubit.
        flux (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send flux pulses to the qubit.
        Other characterization parameters for the qubit, loaded from the runcard.
    """

    name: str

    readout_frequency: int = 0
    drive_frequency: int = 0
    sweetspot: float = 0
    peak_voltage: float = 0
    pi_pulse_amplitude: float = 0
    T1: int = 0
    T2: int = 0
    state0_voltage: int = 0
    state1_voltage: int = 0
    mean_gnd_states: complex = 0 + 0.0j
    mean_exc_states: complex = 0 + 0.0j
    resonator_polycoef_flux: List[float] = field(default_factory=list)

    # filters used for applying CZ gate
    filter: dict = field(default_factory=dict)
    # parameters for single shot classification
    threshold: Optional[float] = None
    iq_angle: float = 0.0
    # required for integration weights (not sure if it should be here)
    rotation_angle: float = 0.0
    # required for mixers (not sure if it should be here)
    mixer_drive_g: float = 0.0
    mixer_drive_phi: float = 0.0
    mixer_readout_g: float = 0.0
    mixer_readout_phi: float = 0.0

    readout: Optional[Channel] = None
    feedback: Optional[Channel] = None
    twpa: Optional[Channel] = None
    drive: Optional[Channel] = None
    flux: Optional[Channel] = None

    def __post_init__(self):
        # register qubit in ``flux`` channel so that we can access
        # ``sweetspot`` and ``filters`` at the channel level
        if self.flux:
            self.flux.qubit = self

    @property
    def channels(self):
        for channel in [self.readout, self.feedback, self.drive, self.flux, self.twpa]:
            if channel is not None:
                yield channel


class AbstractPlatform(ABC):
    """Abstract platform for controlling quantum devices.

    Args:
        name (str): name of the platform.
        runcard (str): path to the yaml file containing the platform setup.
    """

    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard

        self.qubits = {}

        # Values for the following are set from the runcard in ``reload_settings``
        self.settings = None
        self.is_connected = False
        self.nqubits = None
        self.resonator_type = None
        self.topology = None
        self.relaxation_time = None
        self.sampling_rate = None

        self.native_single_qubit_gates = {}
        self.native_two_qubit_gates = {}
        self.two_qubit_natives = set()
        # Load platform settings
        self.reload_settings()

    def __repr__(self):
        return self.name

    def _check_connected(self):
        if not self.is_connected:  # pragma: no cover
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        # TODO: Remove ``self.settings``
        with open(self.runcard) as file:
            settings = self.settings = yaml.safe_load(file)

        self.nqubits = settings["nqubits"]
        if "resonator_type" in self.settings:
            self.resonator_type = self.settings["resonator_type"]
        else:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

        self.topology = settings["topology"]

        self.relaxation_time = settings["settings"]["repetition_duration"]
        self.sampling_rate = settings["settings"]["sampling_rate"]

        # TODO: Create better data structures for native gates
        self.native_gates = settings["native_gates"]
        self.native_single_qubit_gates = self.native_gates["single_qubit"]
        if "two_qubit" in self.native_gates:
            self.native_two_qubit_gates = self.native_gates["two_qubit"]
            for gates in self.native_gates["two_qubit"].values():
                self.two_qubit_natives |= set(gates.keys())
        else:
            # dummy value to avoid transpiler failure for single qubit devices
            self.two_qubit_natives = {"CZ"}

        # Load characterization settings and create ``Qubit`` and ``Channel`` objects
        for q in settings["qubits"]:
            if q in self.qubits:
                for name, value in settings["characterization"]["single_qubit"][q].items():
                    setattr(self.qubits[q], name, value)
            else:
                self.qubits[q] = Qubit(q, **settings["characterization"]["single_qubit"][q])

    @abstractmethod
    def connect(self):
        """Connects to instruments."""

    @abstractmethod
    def setup(self):
        """Prepares instruments to execute experiments."""

    @abstractmethod
    def start(self):
        """Starts all the instruments."""

    @abstractmethod
    def stop(self):
        """Starts all the instruments."""

    @abstractmethod
    def disconnect(self):
        """Disconnects to instruments."""

    def transpile(self, circuit: Circuit):
        """Transforms a circuit to pulse sequence.

        Args:
            circuit (qibo.models.Circuit): Qibo circuit that respects the platform's
                connectivity and native gates.

        Returns:
            sequence (qibolab.pulses.PulseSequence): Pulse sequence that implements the
                circuit on the qubit.
        """
        if not can_execute(circuit, self.two_qubit_natives):
            circuit, _ = transpile(circuit, self.two_qubit_natives)

        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)

        # keep track of gates that were already added to avoid adding them twice
        already_processed = set()
        # process circuit gates
        for moment in circuit.queue.moments:
            moment_start = sequence.finish
            for gate in moment:
                if isinstance(gate, gates.I) or gate is None or gate in already_processed:
                    pass

                elif isinstance(gate, gates.Z):
                    qubit = gate.target_qubits[0]
                    virtual_z_phases[qubit] += np.pi

                elif isinstance(gate, gates.RZ):
                    qubit = gate.target_qubits[0]
                    virtual_z_phases[qubit] += gate.parameters[0]

                elif isinstance(gate, gates.U3):
                    qubit = gate.target_qubits[0]
                    # Transform gate to U3 and add pi/2-pulses
                    theta, phi, lam = gate.parameters
                    # apply RZ(lam)
                    virtual_z_phases[qubit] += lam
                    # Fetch pi/2 pulse from calibration
                    RX90_pulse_1 = self.create_RX90_pulse(
                        qubit,
                        start=max(sequence.get_qubit_pulses(qubit).finish, moment_start),
                        relative_phase=virtual_z_phases[qubit],
                    )
                    # apply RX(pi/2)
                    sequence.add(RX90_pulse_1)
                    # apply RZ(theta)
                    virtual_z_phases[qubit] += theta
                    # Fetch pi/2 pulse from calibration
                    RX90_pulse_2 = self.create_RX90_pulse(
                        qubit, start=RX90_pulse_1.finish, relative_phase=virtual_z_phases[qubit] - np.pi
                    )
                    # apply RX(-pi/2)
                    sequence.add(RX90_pulse_2)
                    # apply RZ(phi)
                    virtual_z_phases[qubit] += phi

                elif isinstance(gate, gates.M):
                    # Add measurement pulse
                    measurement_start = max(sequence.get_qubit_pulses(*gate.target_qubits).finish, moment_start)
                    gate.pulses = ()
                    for qubit in gate.target_qubits:
                        MZ_pulse = self.create_MZ_pulse(qubit, start=measurement_start)
                        sequence.add(MZ_pulse)
                        gate.pulses = (*gate.pulses, MZ_pulse.serial)

                elif isinstance(gate, gates.CZ):
                    # create CZ pulse sequence with start time = 0
                    cz_sequence, cz_virtual_z_phases = self.create_CZ_pulse_sequence(gate.qubits)

                    # determine the right start time based on the availability of the qubits involved
                    cz_qubits = {*cz_sequence.qubits, *gate.qubits}
                    cz_start = max(sequence.get_qubit_pulses(*cz_qubits).finish, moment_start)

                    # shift the pulses
                    for pulse in cz_sequence.pulses:
                        pulse.start += cz_start

                    # add pulses to the sequence
                    sequence.add(cz_sequence)

                    # update z_phases registers
                    for qubit in cz_virtual_z_phases:
                        virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

                else:  # pragma: no cover
                    raise_error(
                        NotImplementedError,
                        f"Transpilation of {gate.__class__.__name__} gate has not been implemented yet.",
                    )

                already_processed.add(gate)
        return sequence

    @abstractmethod
    def execute_pulse_sequence(self, sequence, nshots=1024, relaxation_time=None):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment. Default is 1024.
            relaxation_time (int): Time to wait for the qubit to relax to its ground state between shots in ns.
                If ``None`` the default value provided as ``repetition_duration`` in the runcard will be used.

        Returns:
            Readout results acquired by after execution.
        """

    def __call__(self, sequence, nshots=1024, relaxation_time=None):
        return self.execute_pulse_sequence(sequence, nshots, relaxation_time)

    def sweep(self, sequence, *sweepers, nshots=1024, average=True, relaxation_time=None):
        """Executes a pulse sequence for different values of sweeped parameters.
        Useful for performing chip characterization.

        Example:
            .. testcode::

                import numpy as np
                from qibolab.platform import Platform
                from qibolab.sweeper import Sweeper, Parameter
                from qibolab.pulses import PulseSequence


                platform = Platform("dummy")
                sequence = PulseSequence()
                parameter = Parameter.frequency
                pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
                sequence.add(pulse)
                parameter_range = np.random.randint(10, size=10)
                sweeper = Sweeper(parameter, parameter_range, [pulse])
                platform.sweep(sequence, sweeper)


        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            sweepers (:class:`qibolab.sweeper.Sweeper`): Sweeper objects that specify which
                parameters are being sweeped.
            nshots (int): Number of shots to sample from the experiment. Default is 1024.
            relaxation_time (int): Time to wait for the qubit to relax to its ground state between shots in ns.
                If ``None`` the default value provided as ``repetition_duration`` in the runcard will be used.
            average (bool): If ``True`` the IQ results of individual shots are averaged on hardware.

        Returns:
            Readout results acquired by after execution.
        """
        raise_error(NotImplementedError, f"Platform {self.name} does not support sweeping.")

    def get_qd_channel(self, qubit):
        if self.qubits[qubit].drive:
            return self.qubits[qubit].drive.name
        else:
            return self.settings["qubit_channel_map"][qubit][1]

    # TODO: Maybe create a dataclass for native gates
    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"] / 2.0
        qd_shape = pulse_kwargs["shape"]
        qd_channel = self.get_qd_channel(qubit)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        qd_channel = self.get_qd_channel(qubit)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_CZ_pulse_sequence(self, qubits, start=0):
        # Check in the settings if qubits[0]-qubits[1] is a key
        if f"{qubits[0]}-{qubits[1]}" in self.settings["native_gates"]["two_qubit"]:
            pulse_sequence_settings = self.settings["native_gates"]["two_qubit"][f"{qubits[0]}-{qubits[1]}"]["CZ"]
        elif f"{qubits[1]}-{qubits[0]}" in self.settings["native_gates"]["two_qubit"]:
            pulse_sequence_settings = self.settings["native_gates"]["two_qubit"][f"{qubits[1]}-{qubits[0]}"]["CZ"]
        else:
            raise_error(
                ValueError,
                f"Calibration for CZ gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )

        # If settings contains only one pulse dictionary, convert it into a list that can be iterated below
        if isinstance(pulse_sequence_settings, dict):
            pulse_sequence_settings = [pulse_sequence_settings]

        from qibolab.pulses import FluxPulse, PulseSequence

        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)

        for pulse_settings in pulse_sequence_settings:
            if pulse_settings["type"] == "qf":
                qf_duration = pulse_settings["duration"]
                qf_amplitude = pulse_settings["amplitude"]
                qf_shape = pulse_settings["shape"]
                qubit = pulse_settings["qubit"]
                if self.qubits[qubit].flux:
                    qf_channel = self.qubits[qubit].flux.name
                else:
                    qf_channel = self.settings["qubit_channel_map"][qubit][2]
                sequence.add(
                    FluxPulse(
                        start + pulse_settings["relative_start"], qf_duration, qf_amplitude, qf_shape, qf_channel, qubit
                    )
                )
            elif pulse_settings["type"] == "virtual_z":
                virtual_z_phases[pulse_settings["qubit"]] += pulse_settings["phase"]
            else:
                raise NotImplementedError(
                    "Implementation of CZ gates using pulses of types other than `qf` or `virtual_z` is not supported yet."
                )

        return sequence, virtual_z_phases

    def create_MZ_pulse(self, qubit, start):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["MZ"]
        ro_duration = pulse_kwargs["duration"]
        ro_frequency = pulse_kwargs["frequency"]
        ro_amplitude = pulse_kwargs["amplitude"]
        ro_shape = pulse_kwargs["shape"]
        if self.qubits[qubit].readout:
            ro_channel = self.qubits[qubit].readout.name
        else:
            ro_channel = self.settings["qubit_channel_map"][qubit][0]
        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        qd_channel = self.get_qd_channel(qubit)
        return Pulse(start, duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_qubit_readout_pulse(self, qubit, start):
        return self.create_MZ_pulse(qubit, start)

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi/2 pulse with drag shape
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"] / 2.0
        qd_shape = pulse_kwargs["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.get_qd_channel(qubit)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi pulse with drag shape
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.get_qd_channel(qubit)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    @abstractmethod
    def set_lo_drive_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """

    @abstractmethod
    def get_lo_drive_frequency(self, qubit):
        """Get frequency of the qubit drive local oscillator in Hz."""

    @abstractmethod
    def set_lo_readout_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """

    @abstractmethod
    def get_lo_readout_frequency(self, qubit):
        """Get frequency of the qubit readout local oscillator in Hz."""

    @abstractmethod
    def set_attenuation(self, qubit, att):
        """Set attenuation value. Usefeul for calibration routines such as punchout.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            att (int): new value of the attenuation (dB).
        Returns:
            None
        """

    @abstractmethod
    def get_attenuation(self, qubit):
        """Get attenuation value. Usefeul for calibration routines such as punchout."""

    @abstractmethod
    def set_gain(self, qubit, gain):
        """Set gain value. Usefeul for calibration routines such as Rabi oscillations.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            gain (int): new value of the gain (dimensionless).
        Returns:
            None
        """

    @abstractmethod
    def get_gain(self, qubit):
        """Get gain value. Usefeul for calibration routines such as Rabi oscillations."""

    @abstractmethod
    def set_bias(self, qubit, bias):
        """Set bias value. Usefeul for calibration routines involving flux.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            bias (int): new value of the bias (V).
        Returns:
            None
        """

    @abstractmethod
    def get_bias(self, qubit):
        """Get bias value. Usefeul for calibration routines involving flux."""
