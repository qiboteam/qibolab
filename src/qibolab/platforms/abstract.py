import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import yaml
from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit

from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse
from qibolab.transpilers import can_execute, transpile


@dataclass
class Channel:
    """Representation of physical wire connection (channel).

    Name is used as a unique identifier for channels.
    Channel objects are instantiated by :class:`qibolab.platforms.platform.Platform`,
    but their attributes are modified and used by instrument designs.

    Args:
        name (str): Name of the channel as given in the platform runcard.

    Attributes:
        ports (list): List of tuples (controller (`str`), port (`int`))
            specifying the QM (I, Q) ports that the channel is connected.
        qubits (list): List of tuples (:class:`qibolab.platforms.utils.Qubit`, str)
            for the qubit connected to this channel and the role of the channel.
        Optional arguments holding local oscillators and related parameters.
        These are relevant only for mixer-based insturment designs.
    """

    name: str
    qubits: List[tuple] = field(default_factory=list, init=False, repr=False)
    ports: List[tuple] = field(default_factory=list, init=False)

    local_oscillator: Any = field(default=None, init=False)
    lo_frequency: float = field(default=0, init=False)
    lo_power: float = field(default=0, init=False)
    offset: float = field(default=0, init=False)
    filter: Optional[dict] = field(default=None, init=False)


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
    readout: Channel
    feedback: Channel
    twpa: Optional[Channel] = None
    drive: Optional[Channel] = None
    flux: Optional[Channel] = None

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
    ff_filter: List[float] = field(default_factory=list)
    fb_filter: List[float] = field(default_factory=list)
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

    @property
    def channels(self):
        for name in ["readout", "feedback", "drive", "flux", "twpa"]:
            channel = getattr(self, name)
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
        self.channels = {}

        self.settings = None
        self.is_connected = False
        self.nqubits = None
        self.resonator_type = None
        self.topology = None
        self.sampling_rate = None
        self.options = None

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

    def get_channel(self, name):
        """Returns an existing channel or creates a new one if it does not exist.

        Args:
            name (str): Name of the channel to get or create.
        """
        if name not in self.channels:
            self.channels[name] = Channel(name)
        return self.channels[name]

    def reload_settings(self):
        # TODO: Remove ``self.settings``
        with open(self.runcard) as file:
            settings = self.settings = yaml.safe_load(file)

        self.nqubits = settings["nqubits"]
        self.resonator_type = "3D" if self.nqubits == 1 else "2D"
        self.topology = settings["topology"]

        self.options = settings["settings"]
        self.sampling_rate = self.options["sampling_rate"]

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
            if len(settings["qubit_channel_map"][q]) == 3:
                readout, drive, flux = settings["qubit_channel_map"][q]
                feedback = None
                twpa = None
            else:
                readout, drive, flux, feedback, twpa = settings["qubit_channel_map"][q]
            self.qubits[q] = qubit = Qubit(
                q,
                readout=self.get_channel(readout),
                feedback=self.get_channel(feedback) if feedback else None,
                drive=self.get_channel(drive) if drive else None,
                flux=self.get_channel(flux) if flux else None,
                twpa=self.get_channel(twpa) if twpa else None,
                **settings["characterization"]["single_qubit"][q],
            )
            for mode in ["readout", "feedback", "drive", "flux", "twpa"]:
                channel = getattr(qubit, mode)
                if channel is not None:
                    channel.qubits.append((qubit, mode))

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
        sequence.virtual_z_phases = collections.defaultdict(int)
        clock = collections.defaultdict(int)
        # keep track of gates that were already added to avoid adding them twice
        added = set()
        for moment in circuit.queue.moments:
            for gate in moment:

                if isinstance(gate, gates.I) or gate is None or gate in added:
                    pass

                elif isinstance(gate, gates.Z):
                    qubit = gate.target_qubits[0]
                    sequence.virtual_z_phases[qubit] += np.pi

                elif isinstance(gate, gates.RZ):
                    qubit = gate.target_qubits[0]
                    sequence.virtual_z_phases[qubit] += gate.parameters[0]

                elif isinstance(gate, gates.U3):
                    qubit = gate.target_qubits[0]
                    # Transform gate to U3 and add pi/2-pulses
                    theta, phi, lam = gate.parameters
                    # apply RZ(lam)
                    sequence.virtual_z_phases[qubit] += lam
                    # Fetch pi/2 pulse from calibration
                    RX90_pulse_1 = self.create_RX90_pulse(
                        qubit, clock[qubit], relative_phase=sequence.virtual_z_phases[qubit]
                    )
                    # apply RX(pi/2)
                    sequence.add(RX90_pulse_1)
                    clock[qubit] += RX90_pulse_1.duration
                    # apply RZ(theta)
                    sequence.virtual_z_phases[qubit] += theta
                    # Fetch pi/2 pulse from calibration
                    RX90_pulse_2 = self.create_RX90_pulse(
                        qubit, clock[qubit], relative_phase=sequence.virtual_z_phases[qubit] - np.pi
                    )
                    # apply RX(-pi/2)
                    sequence.add(RX90_pulse_2)
                    clock[qubit] += RX90_pulse_2.duration
                    # apply RZ(phi)
                    sequence.virtual_z_phases[qubit] += phi

                elif isinstance(gate, gates.CZ):
                    control = max(gate.qubits)
                    target = min(gate.qubits)

                    pair = f"{control}-{target}"
                    if pair not in self.native_two_qubit_gates:
                        raise_error(ValueError, f"CZ gate between {control} and {target} is not available.")

                    cz_pulse = self.create_CZ_pulse(control, target, clock[control])
                    pulse_kwargs = self.native_two_qubit_gates[pair]["CZ"]
                    sequence.add(cz_pulse)
                    clock[control] += cz_pulse.duration
                    clock[target] += cz_pulse.duration
                    sequence.virtual_z_phases[control] += pulse_kwargs["phase_control"]
                    sequence.virtual_z_phases[target] += pulse_kwargs["phase_target"]

                elif isinstance(gate, gates.M):
                    # Add measurement pulse
                    mz_pulses = []
                    for qubit in gate.target_qubits:
                        MZ_pulse = self.create_MZ_pulse(qubit, clock[qubit])
                        sequence.add(MZ_pulse)  # append_at_end_of_channel?
                        mz_pulses.append(MZ_pulse.serial)
                    gate.pulses = tuple(mz_pulses)

                else:  # pragma: no cover
                    raise_error(
                        NotImplementedError,
                        f"Transpilation of {gate.__class__.__name__} gate has not been implemented yet.",
                    )
                added.add(gate)

        return sequence

    @abstractmethod
    def execute_pulse_sequence(self, sequence, nshots=None):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
        """

    def __call__(self, sequence, nshots=None):
        return self.execute_pulse_sequence(sequence, nshots)

    def sweep(self, sequence, *sweepers, nshots=1024, average=True):
        """Executes a pulse sequence for different values of sweeped parameters.

        Useful for performing chip characterization.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            sweepers (:class:`qibolab.sweeper.Sweeper`): Sweeper objects that specify which
                parameters are being sweeped.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.
            average (bool): If ``True`` the IQ results of individual shots are averaged
                on hardware.

        Returns:
            Readout results acquired by after execution.
        """
        raise_error(NotImplementedError, f"Platform {self.name} does not support sweeping.")

    # TODO: Maybe create a dataclass for native gates
    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"] / 2.0
        qd_shape = pulse_kwargs["shape"]
        qd_channel = self.qubits[qubit].drive.name
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        qd_channel = self.qubits[qubit].drive.name
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_MZ_pulse(self, qubit, start):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["MZ"]
        ro_duration = pulse_kwargs["duration"]
        ro_frequency = pulse_kwargs["frequency"]
        ro_amplitude = pulse_kwargs["amplitude"]
        ro_shape = pulse_kwargs["shape"]
        ro_channel = self.qubits[qubit].readout.name
        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        qd_channel = self.qubits[qubit].drive.name
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

        qd_channel = self.qubits[qubit].drive.name
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

        qd_channel = self.qubits[qubit].drive.name
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_CZ_pulse(self, control, target, start):
        pulse_kwargs = self.native_two_qubit_gates[f"{control}-{target}"]["CZ"]
        return FluxPulse(
            start,
            duration=pulse_kwargs["duration"],
            amplitude=pulse_kwargs["amplitude"],
            shape=pulse_kwargs["shape"],
            channel=self.qubits[control].flux.name,
            qubit=control,
        )

    def set_attenuation(self, qubit, att):  # pragma: no cover
        """Set attenuation value. Usefeul for calibration routines such as punchout.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            att (int): new value of the attenuation (dB).
        Returns:
            None
        """
        raise_error(NotImplementedError)

    def set_gain(self, qubit, gain):  # pragma: no cover
        """Set gain value. Usefeul for calibration routines such as Rabis.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            gain (int): new value of the gain (dimensionless).
        Returns:
            None
        """
        raise_error(NotImplementedError)

    def set_current(self, qubit, curr):  # pragma: no cover
        """Set current value. Usefeul for calibration routines involving flux.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            curr (int): new value of the current (A).
        Returns:
            None
        """
        raise_error(NotImplementedError)
