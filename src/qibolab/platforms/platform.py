import collections

import numpy as np
import yaml
from qibo import gates
from qibo.config import log, raise_error

from qibolab.platforms.utils import Channel, Qubit
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, ReadoutPulse, Rectangular
from qibolab.transpilers import can_execute, transpile


class Platform:
    def __init__(self, design, runcard):
        self.design = design
        self.runcard = runcard

        self.nqubits = None
        self.resonator_type = None
        self.topology = None
        self.qubits = []
        self.sampling_rate = None
        self.options = None

        self.native_single_qubit_gates = {}
        self.native_two_qubit_gates = {}
        self.two_qubit_natives = set()
        # Load platform settings
        self.reload_settings()

    def reload_settings(self):
        """Reloads the runcard and re-setups the connected instruments using the new values."""
        # TODO: Maybe runcard loading can be improved
        with open(self.runcard) as file:
            settings = yaml.safe_load(file)

        self.nqubits = settings["nqubits"]
        self.resonator_type = "3D" if self.nqubits == 1 else "2D"
        self.topology = settings["topology"]

        self.options = settings["options"]
        self.sampling_rate = self.options["sampling_rate"]

        # TODO: Create better data structures for native gates
        native_gates = settings["native_gates"]
        self.native_single_qubit_gates = native_gates["single_qubit"]
        if "two_qubit" in native_gates:
            self.native_two_qubit_gates = native_gates["two_qubit"]
            for gates in native_gates["two_qubit"].values():
                self.two_qubit_natives |= set(gates.keys())
        else:
            # dummy value to avoid transpiler failure for single qubit devices
            self.two_qubit_natives = ["CZ"]

        # Create list of qubit objects
        characterization = settings["characterization"]
        self.qubits = []
        for q, channel_names in settings["qubit_channel_map"].items():
            channels = (Channel(name) for name in channel_names)
            self.qubits.append(Qubit(q, characterization[q], *channels))

    def is_connected(self):
        return self.design.is_connected

    def connect(self):
        self.design.connect()

    def setup(self):
        self.design.setup(self.qubits, **self.options)

    def start(self):
        self.design.start()

    def stop(self):
        self.design.stop()

    def disconnect(self):
        self.design.disconnect()

    def transpile(self, circuit):
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
        virtual_z_phases = collections.defaultdict(int)
        clock = collections.defaultdict(int)
        # keep track of gates that were already added to avoid adding them twice
        added = set()
        for moment in circuit.queue.moments:
            for gate in moment:

                if isinstance(gate, gates.I) or gate is None or gate in added:
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
                    RX90_pulse_1 = self.create_RX90_pulse(qubit, clock[qubit], relative_phase=virtual_z_phases[qubit])
                    # apply RX(pi/2)
                    sequence.add(RX90_pulse_1)
                    clock[qubit] += RX90_pulse_1.duration
                    # apply RZ(theta)
                    virtual_z_phases[qubit] += theta
                    # Fetch pi/2 pulse from calibration
                    RX90_pulse_2 = self.create_RX90_pulse(
                        qubit, clock[qubit], relative_phase=virtual_z_phases[qubit] - np.pi
                    )
                    # apply RX(-pi/2)
                    sequence.add(RX90_pulse_2)
                    clock[qubit] += RX90_pulse_2.duration
                    # apply RZ(phi)
                    virtual_z_phases[qubit] += phi

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
                    virtual_z_phases[control] += pulse_kwargs["phase_control"]
                    virtual_z_phases[target] += pulse_kwargs["phase_target"]

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

    def sweep(self, sequence, *sweepers, nshots=1024, average=True):
        return self.design.sweep(self.qubits, sequence, *sweepers, nshots=nshots, average=average)

    def execute_pulse_sequence(self, sequence, nshots=1024):
        """Play an arbitrary pulse sequence and retrieve feedback.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Sequence of pulses to play.
            nshots (int): Number of hardware repetitions of the execution.

        Returns:
            TODO: Decide a unified way to return results.
        """
        return self.design.play(self.qubits, sequence, nshots)

    # TODO: Maybe channel should be removed from pulses
    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"] / 2.0
        qd_shape = pulse_kwargs["shape"]
        qd_channel = str(self.qubits[qubit].drive)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        qd_channel = str(self.qubits[qubit].drive)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_MZ_pulse(self, qubit, start):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["MZ"]
        ro_duration = pulse_kwargs["duration"]
        ro_frequency = pulse_kwargs["frequency"]
        ro_amplitude = pulse_kwargs["amplitude"]
        ro_shape = pulse_kwargs["shape"]
        ro_channel = str(self.qubits[qubit].readout)
        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        qd_shape = pulse_kwargs["shape"]
        qd_channel = str(self.qubits[qubit].drive)
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
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = str(self.qubits[qubit].drive)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi pulse with drag shape
        pulse_kwargs = self.native_single_qubit_gates[qubit]["RX"]
        qd_duration = pulse_kwargs["duration"]
        qd_frequency = pulse_kwargs["frequency"]
        qd_amplitude = pulse_kwargs["amplitude"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = str(self.qubits[qubit].drive)
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_CZ_pulse(self, control, target, start):
        pulse_kwargs = self.native_two_qubit_gates[f"{control}-{target}"]["CZ"]
        return FluxPulse(
            start,
            duration=pulse_kwargs["duration"],
            amplitude=pulse_kwargs["amplitude"],
            shape=pulse_kwargs["shape"],
            channel=str(self.qubits[control].flux),
            qubit=control,
        )
