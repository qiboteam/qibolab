# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import yaml
from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit

from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)


class Qubit:
    """Data structure that holds information about all instruments controlling a qubit.

    Args:
        index (int): Qubit index (read from runcard).
        channels (list): List of channels (int) associated to the qubit (read from runcard).
        instruments (dict): Dictionary containing the instrument objects created in ``AbstractPlatform``.

    Attributes:
        instruments (list) (length 3)
        ports (list) (length 3)
    """

    def __init__(self, index, settings):
        self.index = index
        self.channels = settings["qubit_channel_map"][index]
        self.ro_channel, self.qd_channel, self.qf_channel = self.channels

        # Generate qubit_instrument_map from qubit_channel_map and the instruments' channel_port_maps
        self.instruments = [None, None, None]
        for name, inst_settings in settings["instruments"].items():
            for channel in inst_settings.get("channel_port_map", []):
                if channel in self.channels:
                    self.instruments[self.channels.index(channel)] = name
            for channel in inst_settings.get("s4g_modules", []):
                if channel in self.channels:
                    self.instruments[self.channels.index(channel)] = name

        self.native_one_qubit = settings["native_gates"]["single_qubit"][index]
        # TODO: Think how to implement two qubit gates

    def get_native_gate(self, name, start, relative_phase):
        kwargs = dict(self.native_one_qubit.get(name))
        kwargs.pop("phase")
        kwargs["start"] = start
        kwargs["relative_phase"] = relative_phase
        kwargs["qubit"] = self.index
        channel_type = kwargs.pop("type")
        if channel_type == "ro":
            kwargs["channel"] = self.ro_channel
        else:
            kwargs["channel"] = self.qd_channel
        return kwargs


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
        self.is_connected = False
        # Load platform settings
        with open(runcard, "r") as file:
            self.settings = yaml.safe_load(file)

        self.nqubits = self.settings["nqubits"]
        if self.nqubits == 1:
            self.resonator_type = "3D"
        else:
            self.resonator_type = "2D"
        self.hardware_avg = self.settings["settings"]["hardware_avg"]
        self.sampling_rate = self.settings["settings"]["sampling_rate"]
        self.repetition_duration = self.settings["settings"]["repetition_duration"]

        self.topology = self.settings["topology"]
        self.channels = self.settings["channels"]

        # Load Characterization settings
        self.characterization = self.settings["characterization"]
        # Load Native Gates
        self.native_gates = self.settings["native_gates"]

        self.instruments = {}
        # Instantiate instruments
        for name, inst_settings in self.settings["instruments"].items():
            lib = inst_settings["lib"]
            i_class = inst_settings["class"]
            address = inst_settings["address"]
            from importlib import import_module

            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, address)
            self.instruments[name] = instance

        # Instantiate Qubit objects and map them to channels and instruments
        self.qubits = [Qubit(q, self.settings) for q in self.settings["qubits"]]

    def __repr__(self):
        return self.name

    def __getstate__(self):
        return {
            "name": self.name,
            "runcard": self.runcard,
            "settings": self.settings,
            "is_connected": self.is_connected,
        }

    def __setstate__(self, data):
        self.name = data.get("name")
        self.runcard = data.get("runcard")
        self.settings = data.get("settings")
        self.is_connected = data.get("is_connected")

    def _check_connected(self):
        if not self.is_connected:
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        with open(self.runcard, "r") as file:
            self.settings = yaml.safe_load(file)
        self.setup()

    @abstractmethod
    def run_calibration(self, show_plots=False):  # pragma: no cover
        """Executes calibration routines and updates the settings yml file"""
        raise NotImplementedError

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            try:
                for name in self.instruments:
                    log.info(f"Connecting to {self.name} instrument {name}.")
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                raise_error(
                    RuntimeError,
                    "Cannot establish connection to " f"{self.name} instruments. " f"Error captured: '{exception}'",
                )

    def setup(self):
        if not self.is_connected:
            raise_error(
                RuntimeError,
                "There is no connection to the instruments, the setup cannot be completed",
            )

        for name, instrument in self.instruments.items():
            # Set up every with the platform settings and the instrument settings
            instrument.setup(
                **self.settings["settings"],
                **self.settings["instruments"][name]["settings"],
            )

        # Generate ro_channel[qubit], qd_channel[qubit], qf_channel[qubit], qrm[qubit], qcm[qubit], lo_qrm[qubit], lo_qcm[qubit]
        self.ro_channel = {qubit.ro_channel for qubit in self.qubits}
        self.qd_channel = {qubit.qd_channel for qubit in self.qubits}
        self.qf_channel = {qubit.qf_channel for qubit in self.qubits}
        self.qrm = {}
        self.qcm = {}
        self.qbm = {}
        self.ro_port = {}
        self.qd_port = {}
        self.qf_port = {}
        for qubit in self.qubits:
            if not qubit.instruments[0] is None:
                self.qrm[qubit] = self.instruments[qubit.instruments[0]]
                self.ro_port[qubit] = self.qrm[qubit].ports[self.qrm[qubit].channel_port_map[qubit.channels[0]]]
            if not qubit.instruments[1] is None:
                self.qcm[qubit] = self.instruments[qubit.instruments[1]]
                self.qd_port[qubit] = self.qcm[qubit].ports[self.qcm[qubit].channel_port_map[qubit.channels[1]]]
            if not qubit.instruments[2] is None:
                self.qbm[qubit] = self.instruments[qubit.instruments[2]]
                self.qf_port[qubit] = self.qbm[qubit].dacs[qubit.channels[2]]

    def start(self):
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.start()

    def stop(self):
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.stop()

    def disconnect(self):
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.disconnect()
            self.is_connected = False

    # TRANSPILATION
    def transpile(self, circuit: Circuit):
        """Transforms a circuit to pulse sequence.

        Args:
            circuit (qibo.models.Circuit): Qibo circuit that respects the platform's
                connectivity and native gates.

        Returns:
            sequence (qibolab.pulses.PulseSequence): Pulse sequence that implements the
                circuit on the qubit.
        """
        import numpy as np

        from qibolab.transpilers import can_execute, transpile

        if not can_execute(circuit):
            circuit, hardware_qubits = transpile(circuit)

        sequence = PulseSequence()
        sequence.virtual_z_phases = {}
        for qubit in range(circuit.nqubits):
            sequence.virtual_z_phases[qubit] = 0

        for gate in circuit.queue:

            if isinstance(gate, gates.I):
                pass

            # if isinstance(gate, gates.I):
            #     qubit = gate.target_qubits[0]
            #     pulse = self.create_RX_pulse(qubit)
            #     pulse.amplitude = 0
            #     sequence.append_at_end_of_channel(pulse, pulse.copy())

            # elif isinstance(gate, gates.X):
            #     qubit = gate.target_qubits[0]
            #     pulse = self.create_RX90_pulse(qubit, relative_phase=sequence.virtual_z_phases[qubit])
            #     sequence.append_at_end_of_channel(pulse, pulse.copy())

            # elif isinstance(gate, gates.Y):
            #     qubit = gate.target_qubits[0]
            #     pulse = self.create_RX90_pulse(qubit, relative_phase=sequence.virtual_z_phases[qubit] + np.pi / 2)
            #     sequence.append_at_end_of_channel(pulse, pulse.copy())

            # elif isinstance(gate, gates.RX):
            #     qubit = gate.target_qubits[0]
            #     rotation_angle = gate.parameters[0] % (2 * np.pi)
            #     relative_phase = sequence.virtual_z_phases[qubit]
            #     if rotation_angle > np.pi:
            #         rotation_angle = 2 * np.pi - rotation_angle
            #         relative_phase += np.pi
            #     pulse = self.create_RX90_pulse(qubit, relative_phase=relative_phase)
            #     pulse.amplitude *= rotation_angle / np.pi
            #     sequence.append_at_end_of_channel(pulse, pulse.copy())

            # elif isinstance(gate, gates.RY):
            #     qubit = gate.target_qubits[0]
            #     rotation_angle = gate.parameters[0] % (2 * np.pi)
            #     relative_phase = sequence.virtual_z_phases[qubit] + np.pi / 2
            #     if rotation_angle > np.pi:
            #         rotation_angle = 2 * np.pi - rotation_angle
            #         relative_phase += np.pi
            #     pulse = self.create_RX90_pulse(qubit, relative_phase=relative_phase)
            #     pulse.amplitude *= rotation_angle / np.pi
            #     sequence.append_at_end_of_channel(pulse, pulse.copy())

            elif isinstance(gate, gates.Z):
                qubit = gate.target_qubits[0]
                sequence.virtual_z_phases[qubit] += np.pi

            elif isinstance(gate, gates.RZ):
                qubit = gate.target_qubits[0]
                sequence.virtual_z_phases[qubit] += gate.parameters[0]

            elif isinstance(gate, gates.M):
                # Add measurement pulse
                measurement_start = sequence.finish
                for qubit in circuit.measurement_gate.target_qubits:
                    MZ_pulse = self.create_MZ_pulse(qubit, measurement_start)
                    sequence.add(MZ_pulse)  # append_at_end_of_channel?

            elif isinstance(gate, gates.U3):
                qubit = gate.target_qubits[0]
                # Transform gate to U3 and add pi/2-pulses
                theta, phi, lam = gate.parameters
                # apply RZ(lam)
                sequence.virtual_z_phases[qubit] += lam
                # Fetch pi/2 pulse from calibration
                RX90_pulse_1 = self.create_RX90_pulse(
                    qubit, sequence.finish, relative_phase=sequence.virtual_z_phases[qubit]
                )
                # apply RX(pi/2)
                sequence.append_at_end_of_channel(RX90_pulse_1)
                # apply RZ(theta)
                sequence.virtual_z_phases[qubit] += theta
                # Fetch pi/2 pulse from calibration
                RX90_pulse_2 = self.create_RX90_pulse(
                    qubit, sequence.finish, relative_phase=sequence.virtual_z_phases[qubit] - np.pi
                )
                # apply RX(-pi/2)
                sequence.append_at_end_of_channel(RX90_pulse_2)
                # apply RZ(phi)
                sequence.virtual_z_phases[qubit] += phi

            else:
                raise_error(
                    NotImplementedError,
                    f"Transpilation of {gate.__class__.__name__} gate has not been implemented yet.",
                )

        # Finally add measurement gates
        if circuit.measurement_gate:
            measurement_start = sequence.finish
            for qubit in circuit.measurement_gate.target_qubits:
                MZ_pulse = self.create_MZ_pulse(qubit, measurement_start)
                sequence.add(MZ_pulse)
            # FIXME: is there any reason not to include measurement gates as part of the circuit queue?
            # This workaround adds them at the end, but would it not be desirable to be able to insert
            # measurement gates in the middle of circuits? (Alvaro)
        return sequence

    @abstractmethod
    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
        """
        raise NotImplementedError

    def __call__(self, sequence, nshots=None):
        return self.execute_pulse_sequence(sequence, nshots)

    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        kwargs = self.qubits[qubit].get_native_gate("RX", start, relative_phase)
        kwargs["amplitude"] /= 2.0
        return Pulse(**kwargs)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        kwargs = self.qubits[qubit].get_native_gate("RX", start, relative_phase)
        return Pulse(**kwargs)

    def create_MZ_pulse(self, qubit, start):
        kwargs = self.qubits[qubit].get_native_gate("MZ", start, relative_phase=0)
        return ReadoutPulse(**kwargs)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        kwargs = self.qubits[qubit].get_native_gate("RX", start, relative_phase)
        return Pulse(**kwargs)

    def create_qubit_readout_pulse(self, qubit, start):
        kwargs = self.qubits[qubit].get_native_gate("MZ", start, relative_phase=0)
        return ReadoutPulse(**kwargs)

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi/2 pulse with drag shape
        kwargs = self.qubits[qubit].get_native_gate("RX", start, relative_phase)
        kwargs["amplitude"] /= 2
        if beta != None:
            qd_shape = f"Drag(5,{beta})"
        return Pulse(**kwargs)

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi pulse with drag shape
        kwargs = self.qubits[qubit].get_native_gate("RX", start, relative_phase)
        if beta != None:
            qd_shape = f"Drag(5,{beta})"
        return Pulse(**kwargs)
