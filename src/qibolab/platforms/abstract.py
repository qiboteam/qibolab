# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import yaml
from qibo import gates
from qibo.config import log, raise_error

from qibolab.pulses import (
    Drag,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)


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

        self.qubits = self.settings["qubits"]
        self.topology = self.settings["topology"]
        self.channels = self.settings["channels"]
        self.qubit_channel_map = self.settings["qubit_channel_map"]

        # Load Characterization settings
        self.characterization = self.settings["characterization"]
        # Load Native Gates
        self.native_gates = self.settings["native_gates"]

        self.instruments = {}
        # Instantiate instruments
        for name in self.settings["instruments"]:
            lib = self.settings["instruments"][name]["lib"]
            i_class = self.settings["instruments"][name]["class"]
            address = self.settings["instruments"][name]["address"]
            from importlib import import_module

            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, address)
            self.instruments[name] = instance

        # Generate qubit_instrument_map from qubit_channel_map and the instruments' channel_port_maps
        self.qubit_instrument_map = {}
        for qubit in self.qubit_channel_map:
            self.qubit_instrument_map[qubit] = [None, None, None]
            for name in self.instruments:
                if "channel_port_map" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["channel_port_map"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name
                if "s4g_modules" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["s4g_modules"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name

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

        for name in self.instruments:
            # Set up every with the platform settings and the instrument settings
            self.instruments[name].setup(
                **self.settings["settings"],
                **self.settings["instruments"][name]["settings"],
            )

        # Generate ro_channel[qubit], qd_channel[qubit], qf_channel[qubit], qrm[qubit], qcm[qubit], lo_qrm[qubit], lo_qcm[qubit]
        self.ro_channel = {}
        self.qd_channel = {}
        self.qf_channel = {}
        self.qrm = {}
        self.qcm = {}
        self.qbm = {}
        self.ro_port = {}
        self.qd_port = {}
        self.qf_port = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][2]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit] = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.ro_port[qubit] = self.qrm[qubit].ports[
                    self.qrm[qubit].channel_port_map[self.qubit_channel_map[qubit][0]]
                ]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qcm[qubit] = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.qd_port[qubit] = self.qcm[qubit].ports[
                    self.qcm[qubit].channel_port_map[self.qubit_channel_map[qubit][1]]
                ]
            if not self.qubit_instrument_map[qubit][2] is None:
                self.qbm[qubit] = self.instruments[self.qubit_instrument_map[qubit][2]]
                self.qf_port[qubit] = self.qbm[qubit].dacs[self.qubit_channel_map[qubit][2]]

    def start(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].start()

    def stop(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].stop()

    def disconnect(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].disconnect()
            self.is_connected = False

    # TRANSPILATION
    from qibo.models import Circuit

    def transpile(self, circuit: Circuit):  # (self, circuit: qibo.core.circuit.Circuit) -> PulseSequence
        import numpy as np

        from qibolab.transpilers.transpile import transpile

        native_circuit, hardware_qubits = transpile(circuit, fuse_one_qubit=False)
        sequence = PulseSequence()
        sequence.virtual_z_phases = {}
        for qubit in range(circuit.nqubits):
            sequence.virtual_z_phases[qubit] = 0

        for gate in native_circuit.queue:

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
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"] / 2
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.settings["qubit_channel_map"][qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.settings["qubit_channel_map"][qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_MZ_pulse(self, qubit, start):
        ro_duration = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.settings["qubit_channel_map"][qubit][0]
        from qibolab.pulses import ReadoutPulse

        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.settings["qubit_channel_map"][qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_qubit_readout_pulse(self, qubit, start):
        ro_duration = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.settings["qubit_channel_map"][qubit][0]
        from qibolab.pulses import ReadoutPulse

        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi/2 pulse with drag shape
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"] / 2
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.settings["qubit_channel_map"][qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi pulse with drag shape
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.settings["qubit_channel_map"][qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)
