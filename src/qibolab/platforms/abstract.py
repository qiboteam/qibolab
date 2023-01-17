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
        with open(runcard) as file:
            self.settings = yaml.safe_load(file)

        self.nqubits = self.settings["nqubits"]
        if self.nqubits == 1:
            self.resonator_type = "3D"
        else:
            self.resonator_type = "2D"

        self.qubits = self.settings["qubits"]
        self.topology = self.settings["topology"]
        self.channels = self.settings["channels"]
        self.qubit_channel_map = self.settings["qubit_channel_map"]

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
            self.qubit_instrument_map[qubit] = [None, None, None, None]
            for name in self.instruments:
                if "channel_port_map" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["channel_port_map"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name
                if "s4g_modules" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["s4g_modules"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name

        self.reload_settings()

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
        if not self.is_connected:  # pragma: no cover
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        with open(self.runcard) as file:
            self.settings = yaml.safe_load(file)

        self.hardware_avg = self.settings["settings"]["hardware_avg"]
        self.sampling_rate = self.settings["settings"]["sampling_rate"]
        self.repetition_duration = self.settings["settings"]["repetition_duration"]

        # Load Characterization settings
        self.characterization = self.settings["characterization"]
        # Load single qubit Native Gates
        self.native_gates = self.settings["native_gates"]
        self.two_qubit_natives = set()
        # Load two qubit Native Gates, if multiqubit platform
        if "two_qubit" in self.native_gates.keys():
            for pairs, gates in self.native_gates["two_qubit"].items():
                self.two_qubit_natives |= set(gates.keys())
        else:
            self.two_qubit_natives = ["CZ"]

        if self.is_connected:
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
        self.ro_channel = {}  # readout
        self.qd_channel = {}  # qubit drive
        self.qf_channel = {}  # qubit flux
        self.qb_channel = {}  # qubit flux biassing
        self.qrm = {}  # qubit readout module
        self.qdm = {}  # qubit drive module
        self.qfm = {}  # qubit flux module
        self.qbm = {}  # qubit flux biassing module
        self.ro_port = {}
        self.qd_port = {}
        self.qf_port = {}
        self.qb_port = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qb_channel[qubit] = self.qubit_channel_map[qubit][2]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][3]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit] = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.ro_port[qubit] = self.qrm[qubit].ports[
                    self.qrm[qubit].channel_port_map[self.qubit_channel_map[qubit][0]]
                ]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qdm[qubit] = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.qd_port[qubit] = self.qdm[qubit].ports[
                    self.qdm[qubit].channel_port_map[self.qubit_channel_map[qubit][1]]
                ]
            if not self.qubit_instrument_map[qubit][2] is None:
                self.qfm[qubit] = self.instruments[self.qubit_instrument_map[qubit][2]]
                self.qf_port[qubit] = self.qfm[qubit].ports[
                    self.qfm[qubit].channel_port_map[self.qubit_channel_map[qubit][2]]
                ]
            if not self.qubit_instrument_map[qubit][3] is None:
                self.qbm[qubit] = self.instruments[self.qubit_instrument_map[qubit][3]]
                self.qb_port[qubit] = self.qbm[qubit].dacs[self.qubit_channel_map[qubit][3]]

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

        if not can_execute(circuit, self.two_qubit_natives):
            circuit, hardware_qubits = transpile(circuit, self.two_qubit_natives)

        sequence = PulseSequence()
        sequence.virtual_z_phases = {}
        for qubit in range(circuit.nqubits):
            sequence.virtual_z_phases[qubit] = 0

        for gate in circuit.queue:

            if isinstance(gate, gates.I):
                pass

            elif isinstance(gate, gates.Z):
                qubit = gate.target_qubits[0]
                sequence.virtual_z_phases[qubit] += np.pi

            elif isinstance(gate, gates.RZ):
                qubit = gate.target_qubits[0]
                sequence.virtual_z_phases[qubit] += gate.parameters[0]

            elif isinstance(gate, gates.U3):
                qubit = gate.target_qubits[0]
                finish = sequence.get_qubit_finish_time(qubit)
                # Transform gate to U3 and add pi/2-pulses
                theta, phi, lam = gate.parameters
                # apply RZ(lam)
                sequence.virtual_z_phases[qubit] += lam
                # Fetch pi/2 pulse from calibration
                RX90_pulse_1 = self.create_RX90_pulse(qubit, finish, relative_phase=sequence.virtual_z_phases[qubit])
                # apply RX(pi/2)
                sequence.add(RX90_pulse_1)
                # apply RZ(theta)
                sequence.virtual_z_phases[qubit] += theta
                # Fetch pi/2 pulse from calibration
                RX90_pulse_2 = self.create_RX90_pulse(
                    qubit, RX90_pulse_1.finish, relative_phase=sequence.virtual_z_phases[qubit] - np.pi
                )
                # apply RX(-pi/2)
                sequence.add(RX90_pulse_2)
                # apply RZ(phi)
                sequence.virtual_z_phases[qubit] += phi

            elif isinstance(gate, gates.M):
                # Add measurement pulse
                measurement_start = sequence.finish
                mz_pulses = []
                for qubit in gate.target_qubits:
                    MZ_pulse = self.create_MZ_pulse(qubit, measurement_start)
                    sequence.add(MZ_pulse)  # append_at_end_of_channel?
                    mz_pulses.append(MZ_pulse.serial)
                gate.pulses = tuple(mz_pulses)
                
            elif isinstance(gate, gates.CZ):
                # Get channel pulses for both qubits and get the latest finish time
                finish = 0
                for qubit in gate.qubits:
                    if sequence.get_qubit_finish_time(qubit) > finish:
                        finish = sequence.get_qubit_finish_time(qubit)
                cz_sequence = self.create_CZ_pulse(gate.qubits, finish)
                sequence.add(*cz_sequence.pulses)
                for key in cz_sequence.virtual_z_phases:
                    sequence.virtual_z_phases[key] += cz_sequence.virtual_z_phases[key]

            else:  # pragma: no cover
                raise_error(
                    NotImplementedError,
                    f"Transpilation of {gate.__class__.__name__} gate has not been implemented yet.",
                )

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

    def create_CZ_pulse(self, qubits, start=0):
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
        sequence.virtual_z_phases = {}

        for pulse_settings in pulse_sequence_settings:
            if pulse_settings["type"] == "qf":
                qf_duration = pulse_settings["duration"]
                qf_amplitude = pulse_settings["amplitude"]
                qf_shape = pulse_settings["shape"]
                qubits = pulse_settings["qubit"]
                if self.characterization["single_qubit"][qubits[0]]["qubit_freq"] > self.characterization["single_qubit"][qubits[1]]["qubit_freq"]:
                    qf_channel = self.settings["qubit_channel_map"][qubits[0]][2]
                else:
                    qf_channel = self.settings["qubit_channel_map"][qubits[1]][2]
                sequence.add(
                    FluxPulse(
                        start + pulse_settings["relative_start"], qf_duration, qf_amplitude, qf_shape, qf_channel, qubits
                    )
                )
            elif pulse_settings["type"] == "virtual_z":
                if not pulse_settings["qubit"] in sequence.virtual_z_phases:
                    sequence.virtual_z_phases[pulse_settings["qubit"]] = 0
                else:
                    sequence.virtual_z_phases[pulse_settings["qubit"]] += pulse_settings["phase"]
        return sequence

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
