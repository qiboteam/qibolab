import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.utils import Channel, Qubit
from qibolab.pulses import Pulse, ReadoutPulse


class Platform:
    def __init__(self, design, runcard):
        self.design = design
        self.runcard = runcard

        self.native_single_qubit_gates = None
        self.native_two_qubit_gates = None
        self.nqubits = None
        self.resonator_type = None
        self.topology = None
        self.qubits = []
        self.sampling_rate = None
        # time we are waiting between each shot
        self.relaxation_time = None

        # Load platform settings
        self.settings = None
        self.reload_settings()

    def reload_settings(self):
        """Reloads the runcard and re-setups the connected instruments using the new values."""
        # TODO: Maybe runcard loading can be improved
        with open(self.runcard) as file:
            self.settings = yaml.safe_load(file)

        self.nqubits = self.settings["nqubits"]
        self.resonator_type = "3D" if self.nqubits == 1 else "2D"
        self.topology = self.settings["topology"]

        self.sampling_rate = self.settings["options"]["sampling_rate"]
        self.relaxation_time = self.settings["options"]["relaxation_time"]

        self.native_single_qubit_gates = self.settings["native_gates"].get("single_qubit")
        self.native_two_qubit_gates = self.settings["native_gates"].get("two_qubit")

        # Create list of qubit objects
        characterization = self.settings["characterization"]["single_qubit"]
        self.qubits = []
        for q, channel_names in self.settings["qubit_channel_map"].items():
            channels = (Channel(name) for name in channel_names)
            self.qubits.append(Qubit(q, characterization[q], *channels))

    def connect(self):
        self.design.connect()

    def setup(self):
        self.design.setup(self.qubits, self.relaxation_time)

    def start(self):
        self.design.start()

    def stop(self):
        self.design.stop()

    def disconnect(self):
        self.design.disconnect()

    def sweep(self, sequence, *sweepers, nshots=1024):
        return self.design.sweep(self.qubits, sequence, *sweepers, nshots=nshots)

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
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"] / 2
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.settings["qubit_channel_map"][qubit][1]
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.settings["qubit_channel_map"][qubit][2]
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_MZ_pulse(self, qubit, start):
        ro_duration = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.settings["qubit_channel_map"][qubit][0]
        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.settings["qubit_channel_map"][qubit][2]
        return Pulse(start, duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_qubit_readout_pulse(self, qubit, start):
        ro_duration = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.settings["qubit_channel_map"][qubit][0]
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

        qd_channel = self.settings["qubit_channel_map"][qubit][2]
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi pulse with drag shape
        qd_duration = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.settings["qubit_channel_map"][qubit][2]
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)
