import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import yaml
from qibo.config import log, raise_error

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.abstract import Controller, Instrument
from qibolab.native import NativeType, SingleQubitNatives, TwoQubitNatives
from qibolab.qubits import Qubit, QubitId, QubitPair


class Platform:
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform.
        runcard (str): path to the yaml file containing the platform setup.
        instruments:
        channels:
    """

    def __init__(self, name, runcard, instruments, channels):
        log.info(f"Loading platform {name}")

        self.name = name
        self.runcard = runcard
        self.instruments: List[Instrument] = instruments
        self.channels: ChannelMap = channels

        self.qubits: Dict[QubitId, Qubit] = {}
        self.pairs: Dict[Tuple[QubitId, QubitId], QubitPair] = {}

        # Values for the following are set from the runcard in ``reload_settings``
        self.settings = None
        self.is_connected = False

        self.nqubits = None
        self.resonator_type = None
        self.topology = None

        self.nshots = None
        self.relaxation_time = None
        self.sampling_rate = None

        # TODO: Remove this (needed for the multiqubit platform)
        self.native_gates = {}
        self.two_qubit_native_types = NativeType(0)
        # Load platform settings
        self.reload_settings()

    def __repr__(self):
        return self.name

    def _check_connected(self):
        if not self.is_connected:  # pragma: no cover
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        # TODO: Remove ``self.settings``
        if self.settings == None:
            # Load initial configuration
            if isinstance(self.runcard, dict):
                settings = self.settings = self.runcard
            else:
                with open(self.runcard) as file:
                    settings = self.settings = yaml.safe_load(file)
        else:
            # Load current configuration
            settings = self.settings

        self.nqubits = settings["nqubits"]
        if "resonator_type" in self.settings:
            self.resonator_type = self.settings["resonator_type"]
        else:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

        self.relaxation_time = settings["settings"]["relaxation_time"]
        self.nshots = settings["settings"]["nshots"]
        self.sampling_rate = settings["settings"]["sampling_rate"]
        self.native_gates = settings["native_gates"]

        # Load characterization settings and create ``Qubit`` and ``Channel`` objects
        for q in settings["qubits"]:
            if q in self.qubits:
                for name, value in settings["characterization"]["single_qubit"][q].items():
                    setattr(self.qubits[q], name, value)
            else:
                self.qubits[q] = qubit = Qubit(q, **settings["characterization"]["single_qubit"][q])
                # register channels to qubits when we are using the old format
                # needed for ``NativeGates`` to work
                if "qubit_channel_map" in self.settings:
                    ro, qd, qf, _ = self.settings["qubit_channel_map"][q]
                    if ro is not None:
                        qubit.readout = Channel(ro)
                    if qd is not None:
                        qubit.drive = Channel(qd)
                    if qf is not None:
                        qubit.flux = Channel(qf)
                # register single qubit native gates to Qubit objects
                if q in self.native_gates["single_qubit"]:
                    qubit.native_gates = SingleQubitNatives.from_dict(qubit, self.native_gates["single_qubit"][q])

        for pair in settings["topology"]:
            pair = tuple(sorted(pair))
            if pair not in self.pairs:
                self.pairs[pair] = QubitPair(self.qubits[pair[0]], self.qubits[pair[1]])
        # Load native two-qubit gates
        if "two_qubit" in self.native_gates:
            for pair, gatedict in self.native_gates["two_qubit"].items():
                pair = tuple(sorted(int(q) if q.isdigit() else q for q in pair.split("-")))
                self.pairs[pair].native_gates = TwoQubitNatives.from_dict(self.qubits, gatedict)
                self.two_qubit_native_types |= self.pairs[pair].native_gates.types
        else:
            # dummy value to avoid transpiler failure for single qubit devices
            self.two_qubit_native_types = NativeType.CZ

        if self.topology is None:
            self.topology = nx.Graph()
            self.topology.add_nodes_from(self.qubits.keys())
            self.topology.add_edges_from([(pair.qubit1.name, pair.qubit2.name) for pair in self.pairs.values()])

    def dump(self, path: Path):
        with open(path, "w") as file:
            yaml.dump(self.settings, file, sort_keys=False, indent=4, default_flow_style=None)

    def update(self, updates: dict):
        r"""Updates platform common runcard parameters after calibration actions.

        Args:

            updates (dict): Dictionary containing the parameters to update the runcard. A typical dictionary should be of the following form
                            {`parameter_to_update_in_runcard`:{`qubit0`:`par_value_qubit0`, ..., `qubit_i`:`par_value_qubit_i`, ...}}.
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
                    self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["frequency"] = freq

                    mz = self.qubits[qubit].native_gates.MZ
                    mz.frequency = freq
                    if mz.if_frequency is not None:
                        mz.if_frequency = freq - self.get_lo_readout_frequency(qubit)
                        self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["if_frequency"] = mz.if_frequency

                    self.qubits[qubit].readout_frequency = freq
                    self.settings["characterization"]["single_qubit"][qubit]["readout_frequency"] = freq

                # resonator_punchout_attenuation
                elif par == "readout_attenuation":
                    # TODO: Are we going to save the attenuation somwhere in the native_gates or characterization
                    # in all platforms?
                    True

                # resonator_punchout_attenuation
                elif par == "bare_resonator_frequency":
                    freq = int(value * 1e9)
                    self.qubits[qubit].bare_resonator_frequency = freq
                    self.settings["characterization"]["single_qubit"][qubit]["bare_resonator_frequency"] = freq

                # resonator_spectroscopy_flux / qubit_spectroscopy_flux
                elif par == "sweetspot":
                    sweetspot = float(value)
                    self.qubits[qubit].sweetspot = sweetspot
                    self.settings["characterization"]["single_qubit"][qubit]["sweetspot"] = sweetspot

                # qubit_spectroscopy / qubit_spectroscopy_flux / ramsey
                elif par == "drive_frequency":
                    freq = int(value * 1e9)
                    self.settings["native_gates"]["single_qubit"][qubit]["RX"]["frequency"] = freq

                    self.qubits[qubit].native_gates.RX.frequency = freq
                    self.qubits[qubit].drive_frequency = freq
                    self.settings["characterization"]["single_qubit"][qubit]["drive_frequency"] = freq

                elif "amplitude" in par:
                    amplitude = float(value)
                    # resonator_spectroscopy
                    if par == "readout_amplitude" and not math.isnan(amplitude):
                        self.qubits[qubit].native_gates.MZ.amplitude = amplitude
                        self.settings["native_gates"]["single_qubit"][qubit]["MZ"]["amplitude"] = amplitude

                    # rabi_amplitude / flipping
                    if par == "drive_amplitude" or par == "amplitudes":
                        self.qubits[qubit].native_gates.RX.amplitude = amplitude
                        self.settings["native_gates"]["single_qubit"][qubit]["RX"]["amplitude"] = amplitude
                        self.settings["characterization"]["single_qubit"][qubit]["pi_pulse_amplitude"] = amplitude

                # rabi_duration
                elif par == "drive_length":
                    duration = int(value)
                    self.qubits[qubit].native_gates.RX.duration = duration
                    self.settings["native_gates"]["single_qubit"][qubit]["RX"]["duration"] = duration

                # ramsey
                elif par == "t2":
                    t2 = float(value)
                    self.qubits[qubit].T2 = t2
                    self.settings["characterization"]["single_qubit"][qubit]["T2"] = t2

                # spin_echo
                elif par == "t2_spin_echo":
                    t2_spin_echo = float(value)
                    self.qubits[qubit].T2_spin_echo = t2_spin_echo
                    self.settings["characterization"]["single_qubit"][qubit]["T2_spin_echo"] = t2_spin_echo

                # t1
                elif par == "t1":
                    t1 = float(value)
                    self.qubits[qubit].T1 = t1
                    self.settings["characterization"]["single_qubit"][qubit]["T1"] = t1

                # classification
                elif par == "threshold":
                    threshold = float(value)
                    self.qubits[qubit].thresold = threshold
                    self.settings["characterization"]["single_qubit"][qubit]["threshold"] = threshold

                # classification
                elif par == "iq_angle":
                    iq_angle = float(value)
                    self.qubits[qubit].iq_angle = iq_angle
                    self.settings["characterization"]["single_qubit"][qubit]["iq_angle"] = iq_angle

                # classification
                elif par == "mean_gnd_states":
                    mean_gnd_states = str(value)
                    self.qubits[qubit].mean_gnd_states = mean_gnd_states
                    self.settings["characterization"]["single_qubit"][qubit]["mean_gnd_states"] = mean_gnd_states

                # classification
                elif par == "mean_exc_states":
                    mean_exc_states = str(value)
                    self.qubits[qubit].mean_exc_states = mean_exc_states
                    self.settings["characterization"]["single_qubit"][qubit]["mean_exc_states"] = mean_exc_states

                # drag pulse tunning
                elif "beta" in par:
                    rx = self.qubits[qubit].native_gates.RX
                    shape = rx.shape
                    rel_sigma = re.findall(r"[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+", shape)[0]
                    rx.shape = f"Drag({rel_sigma}, {float(value)})"
                    self.settings["native_gates"]["single_qubit"][qubit]["RX"]["shape"] = rx.shape

                elif "length" in par:  # assume only drive length
                    self.qubits[qubit].native_gates.RX.duration = int(value)

                elif par == "classifiers_hpars":
                    self.qubits[qubit].classifiers_hpars = value
                    self.settings["characterization"]["single_qubit"][qubit]["classifiers_hpars"] = value

                elif par == "readout_attenuation":
                    self.set_attenuation(qubit, value)

                else:
                    raise_error(ValueError, f"Unknown parameter {par} for qubit {qubit}")

        # reload_settings after execute any calibration routine keeping fitted parameters
        self.reload_settings()

    def connect(self):
        """Connect to all instruments."""
        if not self.is_connected:
            for instrument in self.instruments:
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
        """Prepares instruments to execute experiments."""
        for instrument in self.instruments:
            instrument.setup()

    def start(self):
        """Starts all the instruments."""
        if self.is_connected:
            for instrument in self.instruments:
                instrument.start()

    def stop(self):
        """Starts all the instruments."""
        if self.is_connected:
            for instrument in self.instruments:
                instrument.stop()

    def disconnect(self):
        """Disconnects from instruments."""
        if self.is_connected:
            for instrument in self.instruments:
                instrument.disconnect()
        self.is_connected = False

    def execute_pulse_sequence(self, sequence, options, **kwargs):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something

        Returns:
            Readout results acquired by after execution.
        """
        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        result = {}
        for instrument in self.instruments:
            if isinstance(instrument, Controller):
                new_result = instrument.play(self.qubits, sequence, options)
                if isinstance(new_result, dict):
                    result.update(new_result)
                elif new_result is not None:
                    # currently the result of QMSim is not a dict
                    result = new_result
        return result

    def sweep(self, sequence, options, *sweepers):
        """Executes a pulse sequence for different values of sweeped parameters.

        Useful for performing chip characterization.

        Example:
            .. testcode::

                import numpy as np
                from qibolab.dummy import create_dummy
                from qibolab.sweeper import Sweeper, Parameter
                from qibolab.pulses import PulseSequence
                from qibolab import ExecutionParameters


                platform = create_dummy()
                sequence = PulseSequence()
                parameter = Parameter.frequency
                pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
                sequence.add(pulse)
                parameter_range = np.random.randint(10, size=10)
                sweeper = Sweeper(parameter, parameter_range, [pulse])
                platform.sweep(sequence, ExecutionParameters(), sweeper)

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            *sweepers (:class:`qibolab.sweeper.Sweeper`): Sweeper objects that specify which
                parameters are being sweeped.
            **kwargs: May need them for something

        Returns:
            Readout results acquired by after execution.
        """
        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        result = {}
        for instrument in self.instruments:
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

    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        return self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        return self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)

    def create_CZ_pulse_sequence(self, qubits, start=0):
        # Check in the settings if qubits[0]-qubits[1] is a key
        pair = tuple(sorted(qubits))
        if pair not in self.pairs or self.pairs[pair].native_gates.CZ is None:
            raise_error(
                ValueError,
                f"Calibration for CZ gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.CZ.sequence(start)

    def create_MZ_pulse(self, qubit, start):
        return self.qubits[qubit].native_gates.MZ.pulse(start)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        pulse = self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)
        pulse.duration = duration
        return pulse

    def create_qubit_readout_pulse(self, qubit, start):
        return self.create_MZ_pulse(qubit, start)

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        pulse = self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)
        if beta is not None:
            pulse.shape = "Drag(5," + str(beta) + ")"
        return pulse

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
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
        self.qubits[qubit].drive.local_oscillator.frequency = freq

    def get_lo_drive_frequency(self, qubit):
        """Get frequency of the qubit drive local oscillator in Hz."""
        return self.qubits[qubit].drive.local_oscillator.frequency

    def set_lo_readout_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].readout.local_oscillator.frequency = freq

    def get_lo_readout_frequency(self, qubit):
        """Get frequency of the qubit readout local oscillator in Hz."""
        return self.qubits[qubit].readout.local_oscillator.frequency

    def set_lo_twpa_frequency(self, qubit, freq):
        """Set frequency of the local oscillator of the TWPA to which the qubit's feedline is connected to.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].twpa.local_oscillator.frequency = freq

    def get_lo_twpa_frequency(self, qubit):
        """Get frequency of the local oscillator of the TWPA to which the qubit's feedline is connected to in Hz."""
        return self.qubits[qubit].twpa.local_oscillator.frequency

    def set_lo_twpa_power(self, qubit, power):
        """Set power of the local oscillator of the TWPA to which the qubit's feedline is connected to.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            power (int): new value of the power in dBm.
        self.qubits[qubit].twpa.local_oscillator.power = power
        """

    def get_lo_twpa_power(self, qubit):
        """Get power of the local oscillator of the TWPA to which the qubit's feedline is connected to in dBm."""
        return self.qubits[qubit].twpa.local_oscillator.power

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
