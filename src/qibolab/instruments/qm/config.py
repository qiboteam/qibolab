import math
from dataclasses import dataclass, field

import numpy as np
from qibo.config import raise_error

from qibolab.pulses import PulseType, Rectangular

from .ports import OPXIQ, OctaveInput, OctaveOutput, OPXOutput

SAMPLING_RATE = 1
"""Sampling rate of Quantum Machines OPX in GSps."""

DEFAULT_INPUTS = {1: {}, 2: {}}
"""Default controller config section.

Inputs are always registered to avoid issues with automatic mixer
calibration when using Octaves.
"""


@dataclass
class QMConfig:
    """Configuration for communicating with the ``QuantumMachinesManager``."""

    version: int = 1
    controllers: dict = field(default_factory=dict)
    octaves: dict = field(default_factory=dict)
    elements: dict = field(default_factory=dict)
    pulses: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)
    digital_waveforms: dict = field(
        default_factory=lambda: {"ON": {"samples": [(1, 0)]}}
    )
    integration_weights: dict = field(default_factory=dict)
    mixers: dict = field(default_factory=dict)

    def register_port(self, port):
        """Register controllers and octaves sections in the ``config``.

        Args:
            ports (QMPort): Port we are registering.
                Contains information about the controller and port number and
                some parameters, such as offset, gain, filter, etc.).
        """
        if isinstance(port, OPXIQ):
            self.register_port(port.i)
            self.register_port(port.q)
        else:
            is_octave = isinstance(port, (OctaveOutput, OctaveInput))
            controllers = self.octaves if is_octave else self.controllers
            if port.device not in controllers:
                if is_octave:
                    controllers[port.device] = {}
                else:
                    controllers[port.device] = {
                        "analog_inputs": DEFAULT_INPUTS,
                        "digital_outputs": {},
                    }

            device = controllers[port.device]
            if port.key in device:
                device[port.key].update(port.config)
            else:
                device[port.key] = port.config

            if is_octave:
                con = port.opx_port.i.device
                number = port.opx_port.i.number
                device["connectivity"] = con
                self.register_port(port.opx_port)
                self.controllers[con]["digital_outputs"][number] = {}

    @staticmethod
    def iq_imbalance(g, phi):
        """Creates the correction matrix for the mixer imbalance caused by the
        gain and phase imbalances.

        More information here:
        https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

        Args:
            g (float): relative gain imbalance between the I & Q ports (unit-less).
                Set to 0 for no gain imbalance.
            phi (float): relative phase imbalance between the I & Q ports (radians).
                Set to 0 for no phase imbalance.
        """
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g**2) * (2 * c**2 - 1))
        return [
            float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]
        ]

    def _new_frequency_element(self, qubit, intermediate_frequency, mode="drive"):
        """Register element on existing port but with different frequency."""
        element = f"{mode}{qubit.name}"
        current_if = self.elements[element]["intermediate_frequency"]
        if intermediate_frequency == current_if:
            return element

        if isinstance(getattr(qubit, mode).port, (OPXIQ, OPXOutput)):
            raise NotImplementedError(
                f"Cannot play two different frequencies on the same {mode} line."
            )
        new_element = f"{element}_{intermediate_frequency}"
        self.elements[new_element] = dict(self.elements[element])
        self.elements[new_element]["intermediate_frequency"] = intermediate_frequency
        return new_element

    def register_drive_element(self, qubit, intermediate_frequency=0):
        """Register qubit drive elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        element = f"drive{qubit.name}"
        if element in self.elements:
            return self._new_frequency_element(qubit, intermediate_frequency, "drive")

        if isinstance(qubit.drive.port, OPXIQ):
            lo_frequency = math.floor(qubit.drive.lo_frequency)
            self.elements[element] = {
                "mixInputs": {
                    "I": qubit.drive.port.i.pair,
                    "Q": qubit.drive.port.q.pair,
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_drive{qubit.name}",
                },
            }
            drive_g = qubit.mixer_drive_g
            drive_phi = qubit.mixer_drive_phi
            self.mixers[f"mixer_drive{qubit.name}"] = [
                {
                    "intermediate_frequency": intermediate_frequency,
                    "lo_frequency": lo_frequency,
                    "correction": self.iq_imbalance(drive_g, drive_phi),
                }
            ]
        else:
            self.elements[element] = {
                "RF_inputs": {"port": qubit.drive.port.pair},
                "digitalInputs": qubit.drive.port.digital_inputs,
            }
        self.elements[element].update(
            {
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
        )
        return element

    def register_readout_element(
        self, qubit, intermediate_frequency=0, time_of_flight=0, smearing=0
    ):
        """Register resonator elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        element = f"readout{qubit.name}"
        if element in self.elements:
            return self._new_frequency_element(qubit, intermediate_frequency, "readout")

        if isinstance(qubit.readout.port, OPXIQ):
            lo_frequency = math.floor(qubit.readout.lo_frequency)
            self.elements[element] = {
                "mixInputs": {
                    "I": qubit.readout.port.i.pair,
                    "Q": qubit.readout.port.q.pair,
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_readout{qubit.name}",
                },
                "outputs": {
                    "out1": qubit.feedback.port.i.pair,
                    "out2": qubit.feedback.port.q.pair,
                },
            }
            readout_g = qubit.mixer_readout_g
            readout_phi = qubit.mixer_readout_phi
            self.mixers[f"mixer_readout{qubit.name}"] = [
                {
                    "intermediate_frequency": intermediate_frequency,
                    "lo_frequency": lo_frequency,
                    "correction": self.iq_imbalance(readout_g, readout_phi),
                }
            ]
        else:
            self.elements[element] = {
                "RF_inputs": {"port": qubit.readout.port.pair},
                "RF_outputs": {"port": qubit.feedback.port.pair},
                "digitalInputs": qubit.readout.port.digital_inputs,
            }
        self.elements[element].update(
            {
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
                "time_of_flight": time_of_flight,
                "smearing": smearing,
            }
        )
        return element

    def register_flux_element(self, qubit, intermediate_frequency=0):
        """Register qubit flux elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        element = f"flux{qubit.name}"
        if element in self.elements:
            return self._new_frequency_element(qubit, intermediate_frequency, "flux")

        self.elements[element] = {
            "singleInput": {
                "port": qubit.flux.port.pair,
            },
            "intermediate_frequency": intermediate_frequency,
            "operations": {},
        }
        return element

    def register_element(self, qubit, pulse, time_of_flight=0, smearing=0):
        if pulse.type is PulseType.DRIVE:
            # register drive element
            if_frequency = pulse.frequency - math.floor(qubit.drive.lo_frequency)
            element = self.register_drive_element(qubit, if_frequency)
            # register flux element (if available)
            if qubit.flux:
                self.register_flux_element(qubit)
        elif pulse.type is PulseType.READOUT:
            # register readout element (if it does not already exist)
            if_frequency = pulse.frequency - math.floor(qubit.readout.lo_frequency)
            element = self.register_readout_element(
                qubit, if_frequency, time_of_flight, smearing
            )
            # register flux element (if available)
            if qubit.flux:
                self.register_flux_element(qubit)
        else:
            # register flux element
            element = self.register_flux_element(qubit, pulse.frequency)
        return element

    def register_pulse(self, qubit, qmpulse):
        """Registers pulse, waveforms and integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit that the pulse acts on.
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to register.

        Returns:
            element (str): Name of the element this pulse will be played on.
                Elements are a part of the QM config and are generated during
                instantiation of the Qubit objects. They are named as
                "drive0", "drive1", "flux0", "readout0", ...
        """
        pulse = qmpulse.pulse
        if qmpulse.operation not in self.pulses:
            if pulse.type is PulseType.DRIVE:
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                self.pulses[qmpulse.operation] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {"I": serial_i, "Q": serial_q},
                    "digital_marker": "ON",
                }
                # register drive pulse in elements
                self.elements[qmpulse.element]["operations"][
                    qmpulse.operation
                ] = qmpulse.operation

            elif pulse.type is PulseType.FLUX:
                serial = self.register_waveform(pulse)
                self.pulses[qmpulse.operation] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {
                        "single": serial,
                    },
                }
                # register flux pulse in elements
                self.elements[qmpulse.element]["operations"][
                    qmpulse.operation
                ] = qmpulse.operation

            elif pulse.type is PulseType.READOUT:
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                self.register_integration_weights(qubit, pulse.duration)
                self.pulses[qmpulse.operation] = {
                    "operation": "measurement",
                    "length": pulse.duration,
                    "waveforms": {
                        "I": serial_i,
                        "Q": serial_q,
                    },
                    "integration_weights": {
                        "cos": f"cosine_weights{qubit.name}",
                        "sin": f"sine_weights{qubit.name}",
                        "minus_sin": f"minus_sine_weights{qubit.name}",
                    },
                    "digital_marker": "ON",
                }
                # register readout pulse in elements
                self.elements[qmpulse.element]["operations"][
                    qmpulse.operation
                ] = qmpulse.operation

            else:
                raise_error(TypeError, f"Unknown pulse type {pulse.type.name}.")

    def register_waveform(self, pulse, mode="i"):
        """Registers waveforms in QM config.

        QM supports two kinds of waveforms, examples:
            "zero_wf": {"type": "constant", "sample": 0.0}
            "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()}

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to read the waveform from.
            mode (str): "i" or "q" specifying which channel the waveform will be played.

        Returns:
            serial (str): String with a serialization of the waveform.
                Used as key to identify the waveform in the config.
        """
        if pulse.type is PulseType.READOUT and mode == "q":
            # Force zero q waveforms for readout
            serial = "zero_wf"
            if serial not in self.waveforms:
                self.waveforms[serial] = {"type": "constant", "sample": 0.0}
        elif isinstance(pulse.shape, Rectangular):
            serial = f"constant_wf{pulse.amplitude}"
            if serial not in self.waveforms:
                self.waveforms[serial] = {"type": "constant", "sample": pulse.amplitude}
        else:
            waveform = getattr(pulse, f"envelope_waveform_{mode}")(SAMPLING_RATE)
            serial = hash(waveform)
            if serial not in self.waveforms:
                self.waveforms[serial] = {
                    "type": "arbitrary",
                    "samples": waveform.data.tolist(),
                }
        return serial

    def register_integration_weights(self, qubit, readout_len):
        """Registers integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.quantum_machines.Qubit`): Qubit
                object that the integration weights will be used for.
            readout_len (int): Duration of the readout pulse in ns.
        """
        angle = 0
        cos, sin = np.cos(angle), np.sin(angle)
        if qubit.kernel is None:
            convert = lambda x: [(x, readout_len)]
        else:
            cos = qubit.kernel * cos
            sin = qubit.kernel * sin
            convert = lambda x: x

        self.integration_weights.update(
            {
                f"cosine_weights{qubit.name}": {
                    "cosine": convert(cos),
                    "sine": convert(-sin),
                },
                f"sine_weights{qubit.name}": {
                    "cosine": convert(sin),
                    "sine": convert(cos),
                },
                f"minus_sine_weights{qubit.name}": {
                    "cosine": convert(-sin),
                    "sine": convert(-cos),
                },
            }
        )
