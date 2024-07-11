from dataclasses import dataclass, field

import numpy as np

from qibolab.pulses import PulseType, Rectangular

from .ports import OPXIQ

SAMPLING_RATE = 1
"""Sampling rate of Quantum Machines OPX in GSps."""

DEFAULT_INPUTS = {1: {}, 2: {}}
"""Default controller config section.

Inputs are always registered to avoid issues with automatic mixer
calibration when using Octaves.
"""
DIGITAL_DELAY = 57
DIGITAL_BUFFER = 18
"""Default calibration parameters for digital pulses.

https://docs.quantum-machines.co/1.1.7/qm-qua-sdk/docs/Guides/octave/#calibrating-the-digital-pulse

Digital markers are used for LO triggering.
"""


def iq_imbalance(g, phi):
    """Creates the correction matrix for the mixer imbalance caused by the gain
    and phase imbalances.

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
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


def operation(pulse):
    """Generate operation name in QM ``config`` for the given pulse."""
    return str(hash(pulse))


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

    def register_opx_output(
        self,
        device: str,
        port: int,
        digital_port: int = None,
        offset: float = 0.0,
        filter: dict[str, float] = None,
    ):
        if device not in self.controllers:
            self.controllers[device] = {
                "analog_inputs": DEFAULT_INPUTS,
                "digital_outputs": {},
                "analog_outputs": {},
            }

        if digital_port is not None:
            self.controllers[device]["digital_outputs"][str(digital_port)] = {}

        self.controllers[device]["analog_outputs"][str(port)] = {
            "offset": offset,
            "filter": filter if filter is not None else {},
        }

    def register_opx_input(
        self, device: str, port: int, offset: float = 0.0, gain: int = 0
    ):
        # assumes output is already registered
        self.controllers[device]["analog_inputs"][str(port)] = {
            "offset": offset,
            "gain_db": gain,
        }

    def register_octave_output(
        self,
        device: str,
        port: int,
        connectivity: str,
        frequency: int = None,
        power: int = None,
    ):
        if device not in self.controllers:
            self.octaves[device] = {
                "RF_outputs": {},
                "connectivity": connectivity,
                "RF_inputs": {},
            }
        self.octaves[device]["RF_outputs"][str(port)] = {
            "LO_frequency": frequency,
            "gain": power,
            "LO_source": "internal",
            "output_mode": "triggered",
        }

    def register_octave_input(self, device: str, port: int, frequency: int = None):
        # assumes output is already registered
        self.octaves[device]["RF_inputs"][str(port)] = {
            "LO_frequency": frequency,
            "LO_source": "internal",
            "IF_mode_I": "direct",
            "IF_mode_Q": "direct",
        }

    def register_dc_element(self, channel: QmChannel):
        """Register qubit flux elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        element = channel.logical_channel.name
        self.elements[element] = {
            "singleInput": {
                "port": (channel.device, channel.port),
            },
            "intermediate_frequency": 0,
            "operations": {},
        }

    def register_iq_element(
        self, channel: QmChannel, intermediate_frequency=0, opx=None
    ):
        """Register qubit drive elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        element = channel.logical_channel.name
        if isinstance(qubit.drive.port, OPXIQ):
            raise NotImplementedError
            # lo_frequency = math.floor(qubit.drive.lo_frequency)
            # self.elements[element] = {
            #    "mixInputs": {
            #        "I": qubit.drive.port.i.pair,
            #        "Q": qubit.drive.port.q.pair,
            #        "lo_frequency": lo_frequency,
            #        "mixer": f"mixer_drive{qubit.name}",
            #    },
            # }
            # drive_g = qubit.mixer_drive_g
            # drive_phi = qubit.mixer_drive_phi
            # self.mixers[f"mixer_drive{qubit.name}"] = [
            #    {
            #        "intermediate_frequency": intermediate_frequency,
            #        "lo_frequency": lo_frequency,
            #        "correction": iq_imbalance(drive_g, drive_phi),
            #    }
            # ]
        else:
            self.elements[element] = {
                "RF_inputs": {"port": (channel.device, channel.port)},
                "digitalInputs": {
                    "output_switch": {
                        "port": (opx, 2 * channel.device - 1),
                        "delay": DIGITAL_DELAY,
                        "buffer": DIGITAL_BUFFER,
                    },
                },
            }
        self.elements[element].update(
            {
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
        )

    def register_acquire_element(
        self, channel: QmChannel, time_of_flight=0, smearing=0
    ):
        """Register resonator elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        element = measure_channel.logical_channel.name
        if isinstance(qubit.readout.port, OPXIQ):
            raise NotImplementedError
            # lo_frequency = math.floor(qubit.readout.lo_frequency)
            # self.elements[element] = {
            #    "mixInputs": {
            #        "I": qubit.readout.port.i.pair,
            #        "Q": qubit.readout.port.q.pair,
            #        "lo_frequency": lo_frequency,
            #        "mixer": f"mixer_readout{qubit.name}",
            #    },
            #    "outputs": {
            #        "out1": qubit.feedback.port.i.pair,
            #        "out2": qubit.feedback.port.q.pair,
            #    },
            # }
            # readout_g = qubit.mixer_readout_g
            # readout_phi = qubit.mixer_readout_phi
            # self.mixers[f"mixer_readout{qubit.name}"] = [
            #    {
            #        "intermediate_frequency": intermediate_frequency,
            #        "lo_frequency": lo_frequency,
            #        "correction": iq_imbalance(readout_g, readout_phi),
            #    }
            # ]
        else:
            self.elements[element]["RF_outputs"] = {
                "port": (channel.device, channel.port)
            }

        self.elements[element].update(
            {
                "time_of_flight": time_of_flight,
                "smearing": smearing,
            }
        )

    def register_iq_pulse(self, element, pulse):
        op = operation(pulse)
        serial_i = self.register_waveform(pulse, "i")
        serial_q = self.register_waveform(pulse, "q")
        self.pulses[op] = {
            "operation": "control",
            "length": pulse.duration,
            "waveforms": {"I": serial_i, "Q": serial_q},
            "digital_marker": "ON",
        }
        # register drive pulse in elements
        self.elements[element]["operations"][op] = op

    def register_dc_pulse(self, element, pulse):
        op = operation(pulse)
        serial = self.register_waveform(pulse)
        self.pulses[op] = {
            "operation": "control",
            "length": pulse.duration,
            "waveforms": {
                "single": serial,
            },
        }
        # register flux pulse in elements
        self.elements[element]["operations"][op] = op

    def register_acquisition_pulse(self, element, pulse, kernel=None):
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
        op = operation(pulse)
        serial_i = self.register_waveform(pulse, "i")
        serial_q = self.register_waveform(pulse, "q")
        self.register_integration_weights(element, pulse.duration, kernel)
        self.pulses[qmpulse.operation] = {
            "operation": "measurement",
            "length": pulse.duration,
            "waveforms": {
                "I": serial_i,
                "Q": serial_q,
            },
            "integration_weights": {
                "cos": f"cosine_weights_{element}",
                "sin": f"sine_weights_{element}",
                "minus_sin": f"minus_sine_weights_{element}",
            },
            "digital_marker": "ON",
        }
        # register readout pulse in elements
        self.elements[element]["operations"][op] = op
        return op

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
        # TODO: Remove this?
        if pulse.type is PulseType.READOUT and mode == "q":
            # Force zero q waveforms for readout
            serial = "zero_wf"
            if serial not in self.waveforms:
                self.waveforms[serial] = {"type": "constant", "sample": 0.0}
            return serial

        phase = (pulse.relative_phase % (2 * np.pi)) / (2 * np.pi)
        serial = f"{hash(pulse)}_{mode}"
        if isinstance(pulse.envelope, Rectangular):
            if serial not in self.waveforms:
                if mode == "i":
                    sample = pulse.amplitude * np.cos(phase)
                else:
                    sample = pulse.amplitude * np.sin(phase)
                self.waveforms[serial] = {"type": "constant", "sample": sample}
        else:
            if serial not in self.waveforms:
                samples_i = pulse.i(SAMPLING_RATE)
                samples_q = pulse.q(SAMPLING_RATE)
                if mode == "i":
                    samples = samples_i * np.cos(phase) - samples_q * np.sin(phase)
                else:
                    samples = samples_i * np.sin(phase) + samples_q * np.cos(phase)
                self.waveforms[serial] = {
                    "type": "arbitrary",
                    "samples": samples.tolist(),
                }
        return serial

    def register_integration_weights(self, element, readout_len, kernel=None):
        """Registers integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.quantum_machines.Qubit`): Qubit
                object that the integration weights will be used for.
            readout_len (int): Duration of the readout pulse in ns.
        """
        angle = 0
        cos, sin = np.cos(angle), np.sin(angle)
        if kernel is None:
            convert = lambda x: [(x, readout_len)]
        else:
            cos = kernel * cos
            sin = kernel * sin
            convert = lambda x: x

        self.integration_weights.update(
            {
                f"cosine_weights_{element}": {
                    "cosine": convert(cos),
                    "sine": convert(-sin),
                },
                f"sine_weights_{element}": {
                    "cosine": convert(sin),
                    "sine": convert(cos),
                },
                f"minus_sine_weights_{element}": {
                    "cosine": convert(-sin),
                    "sine": convert(-cos),
                },
            }
        )
