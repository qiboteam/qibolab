import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
from qibo.config import raise_error

from qibolab.instruments.port import Port
from qibolab.pulses import PulseType, Rectangular

PortId = Tuple[str, int]
"""Type for port definition, for example: ("con1", 2)."""
IQPortId = Union[Tuple[PortId], Tuple[PortId, PortId]]
"""Type for collections of IQ ports."""


@dataclass
class QMPort(Port):
    name: IQPortId
    offset: float = 0.0
    gain: int = 0
    filters: Optional[Dict[str, float]] = None


@dataclass
class QMConfig:
    """Configuration for communicating with the ``QuantumMachinesManager``."""

    version: int = 1
    controllers: dict = field(default_factory=dict)
    elements: dict = field(default_factory=dict)
    pulses: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)
    digital_waveforms: dict = field(default_factory=lambda: {"ON": {"samples": [(1, 0)]}})
    integration_weights: dict = field(default_factory=dict)
    mixers: dict = field(default_factory=dict)

    def register_analog_output_controllers(self, port: QMPort):
        """Register controllers in the ``config``.

        Args:
            ports (QMPort): Port we are registering.
                Contains information about the controller and port number and
                some parameters (offset, gain, filter, etc.).
        """
        for con, port_number in port.name:
            if con not in self.controllers:
                self.controllers[con] = {"analog_outputs": {}}
            self.controllers[con]["analog_outputs"][port_number] = {"offset": port.offset}
            if port.filters is not None:
                self.controllers[con]["analog_outputs"][port_number]["filter"] = port.filters

    @staticmethod
    def iq_imbalance(g, phi):
        """Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances

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

    def register_drive_element(self, qubit, intermediate_frequency=0):
        """Register qubit drive elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"drive{qubit.name}" not in self.elements:
            # register drive controllers
            self.register_analog_output_controllers(qubit.drive.port)
            # register element
            lo_frequency = math.floor(qubit.drive.local_oscillator.frequency)
            self.elements[f"drive{qubit.name}"] = {
                "mixInputs": {
                    "I": qubit.drive.port.name[0],
                    "Q": qubit.drive.port.name[1],
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_drive{qubit.name}",
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
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
            self.elements[f"drive{qubit.name}"]["intermediate_frequency"] = intermediate_frequency
            self.mixers[f"mixer_drive{qubit.name}"][0]["intermediate_frequency"] = intermediate_frequency

    def register_readout_element(self, qubit, intermediate_frequency=0, time_of_flight=0, smearing=0):
        """Register resonator elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"readout{qubit.name}" not in self.elements:
            # register readout controllers
            self.register_analog_output_controllers(qubit.readout.port)
            # register feedback controllers
            controllers = self.controllers
            for con, port_number in qubit.feedback.port.name:
                if con not in controllers:
                    controllers[con] = {
                        "analog_outputs": {},
                        "digital_outputs": {
                            1: {},
                        },
                        "analog_inputs": {},
                    }
                if "digital_outputs" not in controllers[con]:
                    controllers[con]["digital_outputs"] = {
                        1: {},
                    }
                if "analog_inputs" not in controllers[con]:
                    controllers[con]["analog_inputs"] = {}
                controllers[con]["analog_inputs"][port_number] = {"offset": 0.0, "gain_db": qubit.feedback.port.gain}
            # register element
            lo_frequency = math.floor(qubit.readout.local_oscillator.frequency)
            self.elements[f"readout{qubit.name}"] = {
                "mixInputs": {
                    "I": qubit.readout.port.name[0],
                    "Q": qubit.readout.port.name[1],
                    "lo_frequency": lo_frequency,
                    "mixer": f"mixer_readout{qubit.name}",
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
                "outputs": {
                    "out1": qubit.feedback.port.name[0],
                    "out2": qubit.feedback.port.name[1],
                },
                "time_of_flight": time_of_flight,
                "smearing": smearing,
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
            self.elements[f"readout{qubit.name}"]["intermediate_frequency"] = intermediate_frequency
            self.mixers[f"mixer_readout{qubit.name}"][0]["intermediate_frequency"] = intermediate_frequency

    def register_flux_element(self, qubit, intermediate_frequency=0):
        """Register qubit flux elements and controllers in the QM config.

        Args:
            qubit (:class:`qibolab.platforms.utils.Qubit`): Qubit to add elements for.
            intermediate_frequency (int): Intermediate frequency that the OPX
                will send to this qubit. This frequency will be mixed with the
                LO connected to the same channel.
        """
        if f"flux{qubit.name}" not in self.elements:
            # register controller
            self.register_analog_output_controllers(qubit.flux.port)
            # register element
            self.elements[f"flux{qubit.name}"] = {
                "singleInput": {
                    "port": qubit.flux.port.name[0],
                },
                "intermediate_frequency": intermediate_frequency,
                "operations": {},
            }
        else:
            self.elements[f"flux{qubit.name}"]["intermediate_frequency"] = intermediate_frequency

    def register_element(self, qubit, pulse, time_of_flight=0, smearing=0):
        if pulse.type is PulseType.DRIVE:
            # register drive element
            if_frequency = pulse.frequency - math.floor(qubit.drive.local_oscillator.frequency)
            self.register_drive_element(qubit, if_frequency)
            # register flux element (if available)
            if qubit.flux:
                self.register_flux_element(qubit)
        elif pulse.type is PulseType.READOUT:
            # register readout element (if it does not already exist)
            if_frequency = pulse.frequency - math.floor(qubit.readout.local_oscillator.frequency)
            self.register_readout_element(qubit, if_frequency, time_of_flight, smearing)
            # register flux element (if available)
            if qubit.flux:
                self.register_flux_element(qubit)
        else:
            # register flux element
            self.register_flux_element(qubit, pulse.frequency)

    def register_pulse(self, qubit, pulse):
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
        if pulse.serial not in self.pulses:
            if pulse.type is PulseType.DRIVE:
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                self.pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {"I": serial_i, "Q": serial_q},
                }
                # register drive pulse in elements
                self.elements[f"drive{qubit.name}"]["operations"][pulse.serial] = pulse.serial

            elif pulse.type is PulseType.FLUX:
                serial = self.register_waveform(pulse)
                self.pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {
                        "single": serial,
                    },
                }
                # register flux pulse in elements
                self.elements[f"flux{qubit.name}"]["operations"][pulse.serial] = pulse.serial

            elif pulse.type is PulseType.READOUT:
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                self.register_integration_weights(qubit, pulse.duration)
                self.pulses[pulse.serial] = {
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
                self.elements[f"readout{qubit.name}"]["operations"][pulse.serial] = pulse.serial

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
        # Maybe need to force zero q waveforms
        # if pulse.type.name == "READOUT" and mode == "q":
        #    serial = "zero_wf"
        #    if serial not in self.waveforms:
        #        self.waveforms[serial] = {"type": "constant", "sample": 0.0}
        if isinstance(pulse.shape, Rectangular):
            serial = f"constant_wf{pulse.amplitude}"
            if serial not in self.waveforms:
                self.waveforms[serial] = {"type": "constant", "sample": pulse.amplitude}
        else:
            waveform = getattr(pulse, f"envelope_waveform_{mode}")
            serial = waveform.serial
            if serial not in self.waveforms:
                self.waveforms[serial] = {"type": "arbitrary", "samples": waveform.data.tolist()}
        return serial

    def register_integration_weights(self, qubit, readout_len):
        """Registers integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.quantum_machines.Qubit`): Qubit
                object that the integration weights will be used for.
            readout_len (int): Duration of the readout pulse in ns.
        """
        angle = 0
        self.integration_weights.update(
            {
                f"cosine_weights{qubit.name}": {
                    "cosine": [(np.cos(angle), readout_len)],
                    "sine": [(-np.sin(angle), readout_len)],
                },
                f"sine_weights{qubit.name}": {
                    "cosine": [(np.sin(angle), readout_len)],
                    "sine": [(np.cos(angle), readout_len)],
                },
                f"minus_sine_weights{qubit.name}": {
                    "cosine": [(-np.sin(angle), readout_len)],
                    "sine": [(-np.cos(angle), readout_len)],
                },
            }
        )
