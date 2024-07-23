from dataclasses import dataclass, field

from qibolab.components.configs import IqConfig, OscillatorConfig
from qibolab.pulses import Pulse

from ..components import OpxOutputConfig, QmAcquisitionConfig, QmChannel
from .devices import *
from .elements import *
from .pulses import *

__all__ = ["QmConfig"]


@dataclass
class QmConfig:
    """Configuration for communicating with the ``QuantumMachinesManager``."""

    version: int = 1
    controllers: dict[str, Controller] = field(default_factory=dict)
    octaves: dict[str, Octave] = field(default_factory=dict)
    elements: dict[str, Element] = field(default_factory=dict)
    pulses: dict[str, QmPulse | QmAcquisition] = field(default_factory=dict)
    waveforms: dict[str, Waveform] = field(default_factory=dict)
    digital_waveforms: dict = field(
        default_factory=lambda: {"ON": {"samples": [(1, 0)]}}
    )
    integration_weights: dict = field(default_factory=dict)
    mixers: dict = field(default_factory=dict)

    def add_controller(self, device: str):
        if device not in self.controllers:
            self.controllers[device] = Controller()

    def add_octave(self, device: str, connectivity: str):
        if device not in self.octaves:
            self.add_controller(connectivity)
            self.octaves[device] = Octave(connectivity)

    def configure_dc_line(self, channel: QmChannel, config: OpxOutputConfig):
        controller = self.controllers[channel.device]
        controller.analog_outputs[str(channel.port)] = config
        self.elements[channel.logical_channel.name] = DcElement(channel.serial)

    def configure_iq_line(
        self, channel: QmChannel, config: IqConfig, lo_config: OscillatorConfig
    ):
        port = channel.port
        octave = self.octaves[channel.device]
        octave.RF_outputs[str(port)] = OctaveOutput.from_config(lo_config)
        self.controllers[octave.connectivity].add_octave_output(port)

        intermediate_frequency = config.frequency - lo_config.frequency
        self.elements[channel.logical_channel.name] = RfOctaveElement(
            channel.serial,
            output_switch(octave.connectivity, channel.port),
            intermediate_frequency,
        )

    def configure_acquire_line(
        self,
        acquire_channel: QmChannel,
        probe_channel: QmChannel,
        acquire_config: QmAcquisitionConfig,
        probe_config: IqConfig,
        lo_config: OscillatorConfig,
    ):
        port = acquire_channel.port
        octave = self.octaves[acquire_channel.device]
        octave.RF_inputs[str(port)] = OctaveInput(lo_config.frequency)
        self.controllers[octave.connectivity].add_octave_input(port, acquire_config)

        port = probe_channel.port
        octave = self.octaves[probe_channel.device]
        octave.RF_outputs[str(port)] = OctaveOutput.from_config(lo_config)
        self.controllers[octave.connectivity].add_octave_output(port)

        intermediate_frequency = probe_config.frequency - lo_config.frequency
        self.elements[probe_channel.logical_channel.name] = AcquireOctaveElement(
            probe_channel.serial,
            acquire_channel.serial,
            output_switch(octave.connectivity, probe_channel.port),
            intermediate_frequency,
            time_of_flight=acquire_config.delay,
            smearing=acquire_config.smearing,
        )

    def register_iq_pulse(self, element: str, pulse: Pulse):
        op = operation(pulse)
        if op not in self.pulses:
            self.pulses[op] = qmpulse = QmPulse.from_pulse(pulse)
            waveforms = waveforms_from_pulse(pulse)
            for mode in ["I", "Q"]:
                self.waveforms[qmpulse.waveforms[mode]] = waveforms[mode]
        self.elements[element].operations[op] = op
        return op

    def register_dc_pulse(self, element: str, pulse: Pulse):
        op = operation(pulse)
        if op not in self.pulses:
            self.pulses[op] = qmpulse = QmPulse.from_dc_pulse(pulse)
            self.waveforms[qmpulse.waveforms["I"]] = waveforms_from_pulse(pulse)["I"]
        self.elements[element].operations[op] = op
        return op

    def register_acquisition_pulse(self, element: str, pulse: Pulse):
        """Registers pulse, waveforms and integration weights in QM config."""
        op = operation(pulse)
        acquisition = f"{op}_{element}"
        if acquisition not in self.pulses:
            self.pulses[acquisition] = qmpulse = QmAcquisition.from_pulse(
                pulse, element
            )
            waveforms = waveforms_from_pulse(pulse)
            for mode in ["I", "Q"]:
                self.waveforms[qmpulse.waveforms[mode]] = waveforms[mode]
        self.elements[element].operations[op] = acquisition
        return op

    def register_integration_weights(self, element: str, duration: int, kernel):
        self.integration_weights.update(integration_weights(element, duration, kernel))
