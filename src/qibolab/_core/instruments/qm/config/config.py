from dataclasses import dataclass, field
from typing import Optional, Union

from qibolab._core.components import (
    AcquisitionChannel,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Pulse, Readout

from ..components import OpxOutputConfig, QmAcquisitionConfig
from .devices import (
    AnalogOutput,
    Controller,
    ControllerId,
    Controllers,
    FemAnalogOutput,
    ModuleTypes,
    Octave,
    OctaveInput,
    OctaveOutput,
)
from .elements import AcquireOctaveElement, DcElement, Element, RfOctaveElement
from .pulses import (
    QmAcquisition,
    QmPulse,
    Waveform,
    integration_weights,
    operation,
    waveforms_from_pulse,
)

__all__ = ["Configuration"]

DEFAULT_DIGITAL_WAVEFORMS = {"ON": {"samples": [(1, 0)]}}
"""Required to be registered in the config for QM to work.

Also used as triggering that allows the Octave LO signal to pass only
when we are executing something.
"""


@dataclass
class Configuration:
    """Configuration for communicating with the ``QuantumMachinesManager``.

    Contains nested ``dataclass`` objects and is serialized using ``asdict``
    to be sent to the instrument.
    """

    version: int = 1
    controllers: Controllers = field(default_factory=Controllers)
    octaves: dict[str, Octave] = field(default_factory=dict)
    elements: dict[str, Element] = field(default_factory=dict)
    pulses: dict[str, Union[QmPulse, QmAcquisition]] = field(default_factory=dict)
    waveforms: dict[str, Waveform] = field(default_factory=dict)
    digital_waveforms: dict = field(
        default_factory=lambda: DEFAULT_DIGITAL_WAVEFORMS.copy()
    )
    integration_weights: dict = field(default_factory=dict)
    mixers: dict = field(default_factory=dict)

    def add_controller(self, device: ControllerId, modules: dict[str, ModuleTypes]):
        if device not in self.controllers:
            self.controllers[device] = Controller(type=modules[device])

    def add_octave(
        self, device: str, connectivity: ControllerId, modules: dict[str, ModuleTypes]
    ):
        if device not in self.octaves:
            self.add_controller(connectivity, modules)
            self.octaves[device] = Octave(connectivity)

    def configure_dc_line(
        self, id: ChannelId, channel: DcChannel, config: OpxOutputConfig
    ):
        controller = self.controllers[channel.device]
        if controller.type == "opx1":
            controller.analog_outputs[channel.port] = AnalogOutput.from_config(config)
        else:
            controller.analog_outputs[channel.port] = FemAnalogOutput.from_config(
                config
            )
        self.elements[id] = DcElement.from_channel(channel)

    def configure_iq_line(
        self,
        id: ChannelId,
        channel: IqChannel,
        config: IqConfig,
        lo_config: OscillatorConfig,
    ):
        port = channel.port
        octave = self.octaves[channel.device]
        octave.RF_outputs[port] = OctaveOutput.from_config(lo_config)
        self.controllers[octave.connectivity].add_octave_output(port)

        intermediate_frequency = config.frequency - lo_config.frequency
        self.elements[id] = RfOctaveElement.from_channel(
            channel, octave.connectivity, intermediate_frequency
        )

    def configure_acquire_line(
        self,
        id: ChannelId,
        acquire_channel: AcquisitionChannel,
        probe_channel: IqChannel,
        acquire_config: QmAcquisitionConfig,
        probe_config: IqConfig,
        lo_config: OscillatorConfig,
    ):
        port = acquire_channel.port
        octave = self.octaves[acquire_channel.device]
        octave.RF_inputs[port] = OctaveInput(lo_config.frequency)
        self.controllers[octave.connectivity].add_octave_input(port, acquire_config)

        port = probe_channel.port
        octave = self.octaves[probe_channel.device]
        octave.RF_outputs[port] = OctaveOutput.from_config(lo_config)
        self.controllers[octave.connectivity].add_octave_output(port)

        intermediate_frequency = probe_config.frequency - lo_config.frequency
        self.elements[id] = AcquireOctaveElement.from_channel(
            probe_channel,
            acquire_channel,
            octave.connectivity,
            intermediate_frequency,
            time_of_flight=acquire_config.delay,
            smearing=acquire_config.smearing,
        )

    def register_waveforms(
        self,
        pulse: Pulse,
        max_voltage: float,
        element: Optional[str] = None,
        dc: bool = False,
    ):
        if dc:
            qmpulse = QmPulse.from_dc_pulse(pulse)
        else:
            if element is None:
                qmpulse = QmPulse.from_pulse(pulse)
            else:
                qmpulse = QmAcquisition.from_pulse(pulse, element)
        waveforms = waveforms_from_pulse(pulse, max_voltage)
        if dc:
            self.waveforms[qmpulse.waveforms["single"]] = waveforms["I"]
        else:
            for mode in ["I", "Q"]:
                self.waveforms[getattr(qmpulse.waveforms, mode)] = waveforms[mode]
        return qmpulse

    def register_iq_pulse(self, element: str, pulse: Pulse, max_voltage: float):
        op = operation(pulse)
        if op not in self.pulses:
            self.pulses[op] = self.register_waveforms(pulse, max_voltage)
        self.elements[element].operations[op] = op
        return op

    def register_dc_pulse(self, element: str, pulse: Pulse, max_voltage: float):
        op = operation(pulse)
        if op not in self.pulses:
            self.pulses[op] = self.register_waveforms(pulse, max_voltage, dc=True)
        self.elements[element].operations[op] = op
        return op

    def register_acquisition_pulse(
        self, element: str, readout: Readout, max_voltage: float
    ):
        """Registers pulse, waveforms and integration weights in QM config."""
        op = operation(readout)
        acquisition = f"{op}_{element}"
        if acquisition not in self.pulses:
            self.pulses[acquisition] = self.register_waveforms(
                readout.probe, max_voltage, element
            )
        self.elements[element].operations[op] = acquisition
        return op

    def register_integration_weights(self, element: str, duration: int, kernel):
        self.integration_weights.update(integration_weights(element, duration, kernel))
