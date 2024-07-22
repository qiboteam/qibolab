from dataclasses import asdict, dataclass, field


@dataclass
class QmConfig:
    """Configuration for communicating with the ``QuantumMachinesManager``."""

    version: int = 1
    controllers: dict[str, Controller] = field(default_factory=dict)
    octaves: dict[str, Octave] = field(default_factory=dict)
    elements: dict[str, Element] = field(default_factory=dict)
    pulses: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)
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

    def configure_dc_line(self, channel: QmChannel, config: OpxDcConfig):
        controller = self.controllers[channel.device]
        controller.analog_outputs[str(channel.port)] = asdict(config)
        self.elements[channel.logical_channel.name] = DcElement(channel.serial)

    def configure_iq_line(
        self, channel: QmChannel, config: IqConfig, lo_config: OscillatorConfig
    ):
        port = channel.port
        octave = self.octaves[channel.device]
        octave.RF_outputs[str(port)] = OctaveOuput.from_config(lo_config)
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
        self.controllers[octave.connectivity].add_octave_input(port, config)

        port = probe_channel.port
        octave = self.octaves[probe_channel.device]
        octave.RF_outputs[str(port)] = OctaveOuput.from_config(lo_config)
        self.controllers[octave.connectivity].add_octave_output(port)

        intermediate_frequency = probe_config.frequency - lo_config.frequency
        self.elements[channel.logical_channel.name] = RfOctaveElement(
            probe_channel.serial,
            acquire_channel.serial,
            output_switch(octave.connectivity, probe_channel.port),
            intermediate_frequency,
            time_of_flight=acquire_config.delay,
            smearing=acquire_config.smearing,
        )
