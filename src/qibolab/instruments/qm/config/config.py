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

    def add_octave(self, device: str, port: str, connectivity: str):
        if device not in self.octaves:
            self.add_controller(connectivity)
            self.octaves[device] = Octave(connectivity)

    def configure_dc_line(self, channel: QmChannel, config: OpxDcConfig):
        self.controllers[channel.device][str(channel.port)] = asdict(config)
        self.elements[channel.logical_channel.name] = DcElement(
            {
                "port": (channel.device, channel.port),
            }
        )

    def configure_iq_line(
        self, channel: QmChannel, config: IqConfig, lo_config: OscillatorConfig
    ):
        octave = self.octaves[channel.device]
        octave.add(channel.port, OctaveOuput.from_config(lo_config))

        intermediate_frequency = config.frequency - lo_config.frequency
        self.elements[channel.logical_channel.name] = RfElement(
            Input((channel.device, channel.port)),
            DigitalInput(OutputSwitch((opx, opx_i))),
            intermediate_frequency,
        )

    def configure_acquire_line(
        self,
        channel: QmChannel,
        config: QmAcquisitionConfig,
        lo_config: OscillatorConfig,
    ):
        octave = self.octaves[channel.device]
        octave.add(channel.port, OctaveOuput.from_config(lo_config))
        octave.add(channel.port, OctaveInput(lo_config.frequency))

        intermediate_frequency = config.frequency - lo_config.frequency
        self.elements[channel.logical_channel.name] = AcquireElement(
            Input((channel.device, channel.port)),
            DigitalInput(OutputSwitch((opx, opx_i))),
            intermediate_frequency,
        )
