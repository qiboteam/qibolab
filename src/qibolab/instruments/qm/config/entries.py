from dataclasses import dataclass, field

from ..components import OpxDcConfig

AnalogOutput = OpxDcConfig


@dataclass(frozen=True)
class AnalogInput:
    offset: float = 0.0
    gain_db: int = 0


DigitalOutput = dict


@dataclass
class Controller:
    analog_outputs: dict[str, dict[str, AnalogOutput]] = field(default_factory=dict)
    digital_outputs: dict[str, dict] = field(default_factory=dict)
    analog_inputs: dict[str, dict[str, AnalogInput]] = field(default_factory=dict)

    def add(self, port, config):
        if isinstance(config, AnalogOutput):
            self.analog_outputs[str(port)] = config
        elif isinstance(config, DigitalOutput):
            self.digital_outputs[str(port)] = config
        elif isinstance(config, AnalogInput):
            self.analog_inputs[str(port)] = config
        else:
            raise TypeError


@dataclass
class OctaveOuput:
    LO_frequency: int
    gain: int = 0
    LO_source: str = "internal"
    output_mode: str = "triggered"

    @classmethod
    def from_config(cls, config: OscillatorConfig):
        return cls(
            LO_frequency=config.frequency,
            gain=config.power,
        )


@dataclass
class OctaveInput:
    LO_frequency: int
    LO_source: str = "internal"
    IF_mode_I: str = "direct"
    IF_mode_Q: str = "direct"


@dataclass
class Octave:
    connectivity: str
    RF_outputs: dict[str, dict[str, OctaveOuput]] = field(default_factory=dict)
    RF_inputs: dict[str, dict[str, OctaveInput]] = field(default_factory=dict)

    def add(self, port, config):
        if isinstance(config, OctaveOuput):
            self.RF_outputs[str(port)] = config
        elif isinstance(config, OctaveInput):
            self.RF_inputs[str(port)] = config
        else:
            raise TypeError


@dataclass
class Input:
    port: tuple[str, int]


@dataclass
class Output:
    port: tuple[str, int]


@dataclass
class OutputSwitch:
    port: tuple[str, int]
    delay: int = 57
    buffer: int = 18


@dataclass
class DigitalInput:
    output_switch: OutputSwitch


@dataclass
class DcElement:
    singleInput: Input
    intermediate_frequency: int = 0
    operations: dict[str, str] = field(default_factory=dict)


@dataclass
class RfElement:
    RF_inputs: Input
    digitalInputs: DigitalInput
    intermediate_frequency: int
    operations: dict[str, str] = field(default_factory=dict)


@dataclass
class AcquireElement:
    RF_inputs: Input
    RF_outputs: Output
    digitalInputs: DigitalInput
    intermediate_frequency: int
    time_of_flight: int = 24
    smearing: int = 0
    operations: dict[str, str] = field(default_factory=dict)


Element = DcElement | RfElement | AcquireElement
