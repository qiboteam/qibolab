from dataclasses import dataclass, field

from qibolab.components.configs import OscillatorConfig

from ..components import OpxOutputConfig, QmAcquisitionConfig

__all__ = [
    "OctaveOutput",
    "OctaveInput",
    "Controller",
    "Octave",
]


DEFAULT_INPUTS = {"1": {}, "2": {}}
"""Default controller config section.

Inputs are always registered to avoid issues with automatic mixer
calibration when using Octaves.
"""


class PortDict(dict):
    """Dictionary that automatically converts keys to strings.

    Used to register input and output ports to controllers and Octaves
    in the QUA config.
    """

    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)


@dataclass(frozen=True)
class AnalogInput:
    offset: float = 0.0
    gain_db: int = 0

    @classmethod
    def from_config(cls, config: QmAcquisitionConfig):
        return cls(
            offset=config.offset,
            gain_db=config.gain,
        )


@dataclass(frozen=True)
class OctaveOutput:
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


@dataclass(frozen=True)
class OctaveInput:
    LO_frequency: int
    LO_source: str = "internal"
    IF_mode_I: str = "direct"
    IF_mode_Q: str = "direct"


@dataclass
class Controller:
    analog_outputs: PortDict[str, dict[str, OpxOutputConfig]] = field(
        default_factory=PortDict
    )
    digital_outputs: PortDict[str, dict[str, dict]] = field(default_factory=PortDict)
    analog_inputs: PortDict[str, dict[str, AnalogInput]] = field(
        default_factory=lambda: PortDict(DEFAULT_INPUTS)
    )

    def add_octave_output(self, port: int):
        # TODO: Add offset here?
        self.analog_outputs[2 * port - 1] = OpxOutputConfig()
        self.analog_outputs[2 * port] = OpxOutputConfig()

        self.digital_outputs[2 * port - 1] = {}

    def add_octave_input(self, port: int, config: QmAcquisitionConfig):
        self.analog_inputs[2 * port - 1] = self.analog_inputs[2 * port] = (
            AnalogInput.from_config(config)
        )


@dataclass
class Octave:
    connectivity: str
    RF_outputs: PortDict[str, dict[str, OctaveOutput]] = field(default_factory=PortDict)
    RF_inputs: PortDict[str, dict[str, OctaveInput]] = field(default_factory=PortDict)
