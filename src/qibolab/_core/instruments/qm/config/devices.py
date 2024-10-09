from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, Union

from qibolab._core.components import OscillatorConfig

from ..components import (
    OctaveOscillatorConfig,
    OctaveOutputModes,
    OpxOutputConfig,
    QmAcquisitionConfig,
)

__all__ = [
    "AnalogOutput",
    "FemAnalogOutput",
    "OctaveOutput",
    "OctaveInput",
    "Controller",
    "Octave",
    "ControllerId",
    "Controllers",
]


DEFAULT_INPUTS = {"1": {}, "2": {}}
"""Default controller config section.

Inputs are always registered to avoid issues with automatic mixer
calibration when using Octaves.
"""

V = TypeVar("V")


class PortDict(Generic[V], dict[str, V]):
    """Dictionary that automatically converts keys to strings.

    Used to register input and output ports to controllers and Octaves
    in the QUA config.
    """

    def __setitem__(self, key: Union[str, int], value: V):
        super().__setitem__(str(key), value)


@dataclass(frozen=True)
class AnalogOutput:
    offset: float = 0.0
    filter: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: OpxOutputConfig):
        return cls(offset=config.offset, filter=config.filter)


@dataclass(frozen=True)
class FemAnalogOutput(AnalogOutput):
    output_mode: Literal["direct", "amplified"] = "direct"

    @classmethod
    def from_config(cls, config: OpxOutputConfig):
        return cls(
            offset=config.offset, filter=config.filter, output_mode=config.output_mode
        )


@dataclass(frozen=True)
class AnalogInput:
    offset: float = 0.0
    gain_db: int = 0

    @classmethod
    def from_config(cls, config: QmAcquisitionConfig):
        return cls(offset=config.offset, gain_db=config.gain)


@dataclass(frozen=True)
class OctaveOutput:
    LO_frequency: int
    gain: int = 0
    LO_source: Literal["internal", "external"] = "internal"
    output_mode: OctaveOutputModes = "triggered"

    @classmethod
    def from_config(cls, config: Union[OscillatorConfig, OctaveOscillatorConfig]):
        kwargs = dict(LO_frequency=config.frequency, gain=config.power)
        if isinstance(config, OctaveOscillatorConfig):
            kwargs["output_mode"] = config.output_mode
        return cls(**kwargs)


@dataclass(frozen=True)
class OctaveInput:
    LO_frequency: int
    LO_source: Literal["internal", "external"] = "internal"
    IF_mode_I: Literal["direct", "envelop", "mixer"] = "direct"
    IF_mode_Q: Literal["direct", "envelop", "mixer"] = "direct"


@dataclass
class Controller:
    type: Literal["opx1", "LF", "MW"] = "opx1"
    """https://docs.quantum-machines.co/latest/docs/Introduction/config/?h=opx10#controllers"""
    analog_outputs: PortDict[dict[str, AnalogOutput]] = field(default_factory=PortDict)
    digital_outputs: PortDict[dict[str, dict]] = field(default_factory=PortDict)
    analog_inputs: PortDict[dict[str, AnalogInput]] = field(
        default_factory=lambda: PortDict(DEFAULT_INPUTS)
    )

    def add_octave_output(self, port: int):
        # TODO: Add offset here?
        self.analog_outputs[2 * port - 1] = AnalogOutput()
        self.analog_outputs[2 * port] = AnalogOutput()

        self.digital_outputs[2 * port - 1] = {}

    def add_octave_input(self, port: int, config: QmAcquisitionConfig):
        self.analog_inputs[2 * port - 1] = self.analog_inputs[2 * port] = (
            AnalogInput.from_config(config)
        )


@dataclass
class Opx1000:
    type: Literal["opx1000"] = "opx1000"
    fems: dict[str, Controller] = field(default_factory=PortDict)


ControllerId = str
"""Controller identifier.

Single string name for OPX+ clusters, or
'{controller_name}/{fem_number}' for OPX1000 clusters.
"""


@dataclass
class Octave:
    connectivity: Union[str, tuple[str, int]]
    RF_outputs: PortDict[dict[str, OctaveOutput]] = field(default_factory=PortDict)
    RF_inputs: PortDict[dict[str, OctaveInput]] = field(default_factory=PortDict)

    def __post_init__(self):
        if "/" in self.connectivity:
            con, fem = self.connectivity.split("/")
            self.connectivity = (con, int(fem))


class Controllers(dict[str, Union[Controller, Opx1000]]):
    """Dictionary of controllers compatible with both OPX+ and OPX1000
    clusters."""

    def __getitem__(self, key: ControllerId) -> Controller:
        if not isinstance(key, tuple) and "/" not in key:
            return super().__getitem__(key)

        if isinstance(key, tuple):
            con, fem = key
            fem = str(fem)
        else:
            con, fem = key.split("/")
        return super().__getitem__(con).fems[str(fem)]

    def __setitem__(self, key: ControllerId, value: Controller):
        if not isinstance(key, tuple) and "/" not in key:
            super().__setitem__(key, value)
        else:
            if isinstance(key, tuple):
                con, fem = key
                fem = str(fem)
            else:
                con, fem = key.split("/")
            if con not in self:
                super().__setitem__(con, Opx1000())
            self[con].fems[fem] = value
