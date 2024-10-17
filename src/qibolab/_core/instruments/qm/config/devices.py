from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, Union

from qibolab._core.components import OscillatorConfig

from ..components import OctaveOscillatorConfig, OpxOutputConfig, QmAcquisitionConfig
from ..components.configs import OctaveOutputModes

__all__ = [
    "AnalogOutput",
    "FemAnalogOutput",
    "ModuleTypes",
    "OctaveOutput",
    "OctaveInput",
    "Controller",
    "Octave",
    "ControllerId",
    "Controllers",
]


DEFAULT_INPUTS = {"1": {"offset": 0}, "2": {"offset": 0}}
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


ModuleTypes = Literal["opx1", "LF", "MW"]


@dataclass
class Controller:
    type: ModuleTypes = "opx1"
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


@dataclass
class Octave:
    connectivity: Union[str, tuple[str, int]]
    RF_outputs: PortDict[dict[str, OctaveOutput]] = field(default_factory=PortDict)
    RF_inputs: PortDict[dict[str, OctaveInput]] = field(default_factory=PortDict)

    def __post_init__(self):
        if "/" in self.connectivity:
            con, fem = self.connectivity.split("/")
            self.connectivity = (con, int(fem))


ControllerId = Union[str, tuple[str, int]]


def process_controller_id(id: ControllerId):
    """Convert controller identifier depending on cluster type.

    For OPX+ clusters ``id`` is just the controller name (eg. 'con1').
    For OPX1000 clusters ``id`` has the format
    '{controller_name}/{fem_number}' (eg. 'con1/4').
    In that case ``id`` may also be a ``tuple``
    `(controller_name, fem_number)`
    """
    if isinstance(id, tuple):
        con, fem = id
        return con, str(fem)
    if "/" in id:
        return id.split("/")
    return id, None


class Controllers(dict[str, Union[Controller, Opx1000]]):
    """Dictionary of controllers compatible with OPX+ and OPX1000."""

    def __contains__(self, key: ControllerId) -> bool:
        con, fem = process_controller_id(key)
        contains = super().__contains__(con)
        if fem is None:
            return contains
        return contains and fem in self[con].fems

    def __getitem__(self, key: ControllerId) -> Controller:
        con, fem = process_controller_id(key)
        value = super().__getitem__(con)
        if fem is None:
            return value
        return value.fems[fem]

    def __setitem__(self, key: ControllerId, value: Controller):
        con, fem = process_controller_id(key)
        if fem is None:
            super().__setitem__(key, value)
        else:
            if con not in self:
                super().__setitem__(con, Opx1000())
            self[con].fems[fem] = value
