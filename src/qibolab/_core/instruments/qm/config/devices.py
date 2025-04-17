from dataclasses import dataclass, field
from typing import Literal, Union

from qibolab._core.components import OscillatorConfig

from ..components import (
    MwFemOscillatorConfig,
    OctaveOscillatorConfig,
    QmAcquisitionConfig,
)
from ..components.configs import OctaveOutputModes

__all__ = [
    "ModuleTypes",
    "MwFemOutput",
    "MwFemInput",
    "OctaveOutput",
    "OctaveInput",
    "Controller",
    "Octave",
    "ControllerId",
    "Controllers",
]


@dataclass(frozen=True)
class AnalogInput:
    offset: float = 0.0
    gain_db: int = 0

    @classmethod
    def from_config(cls, config: QmAcquisitionConfig):
        return cls(offset=config.offset, gain_db=config.gain)


@dataclass
class MwFemOutput:
    upconverters: dict[int, dict[Literal["frequency"], float]]
    band: int
    sampling_rate: float
    full_scale_power_dbm: int

    @classmethod
    def from_config(cls, config: MwFemOscillatorConfig):
        upconverters = {config.upconverter: {"frequency": config.frequency}}
        return cls(
            upconverters=upconverters,
            band=config.band,
            sampling_rate=config.sampling_rate,
            full_scale_power_dbm=config.power,
        )

    def update(self, config: MwFemOscillatorConfig):
        assert self.band == config.band
        assert self.sampling_rate == config.sampling_rate
        assert self.full_scale_power_dbm == config.power
        if config.upconverter not in self.upconverters:
            self.upconverters[config.upconverter] = {"frequency": config.frequency}
        else:
            assert (
                config.frequency == self.upconverters[config.upconverter]["frequency"]
            )


@dataclass(frozen=True)
class MwFemInput:
    downconverter_frequency: float
    band: int
    sampling_rate: float

    @classmethod
    def from_config(cls, config: MwFemOscillatorConfig):
        return cls(
            downconverter_frequency=config.frequency,
            band=config.band,
            sampling_rate=config.sampling_rate,
        )


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
    analog_outputs: dict[int, dict] = field(default_factory=dict)
    digital_outputs: dict[int, dict] = field(default_factory=dict)
    analog_inputs: dict[int, Union[AnalogInput, MwFemInput]] = field(
        default_factory=dict
    )

    def _set_default_inputs(self):
        """Add default inputs in controller config section.

        Inputs are always registered to avoid issues with automatic mixer
        calibration when using Octaves.
        """
        for port in range(1, 3):
            if port not in self.analog_inputs:
                self.analog_inputs[port] = {"offset": 0}

    def add_octave_output(self, port: int):
        # TODO: Add offset here?
        self._set_default_inputs()
        self.analog_outputs[2 * port - 1] = {"offset": 0}
        self.analog_outputs[2 * port] = {"offset": 0}

        self.digital_outputs[2 * port - 1] = {}

    def add_octave_input(self, port: int, config: QmAcquisitionConfig):
        self._set_default_inputs()
        self.analog_inputs[2 * port - 1] = self.analog_inputs[2 * port] = (
            AnalogInput.from_config(config)
        )


@dataclass
class Opx1000:
    type: Literal["opx1000"] = "opx1000"
    fems: dict[int, Controller] = field(default_factory=dict)


@dataclass
class Octave:
    connectivity: Union[str, tuple[str, int]]
    RF_outputs: dict[int, OctaveOutput] = field(default_factory=dict)
    RF_inputs: dict[int, OctaveInput] = field(default_factory=dict)

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
        return con, fem
    if "/" in id:
        con, fem = id.split("/")
        return con, int(fem)
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
