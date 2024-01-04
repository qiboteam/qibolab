from dataclasses import dataclass, field, fields
from typing import ClassVar, Dict, Optional, Union


@dataclass
class QMPort:
    device: str
    number: int

    key: ClassVar[Optional[str]] = None

    @property
    def pair(self):
        return (self.device, self.number)

    def setup(self, **kwargs):
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise KeyError(f"Unknown port setting {name}.")
            setattr(self, name, value)

    @property
    def settings(self):
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.metadata.get("settings", False)
        }

    @property
    def config(self):
        data = {}
        for fld in fields(self):
            if "config" in fld.metadata:
                data[fld.metadata["config"]] = getattr(self, fld.name)
        return {self.number: data}


class QMOutput(QMPort):
    @property
    def name(self):
        return f"o{self.number}"


class QMInput(QMPort):
    @property
    def name(self):
        return f"i{self.number}"


@dataclass
class OPXOutput(QMOutput):
    key: ClassVar[str] = "analog_outputs"

    offset: float = field(default=0.0, metadata={"config": "offset"})
    filter: Dict[str, float] = field(
        default_factory=dict, metadata={"config": "filter", "settings": True}
    )

    @property
    def settings(self):
        data = super().settings
        if len(self.filter) == 0:
            del data["filter"]
        return data


@dataclass
class OPXInput(QMInput):
    key: ClassVar[str] = "analog_inputs"

    offset: float = field(default=0.0, metadata={"config": "offset"})
    gain: int = field(default=0, metadata={"config": "gain_db", "settings": True})


@dataclass
class OPXIQ:
    i: Union[OPXOutput, OPXInput]
    q: Union[OPXOutput, OPXInput]


@dataclass
class OctaveOutput(QMOutput):
    key: ClassVar[str] = "RF_outputs"

    lo_frequency: float = field(
        default=0.0, metadata={"config": "LO_frequency", "settings": True}
    )
    gain: int = field(default=0, metadata={"config": "gain", "settings": True})
    """Can be in the range [-20 : 0.5 : 20]dB."""
    lo_source: str = field(default="internal", metadata={"config": "LO_source"})
    """Can be external or internal."""
    output_mode: str = field(default="always_on", metadata={"config": "output_mode"})
    """Can be: "always_on" / "always_off"/ "triggered" / "triggered_reversed"."""

    opx_port: Optional[OPXOutput] = None


@dataclass
class OctaveInput(QMInput):
    key: ClassVar[str] = "RF_inputs"

    lo_frequency: float = field(
        default=0.0, metadata={"config": "LO_frequency", "settings": True}
    )
    lo_source: str = field(default="internal", metadata={"config": "LO_source"})
    IF_mode_I: str = field(default="direct", metadata={"config": "IF_mode_I"})
    IF_mode_Q: str = field(default="direct", metadata={"config": "IF_mode_Q"})

    opx_port: Optional[OPXOutput] = None
