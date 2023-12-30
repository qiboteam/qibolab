from dataclasses import asdict, dataclass, field
from typing import ClassVar, Dict, Optional, Union


@dataclass
class QMPort:
    device: str
    number: int

    key: ClassVar[Optional[str]] = None
    _replace: ClassVar[dict] = {}

    @property
    def pair(self):
        return (self.device, self.number)

    @property
    def serial(self):
        data = asdict(self)
        del data["device"]
        del data["number"]
        for old, new in self._replace.items():
            data[new] = data.pop(old)
        return {self.number: data}


@dataclass
class OPXOutput(QMPort):
    key: ClassVar[str] = "analog_outputs"

    offset: float = 0.0
    filter: Dict[str, float] = field(default_factory=dict)


@dataclass
class OPXInput(QMPort):
    key: ClassVar[str] = "analog_inputs"
    _replace: ClassVar[dict] = {"gain": "gain_db"}

    offset: float = 0.0
    gain: int = 0


@dataclass
class OPXIQ:
    i: Union[OPXOutput, OPXInput]
    q: Union[OPXOutput, OPXInput]


@dataclass
class OctaveOutput(QMPort):
    key: ClassVar[str] = "RF_outputs"
    _replace: ClassVar[dict] = {
        "lo_frequency": "LO_frequency",
        "lo_source": "LO_source",
    }

    lo_frequency: float = 0.0
    gain: float = 0
    """Can be in the range [-20 : 0.5 : 20]dB."""
    lo_source: str = "internal"
    """Can be external or internal."""
    output_mode: str = "always_on"
    """Can be: "always_on" / "always_off"/ "triggered" / "triggered_reversed"."""


@dataclass
class OctaveInput(QMPort):
    key: ClassVar[str] = "RF_inputs"
    _replace: ClassVar[dict] = {
        "lo_frequency": "LO_frequency",
        "lo_source": "LO_source",
    }

    lo_frequency: float = 0.0
    lo_source: str = "internal"
    IF_mode_I: str = "direct"
    IF_mode_Q: str = "direct"


class Ports(dict):
    def __init__(self, constructor, device):
        self.constructor = constructor
        self.device = device
        super().__init__()

    def __getitem__(self, number):
        if number not in self:
            self[number] = self.constructor(self.device, number)
        return super().__getitem__(number)
