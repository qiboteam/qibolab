from dataclasses import dataclass
from typing import Optional

from .ports import OctaveInput, OctaveOutput, OPXInput, OPXOutput, Ports


@dataclass
class Octave:
    name: str
    port: int
    connectivity: str

    outputs: Optional[Ports[int, OctaveOutput]] = None
    inputs: Optional[Ports[int, OctaveInput]] = None

    def __post_init__(self):
        self.outputs = Ports(OctaveOutput, self.name)
        self.inputs = Ports(OctaveInput, self.name)

    def ports(self, name, input=False):
        if input:
            return self.inputs[name]
        else:
            return self.outputs[name]


@dataclass
class OPXplus:
    number: int
    outputs: Optional[Ports[int, OPXOutput]] = None
    inputs: Optional[Ports[int, OPXInput]] = None

    @property
    def name(self):
        return f"con{self.number}"

    def __post_init__(self):
        self.outputs = Ports(OPXOutput, self.name)
        self.inputs = Ports(OPXInput, self.name)

    def ports(self, name, input=False):
        if input:
            return self.inputs[name]
        else:
            return self.outputs[name]
