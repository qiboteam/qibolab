from dataclasses import dataclass
from typing import Optional

from .ports import OctaveInput, OctaveOutput, OPXInput, OPXOutput, QMPort


class Ports(dict):
    def __init__(self, constructor, device):
        self.constructor = constructor
        self.device = device
        super().__init__()

    def __getitem__(self, number):
        if number not in self:
            self[number] = self.constructor(self.device, number)
        return super().__getitem__(number)


@dataclass
class QMDevice:
    output_type = QMPort
    input_type = QMPort

    name: str
    outputs: Optional[Ports[int, QMPort]] = None
    inputs: Optional[Ports[int, QMPort]] = None

    def __post_init__(self):
        self.outputs = Ports(self.output_type, self.name)
        self.inputs = Ports(self.input_type, self.name)

    def ports(self, number, input=False):
        if input:
            return self.inputs[number]
        else:
            return self.outputs[number]

    def setup(self, **kwargs):
        for number, settings in kwargs.items():
            if settings.pop("input", False):
                self.inputs[number].setup(**settings)
            else:
                self.outputs[number].setup(**settings)

    def dump(self):
        data = {port.name: port.settings for port in self.outputs}
        data.update(
            {port.name: port.settings | {"input": True} for port in self.inputs}
        )
        return data


@dataclass
class OPXplus(QMDevice):
    output_type = OPXOutput
    input_type = OPXInput


@dataclass
class Octave(QMDevice):
    output_type = OctaveOutput
    input_type = OctaveInput

    port: int = 0
    connectivity: Optional[str] = None
