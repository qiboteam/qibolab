from dataclasses import dataclass
from itertools import chain
from typing import Optional

from qibolab.instruments.abstract import Instrument

from .ports import (
    OPXIQ,
    OctaveInput,
    OctaveOutput,
    OPXInput,
    OPXOutput,
    QMInput,
    QMOutput,
)


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
class QMDevice(Instrument):
    name: str
    port: Optional[int] = None
    connectivity: Optional["QMDevice"] = None

    outputs: Optional[Ports[int, QMOutput]] = None
    inputs: Optional[Ports[int, QMInput]] = None

    def ports(self, number, input=False):
        if input:
            return self.inputs[number]
        else:
            return self.outputs[number]

    def connect(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def start(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def setup(self, **kwargs):
        for name, settings in kwargs.items():
            number = int(name[1:])
            if name[0] == "o":
                self.outputs[number].setup(**settings)
            elif name[0] == "i":
                self.inputs[number].setup(**settings)
            else:
                raise ValueError(
                    f"Invalid port name {name} in instrument settings for {self.name}."
                )

    def stop(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def disconnect(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def dump(self):
        ports = chain(self.outputs.values(), self.inputs.values())
        return {port.name: port.settings for port in ports if len(port.settings) > 0}


@dataclass
class OPXplus(QMDevice):
    def __post_init__(self):
        self.outputs = Ports(OPXOutput, self.name)
        self.inputs = Ports(OPXInput, self.name)


@dataclass
class Octave(QMDevice):
    def __post_init__(self):
        self.outputs = Ports(OctaveOutput, self.name)
        self.inputs = Ports(OctaveInput, self.name)

    def ports(self, number, input=False):
        port = super().ports(number, input)
        if port.opx_port is None:
            iport = self.connectivity.ports(2 * number - 1, input)
            qport = self.connectivity.ports(2 * number, input)
            port.opx_port = OPXIQ(iport, qport)
        return port
