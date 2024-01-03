from dataclasses import dataclass
from typing import Optional

from qibolab.instruments.abstract import Instrument

from .ports import OPXIQ, OctaveInput, OctaveOutput, OPXInput, OPXOutput, QMPort


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

    outputs: Optional[Ports[int, QMPort]] = None
    inputs: Optional[Ports[int, QMPort]] = None

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

    def setup(self, port_settings=None):
        if port_settings is not None:
            for number, settings in port_settings.items():
                if settings.pop("input", False):
                    self.inputs[number].setup(**settings)
                else:
                    self.outputs[number].setup(**settings)

    def stop(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def disconnect(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def dump(self):
        data = {port.number: port.settings for port in self.outputs.values()}
        data.update(
            {
                port.number: port.settings | {"input": True}
                for port in self.inputs.values()
            }
        )
        return data


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
