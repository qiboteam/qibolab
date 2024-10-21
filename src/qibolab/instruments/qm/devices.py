from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Literal, Union

from qibolab.instruments.abstract import Instrument

from .ports import (
    OPXIQ,
    FEMInput,
    FEMOutput,
    OctaveInput,
    OctaveOutput,
    OPXInput,
    OPXOutput,
    QMInput,
    QMOutput,
)


class PortsDefaultdict(defaultdict):
    """Dictionary mapping port numbers to
    :class:`qibolab.instruments.qm.ports.QMPort` objects.

    Automatically instantiates ports that have not yet been created.
    Used by :class:`qibolab.instruments.qm.devices.QMDevice`

    https://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
    """

    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)  # pylint: disable=E1102
        return ret


@dataclass
class QMDevice(Instrument):
    """Abstract class for an individual Quantum Machines devices."""

    name: str
    """Name of the device."""

    outputs: Dict[int, QMOutput] = field(init=False)
    """Dictionary containing the instrument's output ports."""
    inputs: Dict[int, QMInput] = field(init=False)
    """Dictionary containing the instrument's input ports."""

    def __str__(self):
        return self.name

    def ports(self, number, output=True):
        """Provides instrument's ports to the user.

        Args:
            number (int): Port number.
                Can be 1 to 10 for :class:`qibolab.instruments.qm.devices.OPXplus`
                and 1 to 5 for :class:`qibolab.instruments.qm.devices.Octave`.
            output (bool): ``True`` for obtaining an output port, otherwise an
                input port is returned. Default is ``True``.
        """
        ports_ = self.outputs if output else self.inputs
        return ports_[number]

    def connect(self):
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

    def disconnect(self):
        """Only applicable for
        :class:`qibolab.instruments.qm.controller.QMController`, not individual
        devices."""

    def dump(self):
        """Serializes device settings to a dictionary for dumping to the
        runcard YAML."""
        ports = chain(self.outputs.values(), self.inputs.values())
        return {port.name: port.settings for port in ports if len(port.settings) > 0}


@dataclass
class OPXplus(QMDevice):
    """Device handling OPX+ controllers."""

    def __post_init__(self):
        self.outputs = PortsDefaultdict(lambda n: OPXOutput(self.name, n))
        self.inputs = PortsDefaultdict(lambda n: OPXInput(self.name, n))


@dataclass
class FEM:
    """Device handling OPX1000 FEMs."""

    name: int
    type: Literal["LF", "MF"] = "LF"


@dataclass
class OPX1000(QMDevice):
    """Device handling OPX1000 controllers."""

    fems: Dict[int, FEM] = field(default_factory=dict)

    def __post_init__(self):
        def kwargs(fem):
            return {"fem_number": fem, "fem_type": self.fems[fem].type}

        self.outputs = PortsDefaultdict(
            lambda pair: FEMOutput(self.name, pair[1], **kwargs(pair[0]))
        )
        self.inputs = PortsDefaultdict(
            lambda pair: FEMInput(self.name, pair[1], **kwargs(pair[0]))
        )

    def ports(self, fem_number: int, number: int, output: bool = True):
        ports_ = self.outputs if output else self.inputs
        return ports_[(fem_number, number)]

    def connectivity(self, fem_number: int) -> tuple["OPX1000", int]:
        return (self, fem_number)

    def setup(self, **kwargs):
        for name, settings in kwargs.items():
            fem, port = name.split("/")
            fem = int(fem)
            number = int(port[1:])
            if port[0] == "o":
                self.outputs[(fem, number)].setup(**settings)
            elif port[0] == "i":
                self.inputs[(fem, number)].setup(**settings)
            else:
                raise ValueError(
                    f"Invalid port name {name} in instrument settings for {self.name}."
                )


@dataclass
class Octave(QMDevice):
    """Device handling Octaves."""

    port: int
    """Network port of the Octave in the cluster configuration."""
    connectivity: Union[OPXplus, tuple[OPX1000, int]]
    """OPXplus that acts as the waveform generator for the Octave."""

    def __post_init__(self):
        self.outputs = PortsDefaultdict(lambda n: OctaveOutput(self.name, n))
        self.inputs = PortsDefaultdict(lambda n: OctaveInput(self.name, n))

    def ports(self, number, output=True):
        """Provides Octave ports.

        Extension of the abstract :meth:`qibolab.instruments.qm.devices.QMDevice.ports`
        because Octave ports are used for mixing two existing (I, Q) OPX+ ports.
        """
        port = super().ports(number, output)
        if port.opx_port is None:
            if isinstance(self.connectivity, OPXplus):
                iport = self.connectivity.ports(2 * number - 1, output)
                qport = self.connectivity.ports(2 * number, output)
            else:
                opx, fem_number = self.connectivity
                iport = opx.ports(fem_number, 2 * number - 1, output)
                qport = opx.ports(fem_number, 2 * number, output)
            port.opx_port = OPXIQ(iport, qport)
        return port
