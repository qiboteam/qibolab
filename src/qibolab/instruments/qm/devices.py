from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain

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

    outputs: dict[int, QMOutput] = field(init=False)
    """Dictionary containing the instrument's output ports."""
    inputs: dict[int, QMInput] = field(init=False)
    """Dictionary containing the instrument's input ports."""

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
class Octave(QMDevice):
    """Device handling Octaves."""

    port: int
    """Network port of the Octave in the cluster configuration."""
    connectivity: OPXplus
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
            iport = self.connectivity.ports(2 * number - 1, output)
            qport = self.connectivity.ports(2 * number, output)
            port.opx_port = OPXIQ(iport, qport)
        return port
