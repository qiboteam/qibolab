from dataclasses import dataclass, field
from typing import Union

from typing_extensions import TypedDict

from qibolab._core.components import Channel

__all__ = ["DcElement", "RfOctaveElement", "AcquireOctaveElement", "Element"]


InOutType = Union[tuple[str, int], tuple[str, int, int]]
OctavePort = TypedDict("OpxPlusPort", {"port": tuple[str, int]})
Port = TypedDict("Port", {"port": InOutType})
ConnectivityType = Union[str, tuple[str, int]]


@dataclass(frozen=True)
class OutputSwitch:
    port: InOutType
    delay: int = 57
    buffer: int = 18
    """Default calibration parameters for digital pulses.

    https://docs.quantum-machines.co/1.1.7/qm-qua-sdk/docs/Guides/octave/#calibrating-the-digital-pulse

    Digital markers are used for LO triggering.
    """


def _to_port(channel: Channel) -> Port:
    """Convert a channel to the port dictionary required for the QUA config.

    The following syntax is assumed for ``channel.device``:
    * For OPX+ clusters: string with the device name (eg. 'con1')
    * For OPX1000 clusters: string of '{device_name}/{fem_number}'
    """
    if "/" not in channel.device:
        port = (channel.device, channel.port)
    else:
        con, fem = channel.device.split("/")
        port = (con, int(fem), channel.port)
    return {"port": port}


def output_switch(connectivity: ConnectivityType, port: int):
    """Create output switch section."""
    if isinstance(connectivity, tuple):
        args = connectivity + (2 * port - 1,)
    else:
        args = (connectivity, 2 * port - 1)
    return {"output_switch": OutputSwitch(args)}


@dataclass
class DcElement:
    singleInput: Port
    intermediate_frequency: int = 0
    operations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_channel(cls, channel: Channel):
        return cls(_to_port(channel))


DigitalInputs = TypedDict("digitalInputs", {"output_switch": OutputSwitch})


@dataclass
class RfOctaveElement:
    RF_inputs: OctavePort
    digitalInputs: DigitalInputs
    intermediate_frequency: int
    operations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_channel(
        cls,
        channel: Channel,
        connectivity: ConnectivityType,
        intermediate_frequency: int,
    ):
        return cls(
            _to_port(channel),
            output_switch(connectivity, channel.port),
            intermediate_frequency,
        )


@dataclass
class AcquireOctaveElement:
    RF_inputs: OctavePort
    RF_outputs: OctavePort
    digitalInputs: DigitalInputs
    intermediate_frequency: int
    time_of_flight: int = 24
    smearing: int = 0
    operations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_channel(
        cls,
        probe_channel: Channel,
        acquire_channel: Channel,
        connectivity: ConnectivityType,
        intermediate_frequency: int,
        time_of_flight: int,
        smearing: int,
    ):
        return cls(
            _to_port(probe_channel),
            _to_port(acquire_channel),
            output_switch(connectivity, probe_channel.port),
            intermediate_frequency,
            time_of_flight=time_of_flight,
            smearing=smearing,
        )


Element = Union[DcElement, RfOctaveElement, AcquireOctaveElement]
