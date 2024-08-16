from enum import Enum
from typing import Annotated, Optional, Union

from pydantic import BeforeValidator, ConfigDict, Field, PlainSerializer, TypeAdapter

from qibolab.components import AcquireChannel, DcChannel, IqChannel
from qibolab.serialize import Model

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Type for qubit names."""


# TODO: replace with StrEnum, once py3.10 will be abandoned
# at which point, it will also be possible to replace values with auto()
class ChannelType(str, Enum):
    """Names of channels that belong to a qubit.

    Not all channels are required to operate a qubit.
    """

    PROBE = "probe"
    ACQUISITION = "acquisition"
    DRIVE = "drive"
    DRIVE12 = "drive12"
    DRIVE_CROSS = "drive_cross"
    FLUX = "flux"


def _str_to_chid(ch: str) -> "ChannelId":
    elements = ch.split("/")
    # TODO: replace with pattern matching, once py3.9 will be abandoned
    if len(elements) > 3:
        raise ValueError()
    q = TypeAdapter(QubitId).validate_python(elements[0])
    ct = ChannelType(elements[1])
    cross = elements[2] if len(elements) == 3 else None
    return (q, ct, cross)


ChannelId = Annotated[
    tuple[QubitId, ChannelType, Optional[str]],
    BeforeValidator(_str_to_chid),
    PlainSerializer(lambda ch: "/".join(str(el) for el in ch)),
]
"""Unique identifier for a channel."""


class Qubit(Model):
    """Representation of a physical qubit.

    Qubit objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.

    Args:
        name (int, str): Qubit number or name.
        readout (:class:`qibolab.platforms.utils.Channel`): Channel used to
            readout pulses to the qubit.
        drive (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send drive pulses to the qubit.
        flux (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send flux pulses to the qubit.
    """

    model_config = ConfigDict(frozen=False)

    name: QubitId

    probe: Optional[IqChannel] = None
    acquisition: Optional[AcquireChannel] = None
    drive: Optional[IqChannel] = None
    drive12: Optional[IqChannel] = None
    drive_cross: Optional[dict[QubitId, IqChannel]] = None
    flux: Optional[DcChannel] = None

    @property
    def channels(self):
        for ct in ChannelType:
            channel = getattr(self, ct.value)
            if channel is not None:
                yield channel

    @property
    def mixer_frequencies(self):
        """Get local oscillator and intermediate frequencies of native gates.

        Assumes RF = LO + IF.
        """
        freqs = {}
        for name in self.native_gates.model_fields:
            native = getattr(self.native_gates, name)
            if native is not None:
                channel_type = native.pulse_type.name.lower()
                _lo = getattr(self, channel_type).lo_frequency
                _if = native.frequency - _lo
                freqs[name] = _lo, _if
        return freqs


QubitPairId = Annotated[
    tuple[QubitId, QubitId],
    BeforeValidator(lambda p: tuple(p.split("-")) if isinstance(p, str) else p),
    PlainSerializer(lambda p: f"{p[0]}-{p[1]}"),
]
"""Type for holding ``QubitPair``s in the ``platform.pairs`` dictionary."""
