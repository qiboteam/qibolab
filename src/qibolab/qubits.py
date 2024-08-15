from typing import Annotated, Literal, Optional, Union, get_args

from pydantic import BeforeValidator, ConfigDict, Field, PlainSerializer

from qibolab.components import AcquireChannel, DcChannel, IqChannel
from qibolab.serialize import Model

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Type for qubit names."""

ChannelName = Literal["probe", "acquisition", "drive", "drive12", "drive_cross", "flux"]
"""Names of channels that belong to a qubit.

Not all channels are required to operate a qubit.
"""

ChannelId = tuple[QubitId, ChannelName, Optional[str]]
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
        for name in get_args(ChannelName):
            channel = getattr(self, name)
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
