from enum import Enum
from typing import Annotated, Union

from pydantic import BeforeValidator, Field, PlainSerializer

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Qubit name."""

QubitPairId = Annotated[
    tuple[QubitId, QubitId],
    BeforeValidator(lambda p: tuple(p.split("-")) if isinstance(p, str) else p),
    PlainSerializer(lambda p: f"{p[0]}-{p[1]}"),
]
"""Type for holding ``QubitPair``s in the ``platform.pairs`` dictionary."""


# TODO: replace with StrEnum, once py3.10 will be abandoned
# at which point, it will also be possible to replace values with auto()
class ChannelType(str, Enum):
    """Names of channels that belong to a qubit.

    Not all channels are required to operate a qubit.
    """

    PROBE = "probe"
    ACQUISITION = "acquisition"
    DRIVE = "drive"
    FLUX = "flux"

    def __str__(self) -> str:
        return str(self.value)


ChannelId = str
"""Unique identifier for a channel."""


StateId = int
"""State identifier."""


def _split(pair: Union[str, tuple]) -> tuple[str, str]:
    if isinstance(pair, str):
        a, b = pair.split("-")
        return a, b
    return pair


def _join(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


TransitionId = Annotated[
    tuple[StateId, StateId], BeforeValidator(_split), PlainSerializer(_join)
]
"""Identifier for a state transition."""

QubitPairId = Annotated[
    tuple[QubitId, QubitId], BeforeValidator(_split), PlainSerializer(_join)
]
"""Two-qubit active interaction identifier."""
