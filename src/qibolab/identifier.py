from enum import Enum
from typing import Annotated, Optional, Union

from pydantic import Field, TypeAdapter, model_serializer, model_validator

from .serialize import Model

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

    def __str__(self) -> str:
        return str(self.value)


class ChannelId(Model):
    """Unique identifier for a channel."""

    qubit: QubitId
    channel_type: ChannelType
    cross: Optional[str]

    @model_validator(mode="before")
    @classmethod
    def _load(cls, ch: str) -> dict:
        elements = ch.split("/")
        # TODO: replace with pattern matching, once py3.9 will be abandoned
        if len(elements) > 3:
            raise ValueError()
        q = TypeAdapter(QubitId).validate_python(elements[0])
        ct = ChannelType(elements[1])
        assert len(elements) == 2 or ct is ChannelType.DRIVE_CROSS
        dc = elements[2] if len(elements) == 3 else None
        return dict(qubit=q, channel_type=ct, cross=dc)

    @classmethod
    def load(cls, value: str):
        """Unpack from string."""
        return cls.model_validate(value)

    def __str__(self):
        """Represent as its joint components."""
        return "/".join(str(el[1]) for el in self if el[1] is not None)

    @model_serializer
    def _dump(self) -> str:
        """Prepare for serialization."""
        return str(self)
