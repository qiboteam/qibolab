from dataclasses import dataclass, replace
from typing import Optional

from qibo.config import raise_error

from qibolab.channel_config import (
    AcquisitionChannelConfig,
    DCChannelConfig,
    IQChannelConfig,
)


def check_max_offset(offset, max_offset):
    """Checks if a given offset value exceeds the maximum supported offset.

    This is to avoid sending high currents that could damage lab
    equipment such as amplifiers.
    """
    if max_offset is not None and abs(offset) > max_offset:
        raise_error(
            ValueError, f"{offset} exceeds the maximum allowed offset {max_offset}."
        )


@dataclass
class DCChannel:
    name: str
    config: DCChannelConfig

    max_offset: Optional[float] = None
    """Maximum DC voltage that we can safely send through this channel.

    Sending high voltages for prolonged times may damage amplifiers or
    other lab equipment. If the user attempts to send a higher value an
    error will be raised to prevent execution in real instruments.
    """

    @property
    def offset(self):
        """DC offset that is applied to this port."""
        return self.config.offset

    @offset.setter
    def offset(self, value):
        check_max_offset(value, self.max_offset)
        self.config = replace(self.config, offset=value)


@dataclass
class IQChannel:
    name: str
    config: IQChannelConfig


@dataclass
class AcquisitionChannel:
    name: str
    config: AcquisitionChannelConfig
