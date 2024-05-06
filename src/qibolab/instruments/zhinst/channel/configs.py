from dataclasses import dataclass

from qibolab.channel.configs import (
    AcquisitionChannelConfig,
    DCChannelConfig,
    IQChannelConfig,
)

__all__ = [
    "ZurichDCChannelConfig",
    "ZurichIQChannelConfig",
    "ZurichAcquisitionChannelConfig",
]


@dataclass(frozen=True)
class ZurichDCChannelConfig(DCChannelConfig):
    """DC channel config using ZI HDAWG."""

    power_range: float
    """Power range in volts.

    Possible values are [0.2 0.4 0.6 0.8 1. 2. 3. 4. 5.].
    """


@dataclass(frozen=True)
class ZurichIQChannelConfig(IQChannelConfig):
    """IQ channel config for ZI SHF* line instrument."""

    power_range: float
    """Power range in dBm.

    Possible values are [-30. -25. -20. -15. -10. -5. 0. 5. 10.].
    """


@dataclass(frozen=True)
class ZurichAcquisitionChannelConfig(AcquisitionChannelConfig):
    """Acquisition config for ZI SHF* line instrument."""

    power_range: float
    """Power range in dBm.

    Possible values are [-30. -25. -20. -15. -10. -5. 0. 5. 10.].
    """
