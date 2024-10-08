from qibolab._core.components import AcquisitionConfig, DcConfig, IqConfig

__all__ = [
    "ZiDcConfig",
    "ZiIqConfig",
    "ZiAcquisitionConfig",
]


class ZiDcConfig(DcConfig):
    """DC channel config using ZI HDAWG."""

    power_range: float
    """Power range in volts.

    Possible values are [0.2 0.4 0.6 0.8 1. 2. 3. 4. 5.].
    """


class ZiIqConfig(IqConfig):
    """IQ channel config for ZI SHF* line instrument."""

    power_range: float
    """Power range in dBm.

    Possible values are [-30. -25. -20. -15. -10. -5. 0. 5. 10.].
    """


class ZiAcquisitionConfig(AcquisitionConfig):
    """Acquisition config for ZI SHF* line instrument."""

    power_range: float = 0
    """Power range in dBm.

    Possible values are [-30. -25. -20. -15. -10. -5. 0. 5. 10.].
    """
