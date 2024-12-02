from typing import Literal

from qibolab._core.components import IqConfig

__all__ = ["QbloxConfigs", "QbloxIq"]


class QbloxIq(IqConfig):
    """Microwave output channel config using Qblox."""

    kind: Literal["qblox-iq"] = "qblox-iq"

    attenuation: float = 0.0
    """Output attenuation in dB.

    For the specific case of ``out0``, cf.

    - https://docs.qblox.com/en/main/api_reference/module.html#QCM_RF.out0_att
    """


QbloxConfigs = QbloxIq
