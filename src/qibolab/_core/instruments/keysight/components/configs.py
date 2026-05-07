from typing import Annotated, Literal

from pydantic import Field

from qibolab._core.components import AcquisitionConfig
from qibolab._core.serialize import NdArray

__all__ = ["QcsAcquisitionConfig"]


class QcsAcquisitionConfig(AcquisitionConfig):
    """Acquisition config for Keysight QCS."""

    kind: Literal["qcs-acquisition"] = "qcs-acquisition"

    state_iq_values: Annotated[NdArray | None, Field(repr=False)] = None
