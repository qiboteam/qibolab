from typing import Literal

from pydantic import Field

from qibolab.components import AcquisitionConfig, DcConfig

__all__ = [
    "OpxOutputConfig",
    "QmAcquisitionConfig",
]


class OpxOutputConfig(DcConfig):
    """DC channel config using QM OPX+."""

    kind: Literal["opx-output"] = "opx-output"

    offset: float = 0.0
    """DC offset to be applied in V.

    Possible values are -0.5V to 0.5V.
    """
    filter: dict[str, float] = Field(default_factory=dict)
    """FIR and IIR filters to be applied for correcting signal distortions.

    See
    https://docs.quantum-machines.co/1.1.7/qm-qua-sdk/docs/Guides/output_filter/?h=filter#output-filter
    for more details.
    Changing the filters affects the calibration of single shot discrimination (threshold and angle).
    """


class QmAcquisitionConfig(AcquisitionConfig):
    """Acquisition config for QM OPX+."""

    kind: Literal["qm-acquisition"] = "qm-acquisition"

    gain: int = 0
    """Input gain in dB.

    Possible values are -12dB to 20dB in steps of 1dB.
    """
    offset: float = 0.0
    """Constant voltage to be applied on the input."""
