from typing import Literal, Union

from pydantic import Field

from qibolab._core.components import (
    AcquisitionConfig,
    DcConfig,
    OscillatorConfig,
)

__all__ = [
    "OpxOutputConfig",
    "QmAcquisitionConfig",
    "QmConfigs",
    "OctaveOscillatorConfig",
    "MwFemOscillatorConfig",
]

OctaveOutputModes = Literal[
    "always_on", "always_off", "triggered", "triggered_reversed"
]

DEFAULT_SAMPLING_RATE = 1e9


class OpxOutputConfig(DcConfig):
    """DC channel config using QM OPX+."""

    kind: Literal["opx-output"] = "opx-output"

    offset: float = 0.0
    """DC offset to be applied in V.

    Possible values are -0.5V to 0.5V.
    """
    filter: dict[str, list[float]] = Field(default_factory=dict)
    """FIR and IIR filters to be applied for correcting signal distortions.

    See
    https://docs.quantum-machines.co/1.1.7/qm-qua-sdk/docs/Guides/output_filter/?h=filter#output-filter
    for more details.
    Changing the filters affects the calibration of single shot discrimination (threshold and angle).
    """
    output_mode: Literal["direct", "amplified"] = "direct"
    sampling_rate: float = DEFAULT_SAMPLING_RATE
    upsampling_mode: Literal["mw", "pulse"] = "mw"


class OctaveOscillatorConfig(OscillatorConfig):
    """Oscillator confing that allows switching the output mode."""

    kind: Literal["octave-oscillator"] = "octave-oscillator"

    output_mode: OctaveOutputModes = "triggered"


class QmAcquisitionConfig(AcquisitionConfig):
    """Acquisition config for QM OPX+."""

    kind: Literal["qm-acquisition"] = "qm-acquisition"

    gain: int = 0
    """Input gain in dB.

    Possible values are -12dB to 20dB in steps of 1dB.
    """
    offset: float = 0.0
    """Constant voltage to be applied on the input."""


class MwFemOscillatorConfig(OscillatorConfig):
    """Output config for OPX1000 MW-FEM ports.

    For more information see
    https://docs.quantum-machines.co/latest/docs/Guides/opx1000_fems/?h=upsampl#microwave-fem-mw-fem
    """

    kind: Literal["mw-fem-oscillator"] = "mw-fem-oscillator"

    power: int = -11
    """This corresponds to the ``full_scale_power_dbm`` setting."""
    upconverter: int = 1
    band: int = 2
    sampling_rate: float = DEFAULT_SAMPLING_RATE


QmConfigs = Union[
    OpxOutputConfig,
    OctaveOscillatorConfig,
    QmAcquisitionConfig,
    MwFemOscillatorConfig,
]
