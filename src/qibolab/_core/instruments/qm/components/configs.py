from typing import Literal, Union

import numpy as np

from qibolab._core.components import (
    AcquisitionConfig,
    DcConfig,
    OscillatorConfig,
)
from qibolab._core.components.filters import ExponentialFilter

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

DEFAULT_FEEDFORWARD_MAX = 2 - 2**-16
"""Maximum feedforward tap value"""
DEFAULT_FEEDBACK_MAX = 1 - 2**-20
"""Maximum feedback tap value"""


def normalize_feedforward(taps: list[float], threshold: float):
    max_value = np.max(np.abs(taps))
    if max_value > threshold:
        return threshold * np.array(taps / max_value).tolist()
    else:
        return taps


def normalize_feedback(taps: list[float], threshold: float):
    new_taps = np.array(taps)
    if np.any(np.abs(taps) > threshold):
        new_taps[new_taps > threshold] = threshold
        new_taps[new_taps < -threshold] = -threshold
    return new_taps.tolist()


class OpxOutputConfig(DcConfig):
    """DC channel config using QM OPX+."""

    kind: Literal["opx-output"] = "opx-output"

    offset: float = 0.0
    """DC offset to be applied in V.

    Possible values are -0.5V to 0.5V.
    """
    output_mode: Literal["direct", "amplified"] = "direct"
    sampling_rate: float = DEFAULT_SAMPLING_RATE
    upsampling_mode: Literal["mw", "pulse"] = "mw"
    feedback_max: float = DEFAULT_FEEDBACK_MAX
    feedforward_max: float = DEFAULT_FEEDFORWARD_MAX

    @property
    def filter(self):
        feedback_filters = [
            -i.feedback[1] for i in self.filters if isinstance(i, ExponentialFilter)
        ]
        return {
            "filter": {
                "feedback": normalize_feedback(feedback_filters, self.feedback_max)
                if len(feedback_filters) > 0
                else [],
                "feedforward": normalize_feedforward(
                    self.feedforward, self.feedforward_max
                )
                if len(self.feedforward) > 0
                else [],
            }
        }


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
