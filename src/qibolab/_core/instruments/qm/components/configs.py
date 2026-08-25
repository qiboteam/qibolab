from functools import reduce
from typing import Any, Literal

import numpy as np
from pydantic import Field

from qibolab._core.components import (
    AcquisitionConfig,
    DcConfig,
    OscillatorConfig,
)
from qibolab._core.components.filters import ExponentialFilter

__all__ = [
    "MwFemOscillatorConfig",
    "OctaveOscillatorConfig",
    "OpxOutputConfig",
    "QmAcquisitionConfig",
    "QmConfigs",
]

OctaveOutputModes = Literal[
    "always_on", "always_off", "triggered", "triggered_reversed"
]

DEFAULT_SAMPLING_RATE = 1e9

DEFAULT_FEEDFORWARD_MAX = 2 - 2**-16
"""Maximum feedforward tap value"""
DEFAULT_FEEDBACK_MAX = 1 - 2**-20
"""Maximum feedback tap value"""


def normalize_feedforward(taps: list[float], threshold: float) -> list[float]:
    """Feedforward coefficient normalization required by QM."""
    scale = np.max(np.abs(taps) / threshold, initial=1)
    return (np.array(taps) / scale).tolist()


def normalize_feedback(taps: list[float], threshold: float) -> list[float]:
    """Feedback coefficient normalization required by QM."""
    new_taps = np.clip(taps, -threshold, threshold)
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
    feedback_max: float = Field(exclude=True, default=DEFAULT_FEEDBACK_MAX)
    feedforward_max: float = Field(exclude=True, default=DEFAULT_FEEDFORWARD_MAX)

    def filter(self, cluster: str) -> dict[str, list[float | tuple[float, float]]]:
        if cluster == "opx1":
            return self._opx1_filter()
        if cluster in {"opx1000", "LF", "MW"}:
            return self._opx1000_filter()
        raise NotImplementedError(f"Cluster type {cluster} not yet supported")

    def _opx1_filter(self) -> dict[str, list[float | tuple[float, float]]]:
        """Digital filters for the (legacy) OPX+.

        This generation has no dedicated hardware stage for the
        exponential correction, so it is folded into the generic FIR
        ``feedforward`` (and its pole into ``feedback``) together with
        any other configured filter.
        """
        feedback_filters = [
            -i.feedback[1] for i in self.filters if isinstance(i, ExponentialFilter)
        ]
        return {
            "feedforward": normalize_feedforward(self.feedforward, self.feedforward_max)
            if len(self.feedforward) > 0
            else [],
            "feedback": normalize_feedback(feedback_filters, self.feedback_max)
            if len(feedback_filters) > 0
            else [],
        }

    def _opx1000_filter(self) -> dict[str, list[float | tuple[float, float]]]:
        """Digital filters for OPX1000 (LF/MW FEM) clusters.

        Unlike the OPX+, these expose a native ``exponential`` IIR
        stage. ``ExponentialFilter``s are thus excluded from the FIR
        ``feedforward`` convolution, to avoid double-applying the same
        correction through both stages.
        """
        fir_filters = [
            f for f in self.filters if isinstance(f, FiniteImpulseResponseFilter)
        ]
        feedforward = (
            reduce(np.convolve, [f.feedforward for f in fir_filters])
            if len(fir_filters) > 0
            else []
        )
        return {
            "feedforward": normalize_feedforward(feedforward, self.feedforward_max)
            if len(feedforward) > 0
            else [],
            "exponential": [
                (filt.amplitude, filt.tau)
                for filt in self.filters
                if isinstance(filt, ExponentialFilter)
            ],
        }


class OctaveOscillatorConfig(OscillatorConfig):
    """Oscillator confing that allows switching the output mode."""

    kind: Literal["octave-oscillator"] = "octave-oscillator"

    output_mode: OctaveOutputModes = "triggered"


class QmAcquisitionConfig(AcquisitionConfig):
    """Acquisition config for QM."""

    kind: Literal["qm-acquisition"] = "qm-acquisition"

    gain: int = 0
    """Input gain in dB.

    Possible values are -12dB to 20dB in steps of 1dB.
    """
    offset: float = 0.0
    """Constant voltage to be applied on the input."""

    def model_post_init(self, context: Any) -> None:
        # The minimum time-of-flight for QM is 28 ns, so we need to ensure that
        # the delay is at least 28 ns (determined from QM error message during
        # execution)
        if self.delay < 28:
            object.__setattr__(self, "delay", 28)


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


QmConfigs = (
    OpxOutputConfig
    | OctaveOscillatorConfig
    | QmAcquisitionConfig
    | MwFemOscillatorConfig
)
