"""Configuration for various components.

These represent the minimal needed configuration that needs to be
exposed to users. Specific definitions of components can expose more,
and that can be used in any troubleshooting or debugging purposes by
users, but in general any user tool should try to depend only on the
configuration defined by these classes.
"""

from functools import reduce
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from ..serialize import Model, NdArray, eq
from .filters import Filter

__all__ = [
    "AcquisitionConfig",
    "ChannelConfig",
    "Config",
    "Configs",
    "DcConfig",
    "IqConfig",
    "LogConfig",
    "MixerOffsetConfig",
    "OscillatorConfig",
]


class Config(Model):
    """Configuration values depot."""


Configs = dict[str, Config]
"""Configuration database."""


class DcConfig(Config):
    """Configuration for a channel that can be used to send DC pulses (i.e.
    just envelopes without modulation)."""

    kind: Literal["dc"] = "dc"

    offset: float
    """DC offset/bias of the channel."""
    filters: list[Filter] = Field(default_factory=list)
    """List of filters."""

    @property
    def feedback(self) -> list[float]:
        feedback_coeff = [i.feedback for i in self.filters if i is not None]
        return reduce(np.convolve, feedback_coeff, [1])

    @property
    def feedforward(self) -> list[float]:
        feedforward_coeff = [i.feedforward for i in self.filters if i is not None]
        if len(feedforward_coeff) == 0:
            return []
        return reduce(np.convolve, feedforward_coeff)


class OscillatorConfig(Config):
    """Configuration for an oscillator."""

    kind: Literal["oscillator"] = "oscillator"

    frequency: float
    power: float


class MixerOffsetConfig(Config):
    """Per-port IQ mixer DC offsets for LO-leakage suppression.

    A single mixer is shared by all channels on the same RF port, so these
    frequency-independent offsets are defined once per port.
    """

    kind: Literal["mixer-offset"] = "mixer-offset"

    offset_i: float = 0.0
    """DC offset applied to the I component [mV], to suppress LO leakage."""
    offset_q: float = 0.0
    """DC offset applied to the Q component [mV], to suppress LO leakage."""


class IqConfig(Config):
    """Per-channel IQ modulation configuration.

    Holds the carrier frequency of the channel together with the frequency-dependent
    sideband corrections.
    """

    kind: Literal["iq"] = "iq"

    frequency: float
    """The carrier frequency of the channel."""
    scale_q: float = 1.0
    """a dimensionless ratio equal to the Q-channel amplitude divided by the I-channel
    amplitude, correcting I-Q amplitude imbalance."""
    phase_q: float = 0.0
    """Phase offset of the Q channel [rad], correcting I-Q phase imbalance."""


class AcquisitionConfig(Config):
    """Acquisition timing and optional integration or discrimination parameters."""

    kind: Literal["acquisition"] = "acquisition"

    delay: float
    """Delay between readout pulse start and acquisition start, in ns."""
    smearing: float
    """Acquisition timing margin in ns, interpreted by the platform integration."""

    # FIXME: this is temporary solution to deliver the information to drivers
    # until we make acquisition channels first class citizens in the sequences
    # so that each acquisition command carries the info with it.
    threshold: float | None = None
    """Signal threshold for discriminating ground and excited states."""
    iq_angle: float | None = None
    """Signal angle in the IQ-plane for disciminating ground and excited
    states."""
    kernel: Annotated[NdArray | None, Field(repr=False)] = None
    """Integration weights to be used when post-processing the acquired
    signal."""

    def __eq__(self, other) -> bool:
        return eq(self, other)


class LogConfig(Config):
    """Configuration for logging."""

    kind: Literal["log"] = "log"

    path: Path


ChannelConfig = (
    DcConfig
    | MixerOffsetConfig
    | OscillatorConfig
    | IqConfig
    | AcquisitionConfig
    | LogConfig
)
