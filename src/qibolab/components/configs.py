"""Configuration for various components.

These represent the minimal needed configuration that needs to be
exposed to users. Specific definitions of components can expose more,
and that can be used in any troubleshooting or debugging purposes by
users, but in general any user tool should try to depend only on the
configuration defined by these classes.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from qibolab.serialize import Model, NdArray

__all__ = [
    "DcConfig",
    "IqConfig",
    "AcquisitionConfig",
    "IqMixerConfig",
    "OscillatorConfig",
    "Config",
]


class DcConfig(Model):
    """Configuration for a channel that can be used to send DC pulses (i.e.
    just envelopes without modulation)."""

    kind: Literal["dc"] = "dc"

    offset: float
    """DC offset/bias of the channel."""


class OscillatorConfig(Model):
    """Configuration for an oscillator."""

    kind: Literal["oscillator"] = "oscillator"

    frequency: float
    power: float


class IqMixerConfig(Model):
    """Configuration for IQ mixer.

    Mixers usually have various imperfections, and one needs to
    compensate for them. This class holds the compensation
    configuration.
    """

    kind: Literal["iq-mixer"] = "iq-mixer"

    offset_i: float = 0.0
    """DC offset for the I component."""
    offset_q: float = 0.0
    """DC offset for the Q component."""
    scale_q: float = 1.0
    """The relative amplitude scale/factor of the q channel, to account for I-Q
    amplitude imbalance."""
    phase_q: float = 0.0
    """The phase offset of the q channel, to account for I-Q phase
    imbalance."""


class IqConfig(Model):
    """Configuration for an IQ channel."""

    kind: Literal["iq"] = "iq"

    frequency: float
    """The carrier frequency of the channel."""


class AcquisitionConfig(Model):
    """Configuration for acquisition channel.

    Currently, in qibolab, acquisition channels are FIXME:
    """

    kind: Literal["acquisition"] = "acquisition"

    delay: float
    """Delay between readout pulse start and acquisition start."""
    smearing: float
    """FIXME:"""

    # FIXME: this is temporary solution to deliver the information to drivers
    # until we make acquisition channels first class citizens in the sequences
    # so that each acquisition command carries the info with it.
    threshold: Optional[float] = None
    """Signal threshold for discriminating ground and excited states."""
    iq_angle: Optional[float] = None
    """Signal angle in the IQ-plane for disciminating ground and excited
    states."""
    kernel: Annotated[Optional[NdArray], Field(repr=False)] = None
    """Integration weights to be used when post-processing the acquired
    signal."""


Config = Annotated[
    Union[DcConfig, IqMixerConfig, OscillatorConfig, IqConfig, AcquisitionConfig],
    Field(discriminator="kind"),
]
