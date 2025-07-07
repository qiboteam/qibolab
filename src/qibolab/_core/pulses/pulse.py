"""Pulse class."""

from typing import Annotated, Literal, Union, cast
from uuid import uuid4

import numpy as np
from pydantic import UUID4, Field

from ..serialize import Model
from .envelope import Envelope, IqWaveform, Waveform

__all__ = [
    "Acquisition",
    "Align",
    "Delay",
    "Pulse",
    "PulseId",
    "PulseLike",
    "Readout",
    "VirtualZ",
]

PulseId = UUID4
"""Unique identifier for a pulse."""


class _PulseLike(Model):
    id_: PulseId = Field(default_factory=uuid4, exclude=True)

    @property
    def id(self) -> PulseId:
        """Instruction identifier."""
        return self.id_

    def new(self) -> "PulseLike":
        return cast(PulseLike, self.model_copy(deep=True, update={"id_": uuid4()}))

    def __eq__(self, other: object) -> bool:
        """Compare instances."""
        # TODO: for the time being, Pydantic does not support fields
        # inclusion/exclusion in comparison, otherwise it would be much better to
        # exclude them rather then applying this recursive definition
        # https://github.com/pydantic/pydantic/discussions/6717
        s = vars(self)
        o = vars(other)
        return isinstance(other, type(self)) and all(
            s[k] == o[k] for k in s if k != "id_"
        )

    def __hash__(self) -> int:
        return hash(tuple(v for k, v in vars(self).items() if k != "id_"))


class Pulse(_PulseLike):
    """A pulse to be sent to the QPU.

    Valid on any channel, except acquisition ones.
    """

    kind: Literal["pulse"] = "pulse"

    duration: float
    """Pulse duration."""

    amplitude: float
    """Pulse digital amplitude (unitless).

    Pulse amplitudes are normalised between -1 and 1.
    """
    envelope: Envelope
    """The pulse envelope shape.

    See :class:`qibolab.Envelope` for list of available shapes.
    """
    relative_phase: float = 0.0
    """Relative phase of the pulse, in radians."""

    def i(self, sampling_rate: float) -> Waveform:
        """Compute the envelope of the waveform i component."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.i(samples)

    def q(self, sampling_rate: float) -> Waveform:
        """Compute the envelope of the waveform q component."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.q(samples)

    def envelopes(self, sampling_rate: float) -> IqWaveform:
        """Compute a tuple with the i and q envelopes."""
        return np.array([self.i(sampling_rate), self.q(sampling_rate)])


class Delay(_PulseLike):
    """Wait instruction.

    During its length no pulse is sent on the same channel.

    Valid on any channel.
    """

    kind: Literal["delay"] = "delay"

    duration: float
    """Duration in ns."""


class VirtualZ(_PulseLike):
    """Implementation of Z-rotations using virtual phase.

    Only valid on a drive channel.
    """

    kind: Literal["virtualz"] = "virtualz"

    phase: float
    """Phase that implements the rotation."""

    @property
    def duration(self):
        """Duration of the virtual gate should always be zero."""
        return 0


class Acquisition(_PulseLike):
    """Acquisition instruction.

    This event instructs the device to acquire samples for the event
    span.

    Only valid on an acquisition channel.
    """

    kind: Literal["acquisition"] = "acquisition"

    duration: float
    """Duration in ns."""


class Readout(_PulseLike):
    """Readout instruction.

    This event instructs the device to acquire samples for the event
    span.

    Only valid on an acquisition channel.
    """

    kind: Literal["readout"] = "readout"

    acquisition: Acquisition
    probe: Pulse

    @classmethod
    def from_probe(cls, probe: Pulse):
        """Create a whole readout operation from its probe pulse.

        The acquisition is made to match the same probe duration.
        """
        return cls(acquisition=Acquisition(duration=probe.duration), probe=probe)

    @property
    def duration(self) -> float:
        """Duration in ns."""
        return self.acquisition.duration

    @property
    def id(self) -> PulseId:
        """Instruction identifier."""
        return self.acquisition.id

    def new(self) -> "PulseLike":
        return cast(
            PulseLike,
            self.model_copy(
                deep=True,
                update={
                    "id_": uuid4(),
                    "acquisition": self.acquisition.new(),
                    "probe": self.probe.new(),
                },
            ),
        )


class Align(_PulseLike):
    """Brings different channels at the same point in time."""

    kind: Literal["align"] = "align"


PulseLike = Annotated[
    Union[Align, Pulse, Delay, VirtualZ, Acquisition, Readout],
    Field(discriminator="kind"),
]
