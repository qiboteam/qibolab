"""Pulse class."""

from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

import numpy as np

from .shape import SAMPLING_RATE, PulseShape, Waveform


class PulseType(Enum):
    """An enumeration to distinguish different types of pulses.

    READOUT pulses triger acquisitions. DRIVE pulses are used to control
    qubit states. FLUX pulses are used to shift the frequency of flux
    tunable qubits and with it implement two-qubit gates.
    """

    READOUT = "ro"
    DRIVE = "qd"
    FLUX = "qf"
    COUPLERFLUX = "cf"


@dataclass
class Pulse:
    """A class to represent a pulse to be sent to the QPU."""

    start: int
    """Start time of pulse in ns."""
    duration: int
    """Pulse duration in ns."""
    amplitude: float
    """Pulse digital amplitude (unitless).

    Pulse amplitudes are normalised between -1 and 1.
    """
    frequency: int
    """Pulse Intermediate Frequency in Hz.

    The value has to be in the range [10e6 to 300e6].
    """
    relative_phase: float
    """Relative phase of the pulse, in radians."""
    shape: PulseShape
    """Pulse shape, as a PulseShape object.

    See
    :py: mod:`qibolab.pulses` for list of available shapes.
    """
    channel: Optional[str] = None
    """Channel on which the pulse should be played.

    When a sequence of pulses is sent to the platform for execution,
    each pulse is sent to the instrument responsible for playing pulses
    the pulse channel. The connection of instruments with channels is
    defined in the platform runcard.
    """
    type: PulseType = PulseType.DRIVE
    """Pulse type, as an element of PulseType enumeration."""
    qubit: int = 0
    """Qubit or coupler addressed by the pulse."""

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = PulseType(self.type)
        if isinstance(self.shape, str):
            self.shape = PulseShape.eval(self.shape)
        # TODO: drop the cyclic reference
        self.shape.pulse = self

    @classmethod
    def flux(cls, start, duration, amplitude, shape, **kwargs):
        return cls(
            start, duration, amplitude, 0, 0, shape, type=PulseType.FLUX, **kwargs
        )

    @property
    def finish(self) -> Optional[int]:
        """Time when the pulse is scheduled to finish."""
        if None in {self.start, self.duration}:
            return None
        return self.start + self.duration

    @property
    def global_phase(self):
        """Global phase of the pulse, in radians.

        This phase is calculated from the pulse start time and frequency
        as `2 * pi * frequency * start`.
        """
        if self.type is PulseType.READOUT:
            # readout pulses should have zero global phase so that we can
            # calculate probabilities in the i-q plane
            return 0

        # pulse start, duration and finish are in ns
        return 2 * np.pi * self.frequency * self.start / 1e9

    @property
    def phase(self) -> float:
        """Total phase of the pulse, in radians.

        The total phase is computed as the sum of the global and
        relative phases.
        """
        return self.global_phase + self.relative_phase

    @property
    def id(self) -> int:
        return id(self)

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        return self.shape.envelope_waveform_i(sampling_rate)

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        return self.shape.envelope_waveform_q(sampling_rate)

    def envelope_waveforms(self, sampling_rate=SAMPLING_RATE):
        """A tuple with the i and q envelope waveforms of the pulse."""

        return (
            self.shape.envelope_waveform_i(sampling_rate),
            self.shape.envelope_waveform_q(sampling_rate),
        )

    def __hash__(self):
        """Hash the content.

        .. warning::

            unhashable attributes are not taken into account, so there will be more
            clashes than those usually expected with a regular hash

        .. todo::

            This method should be eventually dropped, and be provided automatically by
            freezing the dataclass (i.e. setting ``frozen=true`` in the decorator).
            However, at the moment is not possible nor desired, because it contains
            unhashable attributes and because some instances are mutated inside Qibolab.
        """
        return hash(
            tuple(
                getattr(self, f.name)
                for f in fields(self)
                if f.name not in ("type", "shape")
            )
        )

    def is_equal_ignoring_start(self, item) -> bool:
        """Check if two pulses are equal ignoring start time."""
        return (
            self.duration == item.duration
            and self.amplitude == item.amplitude
            and self.frequency == item.frequency
            and self.relative_phase == item.relative_phase
            and self.shape == item.shape
            and self.channel == item.channel
            and self.type == item.type
            and self.qubit == item.qubit
        )
