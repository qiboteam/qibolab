"""Pulse class."""

from dataclasses import fields
from enum import Enum
from typing import Union

import numpy as np

from qibolab.serialize_ import Model

from .envelope import Envelope, IqWaveform, Waveform


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
    DELAY = "dl"
    VIRTUALZ = "vz"


class Pulse(Model):
    """A pulse to be sent to the QPU."""

    duration: float
    """Pulse duration."""

    amplitude: float
    """Pulse digital amplitude (unitless).

    Pulse amplitudes are normalised between -1 and 1.
    """
    envelope: Envelope
    """The pulse envelope shape.

    See
    :cls:`qibolab.pulses.envelope.Envelopes` for list of available shapes.
    """
    relative_phase: float = 0.0
    """Relative phase of the pulse, in radians."""
    type: PulseType = PulseType.DRIVE
    """Pulse type, as an element of PulseType enumeration."""

    @classmethod
    def flux(cls, **kwargs):
        """Construct a flux pulse.

        It provides a simplified syntax for the :cls:`Pulse` constructor, by applying
        suitable defaults.
        """
        kwargs["relative_phase"] = 0
        if "type" not in kwargs:
            kwargs["type"] = PulseType.FLUX
        return cls(**kwargs)

    @property
    def id(self) -> int:
        return id(self)

    def i(self, sampling_rate: float) -> Waveform:
        """The envelope waveform of the i component of the pulse."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.i(samples)

    def q(self, sampling_rate: float) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        samples = int(self.duration * sampling_rate)
        return self.amplitude * self.envelope.q(samples)

    def envelopes(self, sampling_rate: float) -> IqWaveform:
        """A tuple with the i and q envelope waveforms of the pulse."""
        return np.array([self.i(sampling_rate), self.q(sampling_rate)])

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


class Delay(Model):
    """A wait instruction during which we are not sending any pulses to the
    QPU."""

    duration: int
    """Delay duration in ns."""
    type: PulseType = PulseType.DELAY
    """Type fixed to ``DELAY`` to comply with ``Pulse`` interface."""


class VirtualZ(Model):
    """Implementation of Z-rotations using virtual phase."""

    phase: float
    """Phase that implements the rotation."""
    type: PulseType = PulseType.VIRTUALZ

    @property
    def duration(self):
        """Duration of the virtual gate should always be zero."""
        return 0


PulseLike = Union[Pulse, Delay, VirtualZ]
