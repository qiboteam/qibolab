"""Pulse class."""
from dataclasses import dataclass
from typing import ClassVar

from qibolab.pulse.pulse import Pulse
from qibolab.pulse.pulse_shape.pulse_shape import PulseShape
from qibolab.pulse.pulse_shape.rectangular import Rectangular
from qibolab.typings import PulseName


@dataclass
class ReadoutPulse(Pulse):
    """Describes a single pulse to be added to waveform array."""

    name: ClassVar[PulseName] = PulseName.READOUT_PULSE
    pulse_shape: PulseShape = Rectangular()

    def __repr__(self):  # pylint: disable=useless-super-delegation
        """Redirect __repr__ magic method."""
        return super().__repr__()

    @property
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.3f')}, {self.frequency}, {format(self.phase, '.3f')}, '{self.shape}', {self.channel}, '{self.type}')"
