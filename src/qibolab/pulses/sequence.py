"""PulseSequence class."""

from collections import defaultdict

from pulse import PulseType


class PulseSequence(defaultdict):
    """Synchronized sequence of control instructions across multiple channels.

    The keys are names of channels, and the values are lists of pulses
    and delays
    """

    def __init__(self):
        super().__init__(list)

    @property
    def ro_pulses(self):
        """A new sequence containing only its readout pulses."""
        pulses = []
        for seq in self.values():
            for pulse in seq:
                if pulse.type == PulseType.READOUT:
                    pulses.append(pulse)
        return pulses

    @property
    def duration(self) -> int:
        """Duration of the entire sequence."""
        return max((self.channel_duration(ch) for ch in self), default=0)

    def channel_duration(self, channel: str) -> float:
        """Duration of the given channel."""
        return sum(item.duration for item in self[channel])

    def __add__(self, other: "PulseSequence") -> "PulseSequence":
        """Create a PulseSequence which is self + necessary delays to
        synchronize channels + other."""
        # TODO: implement
        ...
