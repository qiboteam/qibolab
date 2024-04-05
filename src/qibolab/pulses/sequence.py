"""PulseSequence class."""

from collections import defaultdict


class PulseSequence(defaultdict):
    """Synchronized sequence of pulses across multiple channels.

    The keys are names of channels, and the values are lists of pulses
    and delays
    """

    def __init__(self):
        super().__init__(list)

    @property
    def duration(self) -> int:
        """The time when the last pulse of the sequence finishes."""

        def channel_duration(ch: str):
            return sum(item.duration for item in self[ch])

        return max((channel_duration(ch) for ch in self), default=0)

    def __add__(self, other: "PulseSequence") -> "PulseSequence":
        """Create a PulseSequence which is self + necessary delays to
        synchronize channels + other."""
        # TODO: implement
        ...
