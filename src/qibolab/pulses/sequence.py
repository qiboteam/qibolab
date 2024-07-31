"""PulseSequence class."""

from collections import defaultdict
from typing import Optional

from .pulse import Delay, PulseLike, PulseType


class PulseSequence(defaultdict[str, list[PulseLike]]):
    """Synchronized sequence of control instructions across multiple channels.

    The keys are names of channels, and the values are lists of pulses
    and delays
    """

    def __init__(self, seq_dict: Optional[dict[str, list[PulseLike]]] = None):
        initial_content = seq_dict if seq_dict is not None else {}
        super().__init__(list, **initial_content)

    @property
    def probe_pulses(self):
        """Return list of the readout pulses in this sequence."""
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

    def extend(self, other: "PulseSequence") -> None:
        """Appends other in-place such that the result is self + necessary
        delays to synchronize channels + other."""
        tol = 1e-12
        durations = {ch: self.channel_duration(ch) for ch in other}
        max_duration = max(durations.values(), default=0.0)
        for ch, duration in durations.items():
            if (delay := round(max_duration - duration, int(1 / tol))) > 0:
                self[ch].append(Delay(duration=delay))
            self[ch].extend(other[ch])

    def copy(self) -> "PulseSequence":
        """Return shallow copy of self."""
        ps = PulseSequence()
        ps.extend(self)
        return ps
