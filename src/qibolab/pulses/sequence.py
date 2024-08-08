"""PulseSequence class."""

from collections import UserList
from collections.abc import Iterable

from qibolab.components import ChannelId

from .pulse import Delay, Pulse, PulseLike

__all__ = ["PulseSequence"]

_Element = tuple[ChannelId, PulseLike]


class PulseSequence(UserList[_Element]):
    """Synchronized sequence of control instructions across multiple channels.

    The sequence is a linear stream of instructions, which may be
    executed in parallel over multiple channels.

    Each instruction is composed by the pulse-like object representing
    the action, and the channel on which it should be performed.
    """

    @property
    def duration(self) -> float:
        """Duration of the entire sequence."""
        return max((self.channel_duration(ch) for ch in self.channels), default=0.0)

    @property
    def channels(self) -> set[ChannelId]:
        """Channels involved in the sequence."""
        return {ch for (ch, _) in self}

    def channel(self, channel: ChannelId) -> Iterable[PulseLike]:
        """Isolate pulses on a given channel."""
        return (pulse for (ch, pulse) in self if ch == channel)

    def channel_duration(self, channel: ChannelId) -> float:
        """Duration of the given channel."""
        return sum(pulse.duration for pulse in self.channel(channel))

    def concatenate(self, other: "PulseSequence") -> None:
        """Juxtapose two sequences.

        Appends ``other`` in-place such that the result is:
            - ``self``
            - necessary delays to synchronize channels
            - ``other``
        """
        durations = {ch: self.channel_duration(ch) for ch in other.channels}
        max_duration = max(durations.values(), default=0.0)
        for ch, duration in durations.items():
            delay = max_duration - duration
            if delay > 0:
                self.append((ch, Delay(duration=delay)))
            self.extend((ch, pulse) for pulse in other.channel(ch))

    @property
    def probe_pulses(self) -> list[Pulse]:
        """Return list of the readout pulses in this sequence."""
        # pulse filter needed to exclude delays
        return [
            pulse for (ch, pulse) in self if isinstance(pulse, Pulse) if "probe" in ch
        ]
