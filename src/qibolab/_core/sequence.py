"""PulseSequence class."""

from collections import UserList, defaultdict
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any, Union

from pydantic import TypeAdapter
from pydantic_core import core_schema

from qibolab._core.pulses.pulse import PulseId

from .identifier import ChannelId
from .pulses import Acquisition, Align, Delay, PulseLike, Readout

__all__ = ["PulseSequence"]

_Element = tuple[ChannelId, PulseLike]
InputOps = Union[Readout, Acquisition]

_adapted_sequence = TypeAdapter(list[_Element])


def _synchronize(sequence: "PulseSequence", channels: Iterable[ChannelId]) -> None:
    """Helper for ``concatenate`` and ``align_to_delays``.

    Modifies given ``sequence`` in-place!
    """
    durations = {ch: sequence.channel_duration(ch) for ch in channels}
    max_duration = max(durations.values(), default=0.0)
    for ch, duration in durations.items():
        delay = max_duration - duration
        if delay > 0:
            sequence.append((ch, Delay(duration=delay)))


class PulseSequence(UserList[_Element]):
    """Synchronized sequence of control instructions across multiple channels.

    The sequence is a linear stream of instructions, which may be
    executed in parallel over multiple channels.

    Each instruction is composed by the pulse-like object representing
    the action, and the channel on which it should be performed.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        schema = handler(list[_Element])
        return core_schema.no_info_after_validator_function(
            cls._validate,
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=False
            ),
        )

    @classmethod
    def _validate(cls, value):
        return cls(value)

    @staticmethod
    def _serialize(value):
        return _adapted_sequence.dump_python(list(value))

    @classmethod
    def load(cls, value: list[tuple[str, PulseLike]]):
        return TypeAdapter(cls).validate_python(value)

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
        sequence = (
            self.align_to_delays()
            if any(isinstance(pulse, Align) for _, pulse in self)
            else self
        )
        return sum(pulse.duration for pulse in sequence.channel(channel))

    def pulse_channels(self, pulse_id: PulseId) -> list[ChannelId]:
        """Find channels on which a pulse with a given id plays."""
        return [channel for channel, pulse in self if pulse.id == pulse_id]

    def concatenate(self, other: Iterable[_Element]) -> None:
        """Concatenate two sequences.

        Appends ``other`` in-place such that the result is:

            - ``self``
            - necessary delays to synchronize channels
            - ``other``

        Guarantees that the all the channels in the concatenated
        sequence will start simultaneously
        """
        _synchronize(self, PulseSequence(other).channels)
        self.extend(other)

    def __ilshift__(self, other: Iterable[_Element]) -> "PulseSequence":
        """Juxtapose two sequences.

        Alias to :meth:`concatenate`.
        """
        self.concatenate(other)
        return self

    def __lshift__(self, other: Iterable[_Element]) -> "PulseSequence":
        """Juxtapose two sequences.

        A copy is made, and no input is altered.

        Other than that, it is based on :meth:`concatenate`.
        """
        copy = self.copy()
        copy <<= other
        return copy

    def juxtapose(self, other: Iterable[_Element]) -> None:
        """Juxtapose two sequences.

        Appends ``other`` in-place such that the result is:

            - ``self``
            - necessary delays to synchronize channels
            - ``other``

        Guarantee simultaneous start and no overlap.
        """
        _synchronize(self, PulseSequence(other).channels | self.channels)
        self.extend(other)

    def __ior__(self, other: Iterable[_Element]) -> "PulseSequence":
        """Juxtapose two sequences.

        Alias to :meth:`concatenate`.
        """
        self.juxtapose(other)
        return self

    def __or__(self, other: Iterable[_Element]) -> "PulseSequence":
        """Juxtapose two sequences.

        A copy is made, and no input is altered.

        Other than that, it is based on :meth:`concatenate`.
        """
        copy = self.copy()
        copy |= other
        return copy

    def align(self, channels: list[ChannelId]) -> Align:
        """Introduce align commands to the sequence."""
        align = Align()
        for channel in channels:
            self.append((channel, align))
        return align

    def align_to_delays(self) -> "PulseSequence":
        """Compile align commands to delays."""

        # keep track of ``Align`` command that were already played
        # because the same ``Align`` will appear on multiple channels
        # in the sequence
        processed_aligns = set()

        new = type(self)()
        for channel, pulse in self:
            if isinstance(pulse, Align):
                if pulse.id not in processed_aligns:
                    channels = self.pulse_channels(pulse.id)
                    _synchronize(new, channels)
                    processed_aligns.add(pulse.id)
            else:
                new.append((channel, pulse))
        return new

    def trim(self) -> "PulseSequence":
        """Drop final delays.

        The operation is not in place, and does not modify the original
        sequence.
        """
        terminated = set()
        new = []
        for ch, pulse in reversed(self):
            if ch not in terminated:
                if isinstance(pulse, Delay):
                    continue
                terminated.add(ch)
            new.append((ch, pulse))
        return type(self)(reversed(new))

    @property
    def acquisitions(self) -> list[tuple[ChannelId, InputOps]]:
        """Return list of the readout pulses in this sequence.

        .. note::

            This selects only the :class:`Acquisition` events, and not all the
            instructions directed to an acquistion channel
        """
        # pulse filter needed to exclude delays
        return [(ch, p) for ch, p in self if isinstance(p, (Acquisition, Readout))]

    @property
    def split_readouts(self) -> "PulseSequence":
        """Split readout operations in its constituents.

        This will also double the rest of the channels (mainly delays) on which the
        readouts are placed, assuming the probe channels to be absent.

        .. note::

            Since :class:`Readout` is only placed on an acquisition channel, the name of
            the associated probe channel is actually unknown.
            This function assumes the convention that the relevant channels are named
            ``.../acquisition`` and ``.../probe``.
        """

        def unwrap(pulse: PulseLike, double: bool) -> tuple[PulseLike, ...]:
            return (
                (pulse.acquisition, pulse.probe)
                if isinstance(pulse, Readout)
                else (pulse, pulse)
                if double
                else (pulse,)
            )

        @cache
        def probe(channel: ChannelId) -> ChannelId:
            return channel.split("/")[0] + "/probe"

        readouts = {ch for ch, p in self if isinstance(p, Readout)}
        return type(self)(
            [
                (ch_, p_)
                for ch, p in self
                for ch_, p_ in zip((ch, probe(ch)), unwrap(p, ch in readouts))
            ]
        )

    @property
    def by_channel(self) -> dict[ChannelId, list[PulseLike]]:
        """Separate sequence into channels dictionary."""
        seqs = defaultdict(list)
        for ch, pulse in self:
            seqs[ch].append(pulse)

        return seqs
