from typing import Annotated

import numpy as np
from pydantic import PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseLike, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers

from ..ast_ import Program
from ..parse import parse
from .acquisition import Acquisitions, acquisitions
from .program import program
from .waveforms import Waveform, Waveforms, waveform_indices, waveforms

__all__ = ["Sequence"]


Weight = Waveform


class Sequence(Model):
    waveforms: Waveforms
    weights: dict[str, Weight]
    acquisitions: Acquisitions
    program: Annotated[
        Program,
        PlainSerializer(lambda p: p.asm()),
        PlainValidator(lambda p: p if isinstance(p, Program) else parse(p)),
    ]

    @classmethod
    def from_pulses(
        cls,
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        options: ExecutionParameters,
        sampling_rate: float,
    ):
        waveforms_ = waveforms(sequence, sampling_rate)
        acquisitions_ = acquisitions(
            sequence, np.prod(options.bins(sweepers), dtype=int)
        )
        return cls(
            waveforms=waveforms_,
            weights={},
            acquisitions=acquisitions_,
            program=program(
                sequence,
                waveform_indices(waveforms_),
                acquisitions_,
                options,
                sweepers,
                sampling_rate,
            ),
        )

    @classmethod
    def empty(cls):
        return cls(
            waveforms={}, weights={}, acquisitions={}, program=Program(elements=[])
        )


def _split_channels(sequence: PulseSequence) -> dict[ChannelId, PulseSequence]:
    def unwrap(pulse: PulseLike, output: bool) -> PulseLike:
        return (
            pulse
            if not isinstance(pulse, Readout)
            else pulse.probe
            if output
            else pulse.acquisition
        )

    def unwrap_seq(seq: PulseSequence, output: bool) -> PulseSequence:
        return PulseSequence((ch, unwrap(p, output)) for ch, p in seq)

    def ch_pulses(channel: ChannelId) -> PulseSequence:
        return PulseSequence((ch, pulse) for ch, pulse in sequence if ch == channel)

    def probe(channel: ChannelId) -> ChannelId:
        return channel.split("/")[0] + "/probe"

    def split(channel: ChannelId) -> dict[ChannelId, PulseSequence]:
        seq = ch_pulses(channel)
        readouts = any(isinstance(p, Readout) for _, p in seq)
        assert not readouts or probe(channel) not in sequence.channels
        return (
            {channel: seq}
            if not readouts
            else {
                channel: unwrap_seq(seq, output=False),
                probe(channel): unwrap_seq(seq, output=True),
            }
        )

    return {
        ch: seq for channel in sequence.channels for ch, seq in split(channel).items()
    }


def compile(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    sampling_rate: float,
) -> dict[ChannelId, Sequence]:
    return {
        ch: Sequence.from_pulses(seq, sweepers, options, sampling_rate)
        for ch, seq in _split_channels(sequence).items()
    }
