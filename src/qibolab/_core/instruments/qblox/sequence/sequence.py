from typing import Annotated, Optional

import numpy as np
from pydantic import PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseLike, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers

from ..q1asm import Program, parse
from .acquisition import Acquisitions, MeasureId, Weight, Weights, acquisitions
from .program import program
from .waveforms import Waveforms, waveform_indices, waveforms

__all__ = ["Q1Sequence"]


def _weight_len(w: Optional[Weight]) -> Optional[int]:
    return len(w.data) if w is not None else None


class Q1Sequence(Model):
    waveforms: Waveforms
    weights: Weights
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
        channel: ChannelId,
    ):
        waveforms_ = waveforms(sequence, sampling_rate)
        acquisitions_ = acquisitions(
            sequence, np.prod(options.bins(sweepers), dtype=int)
        )
        return cls(
            waveforms=waveforms_,
            weights={},
            acquisitions={k: a.acquisition for k, a in acquisitions_.items()},
            program=program(
                sequence,
                waveform_indices(waveforms_),
                acquisitions_,
                options,
                sweepers,
                sampling_rate,
                channel,
            ),
        )

    @classmethod
    def empty(cls):
        return cls(
            waveforms={}, weights={}, acquisitions={}, program=Program(elements=[])
        )

    @property
    def integration_lengths(self) -> dict[MeasureId, Optional[int]]:
        """Determine the integration lengths fixed by weights.

        Returns ``None`` for those acquisitions which are non-weighted, since the length
        is determined by a runtime configuration.

        For those, cf.
        https://docs.qblox.com/en/main/api_reference/sequencer.html#Sequencer.integration_length_acq
        """

        return {acq: _weight_len(self.weights.get(acq)) for acq in self.acquisitions}


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
) -> dict[ChannelId, Q1Sequence]:
    return {
        ch: Q1Sequence.from_pulses(seq, sweepers, options, sampling_rate, ch)
        for ch, seq in _split_channels(sequence).items()
    }
