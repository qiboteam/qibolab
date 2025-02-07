from typing import Annotated, Optional

import numpy as np
from pydantic import PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseLike
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
        sequence: list[PulseLike],
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
            waveforms={k: w.waveform for k, w in waveforms_.items()},
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


def compile(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    sampling_rate: float,
) -> dict[ChannelId, Q1Sequence]:
    return {
        ch: Q1Sequence.from_pulses(seq, sweepers, options, sampling_rate, ch)
        for ch, seq in sequence.by_channel.items()
    }
