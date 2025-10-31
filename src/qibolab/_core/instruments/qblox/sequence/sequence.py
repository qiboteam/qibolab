from collections.abc import Iterable
from typing import Annotated, Optional

import numpy as np
from pydantic import PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Align, Pulse, PulseLike, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import (
    ParallelSweepers,
    Parameter,
    swept_channels,
    swept_pulses,
)

from ..q1asm import Program, parse
from .acquisition import Acquisitions, MeasureId, Weight, Weights, acquisitions
from .program import program
from .waveforms import Waveforms, waveform_indices, waveforms

__all__ = ["Q1Sequence"]


def _weight_len(w: Optional[Weight]) -> Optional[int]:
    return len(w.data) if w is not None else None


def _apply_sampling_rate(
    sequence: Iterable[PulseLike],
    sweepers: list[ParallelSweepers],
    sampling_rate: float,
) -> tuple[list[PulseLike], list[ParallelSweepers]]:
    sequence_ = [
        p.model_copy(
            update={"duration": p.duration * sampling_rate}
            if not isinstance(p, Align)
            # TODO: replace with `hasattr(p, "duration")`
            # but it's not passing the type checker
            # https://github.com/microsoft/pylance-release/issues/2237
            else {}
        )
        for p in sequence
    ]
    sweepers_ = [
        [
            (s * sampling_rate) if s.parameter is Parameter.duration else s
            for s in parsweep
        ]
        for parsweep in sweepers
    ]
    return (sequence_, sweepers_)


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
        channel: set[ChannelId],
        time_of_flight: Optional[float],
        duration: float,
    ) -> "Q1Sequence":
        padding = (
            duration
            - sum(p.duration if not isinstance(p, Align) else 0 for p in sequence)
        ) * sampling_rate
        waveforms_ = waveforms(
            sequence,
            sampling_rate,
            amplitude_swept={
                p.id for p in swept_pulses(sweepers, {Parameter.amplitude})
            },
            duration_swept={
                k: v
                for k, v in swept_pulses(sweepers, {Parameter.duration}).items()
                if isinstance(k, (Pulse, Readout)) and k in sequence
            },
        )
        sequence, sweepers = _apply_sampling_rate(sequence, sweepers, sampling_rate)
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
                channel,
                time_of_flight,
                int(padding),
            ),
        )

    @classmethod
    def empty(cls) -> "Q1Sequence":
        return cls(
            waveforms={}, weights={}, acquisitions={}, program=Program(elements=[])
        )

    @property
    def is_empty(self) -> bool:
        return len(self.program.elements) == 0

    @property
    def integration_lengths(self) -> dict[MeasureId, Optional[int]]:
        """Determine the integration lengths fixed by weights.

        Returns ``None`` for those acquisitions which are non-weighted, since the length
        is determined by a runtime configuration.

        For those, cf.
        https://docs.qblox.com/en/main/api_reference/sequencer.html#Sequencer.integration_length_acq
        """

        return {acq: _weight_len(self.weights.get(acq)) for acq in self.acquisitions}


def _effective_channels(ch: ChannelId, seq: Iterable[PulseLike]) -> set[ChannelId]:
    """Identify effective channels related to a subsequence.

    The channel is the declared one, unless in presence of :class:`Readout` operations.
    In which case the assumed *acquisition* channel is supplemented with a *probe* one.
    """
    return (
        {ch}
        if not any(isinstance(e, Readout) for e in seq)
        else {ch, f"{'/'.join(ch.split('/')[:-1])}/probe"}
    )


def compile(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    sampling_rate: float,
    time_of_flights: dict[ChannelId, float],
) -> dict[ChannelId, Q1Sequence]:
    duration = sequence.duration
    sweeper_channels = {ch: [] for ch in swept_channels(sweepers)}
    return {
        ch: Q1Sequence.from_pulses(
            seq,
            sweepers[::-1],
            options,
            sampling_rate,
            _effective_channels(ch, seq),
            time_of_flights.get(ch),
            duration,
        )
        for ch, seq in (sweeper_channels | sequence.by_channel).items()
    }
