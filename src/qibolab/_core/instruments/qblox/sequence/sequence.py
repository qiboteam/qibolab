from typing import Annotated

import numpy as np
from pydantic import PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
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
