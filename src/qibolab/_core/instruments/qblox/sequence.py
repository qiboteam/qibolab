from typing import Annotated

from pydantic import AfterValidator, PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.pulses.pulse import Pulse, PulseLike, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import ParallelSweepers

from .ast_ import Program
from .parse import parse
from .program import program

__all__ = []


class Waveform(Model):
    data: Annotated[ArrayList, AfterValidator(lambda a: a.astype(float))]
    index: int


Weight = Waveform
Waveforms = dict[str, Waveform]


class Acquisition(Model):
    num_bins: int
    index: int


def waveforms(sequence: PulseSequence, sampling_rate: float) -> Waveforms:
    def id_(pulse):
        return str(hash(pulse))

    def waveform(pulse):
        return Waveform(data=pulse.envelopes(sampling_rate), index=0)

    def pulse_(event: PulseLike):
        return event.probe if isinstance(event, Readout) else event

    return {
        id_(pulse_(event)): waveform(pulse_(event))
        for _, event in sequence
        if isinstance(event, (Pulse, Readout))
    }


class Sequence(Model):
    waveforms: Waveforms
    weights: dict[str, Weight]
    acquisitions: dict[str, Acquisition]
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
        return cls(
            waveforms=waveforms(sequence, sampling_rate),
            weights={},
            acquisitions={},
            program=program(),
        )
