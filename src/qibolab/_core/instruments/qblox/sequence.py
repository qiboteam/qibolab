from typing import Annotated, Union

from pydantic import AfterValidator, PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.pulses.pulse import Pulse, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import ParallelSweepers

from .ast_ import Program
from .parse import parse
from .program import ComponentId, WaveformIndices, program, pulse_uid

__all__ = []


class Waveform(Model):
    data: Annotated[ArrayList, AfterValidator(lambda a: a.astype(float))]
    index: int


Weight = Waveform
Waveforms = dict[ComponentId, Waveform]


class Acquisition(Model):
    num_bins: int
    index: int


def waveforms(sequence: PulseSequence, sampling_rate: float) -> Waveforms:
    def waveform(pulse: Pulse, component: str) -> Waveform:
        return Waveform(data=getattr(pulse, component)(sampling_rate), index=0)

    def pulse_(event: Union[Pulse, Readout]) -> Pulse:
        return event.probe if isinstance(event, Readout) else event

    indexless = {
        k: v
        for d in (
            {
                (pulse_uid(pulse_(event)), 0): waveform(pulse_(event), "i"),
                (pulse_uid(pulse_(event)), 1): waveform(pulse_(event), "q"),
            }
            for _, event in sequence
            if isinstance(event, (Pulse, Readout))
        )
        for k, v in d.items()
    }

    return {
        k: Waveform(data=v.data, index=i) for i, (k, v) in enumerate(indexless.items())
    }


def waveform_indices(waveforms: Waveforms) -> WaveformIndices:
    return {k: w.index for k, w in waveforms.items()}


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
        waveforms_ = waveforms(sequence, sampling_rate)
        return cls(
            waveforms=waveforms_,
            weights={},
            acquisitions={},
            program=program(
                sequence, waveform_indices(waveforms_), options, sweepers, sampling_rate
            ),
        )
