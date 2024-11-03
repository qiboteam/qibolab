from typing import Annotated

from pydantic import AfterValidator, PlainSerializer, PlainValidator

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import ArrayList, Model
from qibolab._core.sweeper import ParallelSweepers

from .ast_ import Program
from .parse import parse

__all__ = []


class Waveform(Model):
    data: Annotated[ArrayList, AfterValidator(lambda a: a.astype(float))]
    index: int


Weight = Waveform


class Acquisition(Model):
    num_bins: int
    index: int


class Sequence(Model):
    waveforms: dict[str, Waveform]
    weights: dict[str, Weight]
    acquisitions: dict[str, Acquisition]
    program: Annotated[
        Program, PlainSerializer(lambda p: p.asm()), PlainValidator(parse)
    ]

    @classmethod
    def from_pulses(
        cls,
        sequence: PulseSequence,
        sweepers: list[ParallelSweepers],
        options: ExecutionParameters,
    ):
        return cls(
            waveforms={}, weights={}, acquisitions={}, program=Program(elements=[])
        )
