from typing import Annotated

from pydantic import AfterValidator, PlainSerializer, PlainValidator

from qibolab._core.serialize import ArrayList, Model

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
