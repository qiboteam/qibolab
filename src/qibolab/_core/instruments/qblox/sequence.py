from typing import Annotated

from pydantic import PlainSerializer, PlainValidator

from qibolab._core.serialize import Model

from .ast_ import Program
from .parse import parse


class Waveforms(Model):
    pass


class Weights(Model):
    pass


class Acquisitions(Model):
    pass


class Sequence(Model):
    waveforms: Waveforms
    weights: Weights
    acquisitions: Acquisitions
    program: Annotated[
        Program, PlainSerializer(lambda p: p.asm()), PlainValidator(parse)
    ]
