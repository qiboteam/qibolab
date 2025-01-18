import inspect
import re
from typing import Optional, Union

from pydantic import model_validator

from qibolab.serialize_ import Model

Register = str
Immediate = int
Value = Union[Register, Immediate]

CAMEL_TO_SNAKE = re.compile("(?<=[a-z0-9])(?=[A-Z])(?!^)(?=[A-Z][a-z])")


class Instr(Model):
    @classmethod
    def keyword(cls):
        return CAMEL_TO_SNAKE.sub("_", cls.__name__).lower()

    @classmethod
    def from_args(cls, *args):
        return cls(**dict(zip(cls.model_fields.keys(), args)))

    def args(self):
        return list(self.model_dump().values())


class Illegal(Instr):
    """"""


class Stop(Instr):
    """"""


class Nop(Instr):
    """"""


Control = Union[Illegal, Stop, Nop]


class Jmp(Instr):
    address: Value


class Jge(Instr):
    a: Register
    b: Immediate
    address: Value


class Jlt(Instr):
    a: Register
    b: Immediate
    address: Value


class Loop(Instr):
    a: Register
    address: Value


Jump = Union[Jmp, Jge, Jlt, Loop]


class Move(Instr):
    source: Value
    destination: Register


class Not(Instr):
    source: Value
    destination: Register


class Add(Instr):
    a: Register
    b: Value
    destination: Register


class Sub(Instr):
    a: Register
    b: Value
    destination: Register


class And(Instr):
    a: Register
    b: Value
    destination: Register


class Or(Instr):
    a: Register
    b: Value
    destination: Register


class Xor(Instr):
    a: Register
    b: Value
    destination: Register


class Asl(Instr):
    a: Register
    b: Value
    destination: Register


class Asr(Instr):
    a: Register
    b: Value
    destination: Register


Arithmetic = Union[Move, Not, Add, Sub, And, Or, Xor, Asl, Asr]


class SetMrk(Instr):
    mask: Value


class SetFreq(Instr):
    value: Value


class ResetPh(Instr):
    """"""


class SetPh(Instr):
    value: Value


class SetPhDelta(Instr):
    value: Value


class SetAwgGain(Instr):
    value_0: Value
    value_1: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.value_0) == type(self.value_1)
        return self


class SetAwgOffs(Instr):
    value_0: Value
    value_1: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.value_0) == type(self.value_1)
        return self


ParamOps = Union[SetMrk, SetFreq, ResetPh, SetPh, SetPhDelta, SetAwgGain, SetAwgOffs]

Q1Instr = Union[Control, Jump, Arithmetic, ParamOps]


class SetCond(Instr):
    enable: Value
    mask: Value
    operator: Value
    else_duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.enable) == type(self.mask)
        assert type(self.enable) == type(self.operator)
        return self


Conditional = SetCond


class UpdParam(Instr):
    duration: Immediate


class Play(Instr):
    wave_0: Value
    wave_1: Value
    duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.wave_0) == type(self.wave_1)
        return self


class Acquire(Instr):
    acquisition: Immediate
    bin: Value
    duration: Immediate


class AcquireWeighed(Instr):
    acquisition: Immediate
    bin: Value
    weight_0: Value
    weight_1: Value
    duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.bin) == type(self.weight_0)
        assert type(self.bin) == type(self.weight_1)
        return self


class AcquireTtl(Instr):
    acquisition: Immediate
    bin: Value
    enable: Immediate
    duration: Immediate


Io = Union[UpdParam, Play, Acquire, AcquireWeighed, AcquireTtl]


class SetLatchEn(Instr):
    enable: Value
    duration: Immediate


class LatchRst(Instr):
    duration: Value


Trigger = Union[SetLatchEn, LatchRst]


class Wait(Instr):
    duration: Value


class WaitTrigger(Instr):
    trigger: Value
    duration: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.trigger) == type(self.duration)
        return self


class WaitSync(Instr):
    duration: Value


WaitOps = Union[Wait, WaitTrigger, WaitSync]

RealTimeInstr = Union[Conditional, Io, Trigger, WaitOps]


Instruction = Union[Q1Instr, RealTimeInstr]


INSTRUCTIONS = {
    c.keyword(): c
    for c in locals().values()
    if inspect.isclass(c) and issubclass(c, Instr)
}


class Comment(str):
    pass


class Line(Model):
    instruction: Instruction
    label: Optional[str]
    comment: Optional[str]

    def __rich_repr__(self):
        print(self.asm)
        yield self.instruction
        yield "label", self.label, None
        yield "comment", self.comment, None

    @property
    def asm(self):
        key = self.instruction.keyword()
        args = self.instruction.args()
        instr = " ".join([key] + [str(a) for a in args])
        return self.label, instr, self.comment


Element = Union[Line, Comment, "Block"]


class Block(Model):
    elements: list[Element]


class Program(Model):
    elements: list[Element]

    @classmethod
    def from_elements(cls, elements: list[Element]):
        comments = []
        elements_ = []

        # group comments
        for el in elements:
            if isinstance(el, Comment):
                comments.append(el)
                continue
            if len(comments) > 0:
                comment = "\n".join(comments)
                elements_.append(Comment(comment))
                comments = []
            elements_.append(el)

        return cls(elements=elements_)
