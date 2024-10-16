import inspect
import re
import textwrap
from typing import Annotated, Any, Optional, Union

from pydantic import BeforeValidator, model_serializer, model_validator

from ...serialize import Model


class Register(Model):
    number: int

    @model_validator(mode="before")
    @classmethod
    def load(cls, data: Any) -> Any:
        assert data[0] == "R"
        num = int(data[1:])
        assert 0 <= num < 64
        return {"number": num}

    @model_serializer
    def dump(self) -> str:
        return f"R{self.number}"


class Reference(Model):
    label: str

    @model_validator(mode="before")
    @classmethod
    def load(cls, data: Any) -> Any:
        assert data[0] == "@"
        return {"label": data[1:]}

    @model_serializer
    def dump(self) -> str:
        return f"@{self.label}"


MultiBaseInt = Annotated[int, BeforeValidator(lambda n: int(n, 0))]
Immediate = Union[MultiBaseInt, Reference]
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

    @property
    def asm(self) -> str:
        key = self.keyword()
        args = self.args()
        instr = " ".join([key] + [str(a) for a in args])
        return instr


class Illegal(Instr):
    """Illegal instruction.

    Instruction that should not be executed. If it is executed, the
    sequencer will stop with the illegal instruction flag set.
    """


class Stop(Instr):
    """Stop the sequencer.

    Instruction that stops the sequencer.
    """


class Nop(Instr):
    """No operation.

    No operation instruction, that does nothing. It is used to pass a
    single cycle in the classic part of the sequencer without any
    operations.
    """


Control = Union[Illegal, Stop, Nop]


class Jmp(Instr):
    """Jump.

    Jump to the instruction indicated by :attr:`address`.
    """

    address: Value


class Jge(Instr):
    """Jump if greater.

    If :attr:`a` is greater than or equal to :attr:`b`, jump to the instruction
    indicated by :attr:`address`.
    """

    a: Register
    b: Immediate
    address: Value


class Jlt(Instr):
    """Jump if smaller.

    If :attr:`a` is less than :attr:`b`, jump to the instruction indicated by
    :attr:`address`.
    """

    a: Register
    b: Immediate
    address: Value


class Loop(Instr):
    """Loop.

    Decrement :attr:`a` by one and, if the result is non-zero, jump to the instruction
    indicated by :attr:`address`.
    """

    a: Register
    address: Value


Jump = Union[Jmp, Jge, Jlt, Loop]


class Move(Instr):
    """Move value.

    :attr:`source` is moved / copied to :attr:`destination`.

    ::

        destination = source
    """

    source: Value
    destination: Register


class Not(Instr):
    """Bit-wise inversion.

    Bit-wise invert :attr:`source` and move the result to :attr:`destination`.

    ::

        destination = ~source
    """

    source: Value
    destination: Register


class Add(Instr):
    """Addition.

    Add :attr:`b` to :attr:`a` and move the result to :attr:`destination`.

    ::

        destination = a + b
    """

    a: Register
    b: Value
    destination: Register


class Sub(Instr):
    """Subtraction.

    Subtract :attr:`b` from :attr:`a` and move the result to :attr:`destination`.

    ::

        destination = a - b
    """

    a: Register
    b: Value
    destination: Register


class And(Instr):
    """Bit-wise conjuction.

    Bit-wise AND :attr:`a` and :attr:`b` and move the result to :attr:`destination`.

    ::

        destination = a & b
    """

    a: Register
    b: Value
    destination: Register


class Or(Instr):
    """Bit-wise disjuction.

    Bit-wise OR :attr:`a` and :attr:`b` and move the result to :attr:`destination`.

    ::

        destination = a | b
    """

    a: Register
    b: Value
    destination: Register


class Xor(Instr):
    """Bit-wise exclusive disjuction.

    Bit-wise XOR :attr:`a` and :attr:`b` and move the result to :attr:`destination`.

    ::

        destination = a ^ b
    """

    a: Register
    b: Value
    destination: Register


class Asl(Instr):
    """Bit-wise left-shift.

    Bit-wise left-shift :attr:`a` by :attr:`b` number of bits and move the result to
    :attr:`destination`.

    ::

        destination = a << b
    """

    a: Register
    b: Value
    destination: Register


class Asr(Instr):
    """Bit-wise right-shift.

    Bit-wise right-shift :attr:`a` by :attr:`b` number of bits and move the result to
    :attr:`destination`.

    ::

        destination = a << b
    """

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


def _format_comment(text: str, width: Optional[int] = None) -> str:
    lines = [""]
    for block in text.splitlines():
        lines.extend(
            textwrap.wrap(
                block, width=width - 2, break_long_words=False, break_on_hyphens=False
            )
            if width is not None
            else [block]
        )
    return "\n# ".join(lines)[1:]


class Comment(str):
    def asm(self, width: Optional[int] = None) -> str:
        return _format_comment(self, width) + "\n"


class Line(Model):
    instruction: Instruction
    label: Optional[str]
    comment: Optional[str]

    def __rich_repr__(self):
        yield self.instruction
        yield "label", self.label, None
        yield "comment", self.comment, None

    def asm(
        self,
        width: Optional[int] = None,
        label_width: Optional[int] = None,
        instr_width: Optional[int] = None,
    ) -> str:
        if label_width is None:
            label_width = len(self.label) if self.label is not None else -1
        label = self.label if self.label is not None else ""
        if instr_width is None:
            instr_width = len(self.instruction.asm)
        code = f"{label:{label_width+1}}{self.instruction.asm:{instr_width+1}}"
        if self.comment is None:
            return code
        comment = _format_comment(
            self.comment, width - len(code) if width is not None else None
        ).splitlines()
        return "\n".join(
            [code + comment[0]] + [" " * len(code) + c for c in comment[1:]]
        )


Element = Union[Line, Comment]


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

    def asm(self, width: Optional[int] = None, comments: bool = True) -> str:
        max_label_len = max(
            len(line.label)
            for line in self.elements
            if isinstance(line, Line) and line.label is not None
        )
        max_instr_len = max(
            len(line.instruction.asm)
            for line in self.elements
            if isinstance(line, Line)
        )
        code = "\n".join(
            (
                (el if comments else el.model_copy(update={"comment": None})).asm(
                    width, label_width=max_label_len, instr_width=max_instr_len
                )
                if isinstance(el, Line)
                else (el.asm(width) if comments else "")
            )
            for el in self.elements
        )
        return code if comments else re.sub("^\n*", "", re.sub("\n+", "\n", code))
