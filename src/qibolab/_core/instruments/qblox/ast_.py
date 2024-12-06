import inspect
import re
import textwrap
from typing import Annotated, Any, Optional, Union

from pydantic import (
    AfterValidator,
    BeforeValidator,
    field_validator,
    model_serializer,
    model_validator,
)

from ...serialize import Model

__all__ = []


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


MultiBaseInt = Annotated[
    int, BeforeValidator(lambda n: int(n, 0) if isinstance(n, str) else n)
]
Immediate = Union[MultiBaseInt, Reference]
Value = Union[Register, Immediate]

CAMEL_TO_SNAKE = re.compile("(?<=[a-z0-9])(?=[A-Z])(?!^)(?=[A-Z][a-z])")


class Instr(Model):
    @classmethod
    def keyword(cls) -> str:
        return CAMEL_TO_SNAKE.sub("_", cls.__name__).lower()

    @classmethod
    def from_args(cls, *args):
        return cls(**dict(zip(cls.model_fields.keys(), args)))

    @property
    def args(self) -> list:
        return list(self.model_dump().values())

    def asm(self, key_width: Optional[int] = None) -> str:
        key = self.keyword()
        if key_width is None:
            key_width = len(key)
        instr = f"{key:<{key_width+1}}"
        return (instr + ",".join([str(a) for a in self.args])).strip()


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
"""Control instructions."""


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
"""Jump instructions."""


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
"""Arithmetic instructions."""


class SetMrk(Instr):
    """Set marker.

    Set marker output channels to ``val`` (bits 0-3), where the bit index
    corresponds to the channel index for baseband modules.

    For QCM-RF modules, bit indices 0 & 1 correspond to enabling output
    1 and 2 switches respectively; indices 2 & 3 correspond to marker
    output 2 and 1 (sic!) respectively.

    For QRM-RF modules, bit index 0 is inactive, bit index 1 corresponds
    to enabling output 1 switch; indices 2 & 3 correspond to marker
    output 1 and 2 (sic!) respectively.

    The values are OR'ed by that of other sequencers.
    """

    mask: Value


class SetFreq(Instr):
    """Set frequency.

    Set the frequency of the NCO used by the AWG and acquisition using
    :attr:`value`.
    The frequency is divided into 4e9 steps between -500 and 500 MHz and
    expressed as an integer between -2e9 and 2e9 (e.g.1 MHz=4e6).
    """

    value: Value


class ResetPh(Instr):
    """Phase reset.

    Reset the absolute phase of the NCO used by the AWG and acquisition
    to 0°. This also resets any relative phase offsets that were already
    statically or dynamically set.
    """


class SetPh(Instr):
    """Set absolute phase.

    Set the absolute phase of the NCO used by the AWG and acquisition
    using value. The phase is divided into 1e9 steps between 0° and
    360°, expressed as an integer between 0 and 1e9 (e.g 45°=125e6).
    """

    value: Value


class SetPhDelta(Instr):
    """Set phase offset.

    Set an offset on top of the relative phase of the NCO used by the
    AWG and acquisition. The offset is applied on top of the phase set
    using ``set_ph``. See :class:`SetPh` for more details regarding the
    argument.
    """

    value: Value


class SetAwgGain(Instr):
    """Set AWG gain.

    Set AWG gain for path 0 using :attr:`value_0` and path 1 using :attr:`value_1`.
    Both are integers in ``[-32 768, 32 767]``.
    """

    value_0: Value
    value_1: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.value_0) is type(self.value_1)
        return self

    @field_validator("value_0", "value_1")
    @classmethod
    def check_range(cls, v: Value) -> Value:
        if isinstance(v, int):
            assert -32_768 <= v <= 32_767
        return v


class SetAwgOffs(Instr):
    """Set AWG offset.

    Set AWG offset for path 0 using :attr:`value_0` and path 1 using :attr:`value_1`.
    Both are integers in ``[-32 768, 32 767]``.
    """

    value_0: Value
    value_1: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.value_0) is type(self.value_1)
        return self

    @field_validator("value_0", "value_1")
    @classmethod
    def check_range(cls, v: Value) -> Value:
        if isinstance(v, int):
            assert -32_768 <= v <= 32_767
        return v


ParamOps = Union[SetMrk, SetFreq, ResetPh, SetPh, SetPhDelta, SetAwgGain, SetAwgOffs]
"""Parameter operations.

The parameters are latched and only updated when the ``upd_param``, ``play``, ``acquire``,
``acquire_weighed`` or ``acquire_ttl`` instructions are executed.
"""

Q1Instr = Union[Control, Jump, Arithmetic, ParamOps]
"""Q1 Instructions.

These instructions are used to compose and manipulate the arguments of
real-time instructions. They are always executed before the next real-
time instruction, and therefore take zero wall-time.
"""


class SetCond(Instr):
    """Condition all subsequent instructions.

    Enable/disable conditionality on all following real-time
    instructions based on :attr:`enable`. The condition is based on the
    trigger network address counters being thresholded based on the associated
    counter threshold parameters set through QCoDeS.

    The results are masked using :attr:`mask` (bits 0-14), where the bit
    index plus one corresponds to the trigger address.

    This creates a selection to include in the final logical operation
    set using :attr:`operator`. Logical operators are OR, NOR, AND, NAND,
    XOR, XNOR, where a value for operator of 0 is OR and 5 is XNOR
    respectively.

    The logical operation result (true/false) determines the condition.
    If the condition is true upon evaluation, the next real-time
    instruction is executed. Else the real-time path ignores the
    instruction and waits for :attr:`else_duration` nanoseconds before
    continuing to the next.

    All following real-time instructions are subject to the same
    condition, until either the conditionality is disabled or updated.
    Disabling the conditionality does not affect the address counters
    and does not trigger else_duration.
    """

    enable: Value
    mask: Value
    operator: Value
    else_duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.enable) is type(self.mask)
        assert type(self.enable) is type(self.operator)
        return self


Conditional = SetCond
"""Conditional instructions."""


class UpdParam(Instr):
    """Update parameters.

    Update the latched parameters, and then wait for :attr:`duration` nanoseconds.
    """

    duration: Immediate


class Play(Instr):
    """Play waveforms.

    Update the latched parameters, interrupt currently playing waves and
    start playing AWG waveforms stored at indexes :attr:`wave_0` on path 0 and
    :attr:`wave_1` on path 1.
    """

    wave_0: Value
    wave_1: Value
    duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.wave_0) is type(self.wave_1)
        return self


class Acquire(Instr):
    """Start new acquisition.

    Update the latched parameters, interrupt currently active
    acquisitions and start the acquisition referred to using index
    :attr:`acquisition` and store the bin data in :attr:`bin` index bin.

    Integration is executed using a square weight with a preset length
    through the associated QCoDeS parameter.
    """

    acquisition: Immediate
    bin: Value
    duration: Immediate


class AcquireWeighed(Instr):
    """Start new acquisition with given weights.

    Update the latched parameters, interrupt currently active
    acquisitions and start the acquisition referred to using index
    :attr:`acquisition` and store the bin data in :attr:`bin` index bin.

    Integration is executed using weights stored at indices :attr:`weight_0` for
    path 0 and :attr:`weight_1` for path 1.
    """

    acquisition: Immediate
    bin: Value
    weight_0: Value
    weight_1: Value
    duration: Immediate

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.bin) is type(self.weight_0)
        assert type(self.bin) is type(self.weight_1)
        return self


class AcquireTtl(Instr):
    """Start TTL trigger acquisition.

    Update the latched parameters, start the TTL trigger acquisition
    referred to using index :attr:`acquisition` and store the bin data in bin
    index :attr:`bin`. Enable the acquisition by writing 1 to :attr:`enable`.

    The TTL trigger acquisition has to be actively disabled afterwards
    by writing 0 to :attr:`enable`.
    """

    acquisition: Immediate
    bin: Value
    enable: Immediate
    duration: Immediate


Io = Union[UpdParam, Play, Acquire, AcquireWeighed, AcquireTtl]
"""Real-time IO operation instructions.

The execution of any of these instructions will cause the latched
parameters to be updated.
"""


class SetLatchEn(Instr):
    """Toggle trigger counters.

    Enable/disable all trigger network address counters. Once enabled,
    the trigger network address counters will count all triggers on the
    trigger network. When disabled, the counters hold their last values.
    """

    enable: Value
    duration: Immediate


class LatchRst(Instr):
    """Reset trigger counters.

    Reset all trigger network address counters back to 0.
    """

    duration: Value


Trigger = Union[SetLatchEn, LatchRst]
"""Real-time trigger count control instructions."""


class Wait(Instr):
    """Wait.

    Wait for :attr:`duration` nanoseconds.
    """

    duration: Value


class WaitTrigger(Instr):
    """Wait trigger.

    Wait for a trigger on the trigger network at the address set using :attr:`trigger`.
    """

    trigger: Value
    duration: Value

    @model_validator(mode="after")
    def check_signature(self):
        assert type(self.trigger) is type(self.duration)
        return self


class WaitSync(Instr):
    """Syncronization.

    Wait for SYNQ to complete on all connected sequencers over all
    connected instruments.
    """

    duration: Value


WaitOps = Union[Wait, WaitTrigger, WaitSync]
"""Real-time wait operation instructions."""

RealTimeInstr = Union[Conditional, Io, Trigger, WaitOps]
"""Real-time instructions.

These instructions have a duration argument, which corresponds to the
wall-time of an experiment. The duration is counted from the beginning
of an instruction. This is what enables e.g. parallel playback and
acquisition. The duration can be set at a resolution of 1 ns, with a
minimum of 4 ns.

The real-time instructions to the pipeline form a time-line of real-time
operations. To ensure deterministic behavior, this time-line cannot be
interrupted/broken. As a result, once the real-time pipeline is started
by the first real-time instruction in a sequence, it cannot be stalled.
Therefore, the real-time pipeline will never wait for the Q1 processor.
In case of a conflict, the sequencer will halt and raise an error flag.
"""

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
    label: Optional[str] = None
    comment: Optional[Annotated[str, AfterValidator(lambda c: c.strip())]] = None

    def __rich_repr__(self):
        yield self.instruction
        yield "label", self.label, None
        yield "comment", self.comment, None

    def asm(
        self,
        width: Optional[int] = None,
        label_width: Optional[int] = None,
        instr_name_width: Optional[int] = None,
        instr_width: Optional[int] = None,
    ) -> str:
        if label_width is None:
            label_width = len(self.label) if self.label is not None else -2
        label = f"{self.label}:" if self.label is not None else ""
        if instr_name_width is None:
            instr_name_width = len(self.instruction.keyword())
        if instr_width is None:
            instr_width = len(self.instruction.asm(instr_name_width))
        code = f"{label:<{label_width+2}}{self.instruction.asm(instr_name_width):<{instr_width+1}}"
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
            (
                len(line.label)
                for line in self.elements
                if isinstance(line, Line) and line.label is not None
            ),
            default=None,
        )
        max_instr_name_len = max(
            (
                len(line.instruction.keyword())
                for line in self.elements
                if isinstance(line, Line)
            ),
            default=None,
        )
        max_instr_len = max(
            (
                len(line.instruction.asm(max_instr_name_len))
                for line in self.elements
                if isinstance(line, Line)
            ),
            default=None,
        )
        code = "\n".join(
            (
                (el if comments else el.model_copy(update={"comment": None})).asm(
                    width,
                    label_width=max_label_len,
                    instr_name_width=max_instr_name_len,
                    instr_width=max_instr_len,
                )
                if isinstance(el, Line)
                else (el.asm(width) if comments else "")
            )
            for el in self.elements
        )
        return (
            code if comments else re.sub("^\n*", "", re.sub("\n+", "\n", code))
        ) + "\n"
