from collections import defaultdict
from functools import reduce
from typing import Callable, Optional, cast

from pydantic import BaseModel, Field

from ..q1asm.ast_ import (
    Block,
    Instruction,
    Line,
    Loop,
    Move,
    Program,
    Reference,
    Register,
    Sub,
    UpdParam,
    Wait,
)
from .asm import Registers

__all__ = []

MAX_WAIT = 2**16 - 1


class State(BaseModel):
    pass


class FirstPassState(State):
    """
    - nwait: Tracks the number of Wait instructions encountered. This is used to label
      waits, and to coordinate duration adjustments due to UpdParam instructions.
    - nwait_to_subtract: Maps each wait index to the total duration that should be
      subtracted from that wait due to UpdParam instructions. This subtraction is done
      in the second pass.
    """

    nwait: int = 0
    nwait_to_subtract: defaultdict[int, int] = defaultdict(int)


class SecondPassState(State):
    """``nwait_to_subtract`` is set during the first pass and remains unchanged here.
    The first Wait we encounter in the second loop has id 1, not id 0, therefore we
    initialize nwait with an offset of 1. Key 0 in ``nwait_to_subtract`` is non-zero due
    to the initial UpdParam that is prepended for all channels and therefore  does not
    need to be compensated.
    """

    nwait: int = 1
    nwait_to_subtract: defaultdict[int, int] = defaultdict(int)


def _long_wait(duration: int, n: int) -> Block:
    """Split a statically long wait.

    It accounts for the wait instruction limit, defined by :const:`MAX_WAIT`.

    ``n`` is used for labelling the loop, and it should be different for each wait
    instruction in a sequencer.
    """
    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Registers.wait.value
    label = f"wait{n}"
    return [Wait(duration=remainder)] + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]


def _decompose_wait(instr: Wait, n: int) -> Optional[Block]:
    """
    Decompose a wait instruction into a loop if its duration exceeds MAX_WAIT, splitting
    it into a remainder and repeated MAX_WAIT-sized waits to fit hardware limits.
    """
    duration = instr.duration
    if not isinstance(duration, int) or duration <= MAX_WAIT:
        return None
    return _long_wait(duration, n)


def _negative_move(instr: Move):
    """Compile negative value sets.

    Apparently, the only place where negative numbers are not allowed
    are registers, otherwise they are handled by the internal compiler.

    https://docs.qblox.com/en/main/tutorials/q1asm_tutorials/intermediate/nco_control_adv.html#:~:text=Internally,%20the%20processor%20stores

    Thus, we compile instructions setting negative values as suggested:
    first setting them to 0, than subtracting the desired amount. This
    is more reliable than manually complementing the number, since it
    makes no assumption about the registers size.

    https://docs.qblox.com/en/main/cluster/troubleshooting.html#:~:text=How%20do%20I%20set%20negative%20numbers
    """
    src = cast(int, instr.source)
    dest = instr.destination
    return [Move(source=0, destination=dest), Sub(a=dest, b=abs(src), destination=dest)]


def _decompose_move(instr: Move) -> Optional[Block]:
    src = instr.source
    if isinstance(src, Register):
        return None
    assert isinstance(src, int)
    if src >= 0:
        return None
    return _negative_move(instr)


LineTransformed = list[Line]


def _first_pass(
    line: Line, state: FirstPassState
) -> tuple[LineTransformed, FirstPassState]:
    """Decomposes long Wait and negative Move instructions into valid Q1ASM blocks if
    needed, updating the state (e.g., wait counter). Returns the transformed lines and
    updated state. All other instructions are returned unchanged.
    """
    instr = line.instruction
    block, state = (
        # if Wait, increment state.nwait and decompose into a loop if needed
        (
            _decompose_wait(instr, state.nwait),
            state.model_copy(update={"nwait": state.nwait + 1}),
        )
        if isinstance(instr, Wait)
        else (_decompose_move(instr), state)
        if isinstance(instr, Move)
        # if UpdParam with duration, update state.nwait_to_subtract to account for the
        # wait time that will be subtracted from the Wait instruction in the _second_pass
        else (
            None,
            state.model_copy(
                update={
                    "nwait_to_subtract": state.nwait_to_subtract
                    | {
                        state.nwait: state.nwait_to_subtract[state.nwait]
                        + instr.duration
                    }
                }
            ),
        )
        if isinstance(instr, UpdParam)
        else (None, state)
    )

    # default
    if block is None:
        return [line], state

    assert isinstance(block[0], Instruction)
    return [
        el if isinstance(el, Line) else Line.instr(el)
        for el in (
            (
                Line(instruction=block[0], label=line.label, comment=line.comment),
                *(el for el in block[1:]),
            )
            if block is not None
            else [line]
        )
    ], state


def _second_pass(
    block: LineTransformed, state: SecondPassState
) -> tuple[LineTransformed, SecondPassState]:
    """Subtracts the additional duration incurred due to UpdParam from Wait instructions
    to ensure alignment between channels.
    """
    # this is only true in the case of the wait blocks merged in step 1.
    if len(block) > 1:
        instr = block[0].instruction
        assert isinstance(instr, Wait) and isinstance(instr.duration, int)
        new_duration = instr.duration - state.nwait_to_subtract[state.nwait]
        new_line = Line(
            label=block[0].label,
            instruction=Wait(duration=new_duration),
            comment=block[0].comment,
        )
        return [new_line, *block[1:]], state.model_copy(
            update={"nwait": state.nwait + 1}
        )
    return block, state


def _line_transform_apply(f: Callable) -> Callable:
    def reduction(
        value: tuple[list[LineTransformed], State], line: Line
    ) -> tuple[list[LineTransformed], State]:
        """Accumulate."""
        transformed, state = f(line, value[1])
        return (value[0] + [transformed]), state

    return reduction


class _WaitBatch(BaseModel):
    duration: int = 0
    comment: list[str] = Field(default_factory=list)
    label: Optional[str] = None

    def increment(self, line: Line):
        instr = line.instruction
        assert isinstance(instr, Wait) and isinstance(instr.duration, int)
        self.duration += instr.duration
        if line.comment is not None:
            self.comment.append(line.comment)
        if line.label is not None:
            self.label = line.label

    @property
    def lines(self) -> list[Line]:
        return (
            [
                Line(
                    label=self.label,
                    instruction=Wait(duration=self.duration),
                    comment="\n".join(self.comment),
                )
            ]
            if self.duration > 0
            else []
        )


def _merge_wait(block: list[Line]) -> list[Line]:
    """Merge subsequent static (immediate) waits."""
    batch = _WaitBatch()
    new = []
    for line in block:
        instr = line.instruction
        intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
        if not intwait or line.label is not None:
            new += batch.lines + ([line] if not intwait else [])
            batch = _WaitBatch()
        if intwait:
            batch.increment(line)

    return new + batch.lines


def _block_transform(block: Block) -> list[Line]:
    lines = [el if isinstance(el, Line) else Line.instr(el) for el in block]
    return _merge_wait(lines)


def transpile(prog: Block) -> Program:
    block_replaced = _block_transform(prog)
    lines_first_pass, first_pass_state = reduce(
        _line_transform_apply(_first_pass), block_replaced, ([], FirstPassState())
    )
    lines_second_pass, _state = reduce(
        _line_transform_apply(_second_pass),
        lines_first_pass,
        ([], SecondPassState(nwait_to_subtract=first_pass_state.nwait_to_subtract)),
    )
    return Program(elements=[line for block in lines_second_pass for line in block])
