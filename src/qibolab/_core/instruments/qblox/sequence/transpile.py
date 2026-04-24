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
    Play,
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
    """
    - count_rt_instr: Tracks the number of Wait or Play instructions encountered. This
      is used to provide unique labels for loops around waits, and to coordinate
      duration adjustments due to UpdParam instructions.
    - subtract_from_count: Maps each wait index to the total duration that should be
      subtracted from that wait due to UpdParam instructions that **follow** it. This
      subtraction is done in the second pass.
    """

    count_rt_instr: int = 0
    subtract_from_count: defaultdict[int, int] = defaultdict(int)


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


def _negative_move(instr: Move) -> Block:
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
    assert src < 0
    dest = instr.destination
    return [Move(source=0, destination=dest), Sub(a=dest, b=abs(src), destination=dest)]


def _decompose_move(instr: Move) -> Optional[Block]:
    src = instr.source
    if isinstance(src, Register) or (isinstance(src, int) and src >= 0):
        return None
    return _negative_move(instr)


def _first_pass(block: Block, state: State) -> tuple[list[Line], State]:
    """Decomposes long Wait and negative Move instructions into valid Q1ASM blocks if
    needed, updating the state (e.g., wait counter). Returns the transformed lines and
    updated state. All other instructions are returned unchanged.
    """
    # _first_pass is called through _line_transform_apply and therefore receives as
    # input a list of Lines (a block), even though prior to the first pass each block
    # contains only a single Line.
    assert len(block) == 1
    line = block[0]
    instr = line.instruction
    # increment state.count_rt_instr upon encountering realtime instructions
    updated_block, updated_state = (
        (
            _decompose_wait(instr, state.count_rt_instr),
            state.model_copy(update={"count_rt_instr": state.count_rt_instr + 1}),
        )
        # we skip Waits in registers because we do not need to decompose them and for
        # second pass subtracting duration would require subtracting from the initial
        # value of the register, bt would be painful if the result becomes negative
        # which is not unlikely if a duration sweep starts at 0.
        if isinstance(instr, Wait) and not isinstance(instr.duration, Register)
        else (_decompose_move(instr), state)
        if isinstance(instr, Move)
        # if UpdParam with duration, update state.subtract_from_count to account for the
        # wait time that will be subtracted from the Wait instruction in the _second_pass
        else (
            None,
            state.model_copy(
                update={
                    "subtract_from_count": state.subtract_from_count
                    | {
                        state.count_rt_instr: state.subtract_from_count[
                            state.count_rt_instr
                        ]
                        + instr.duration
                    }
                }
            ),
        )
        if isinstance(instr, UpdParam)
        else (
            None,
            state.model_copy(update={"count_rt_instr": state.count_rt_instr + 1}),
        )
        if isinstance(instr, Play)
        else (None, state)
    )

    # default
    if updated_block is None:
        return [line], updated_state

    assert isinstance(updated_block[0], Instruction)
    return [
        Line(instruction=updated_block[0], label=line.label, comment=line.comment),
        *[el if isinstance(el, Line) else Line.instr(el) for el in updated_block[1:]],
    ], updated_state


def _second_pass(block: list[Line], state: State) -> tuple[list[Line], State]:
    """Subtracts the additional duration incurred due to UpdParam from Wait instructions
    to ensure alignment between channels.
    """
    instr = block[0].instruction
    if (
        isinstance(instr, Wait) and not isinstance(instr.duration, Register)
    ) or isinstance(instr, Play):
        # Wait can either be in a block generated by _decompose_wait or in a block on
        # its own. In the former case the block consists of a remainder (index 0) and a
        # loop (indices 1-3). In that case we want to subtract the duration from the
        # wait at index 0 since the loop is repeated 3 times and duration/3 might not be
        # an integer value.
        assert isinstance(instr.duration, int)
        new_duration = instr.duration - state.subtract_from_count[state.count_rt_instr]
        if new_duration < 0:
            raise ValueError(
                "Wait/Play duration underflow: computed duration is negative: "
                f"{new_duration}. Possibly there are too many uninterrupted virtual-Z "
                " gates. If so, consider merging them."
            )
        updated_block = [
            block[0].model_copy(
                update={
                    "instruction": instr.model_copy(update={"duration": new_duration})
                }
            ),
            *block[1:],
        ]
        return updated_block, state.model_copy(
            update={"count_rt_instr": state.count_rt_instr + 1}
        )
    return block, state


def _line_transform_apply(
    f: Callable[[list[Line], State], tuple[list[Line], State]],
) -> Callable[
    [tuple[list[list[Line]], State], list[Line]],
    tuple[list[list[Line]], State],
]:
    def reduction(
        accumulator: tuple[list[list[Line]], State], block: list[Line]
    ) -> tuple[list[list[Line]], State]:
        """Transform a block using f, and append to the accumulator."""
        transformed, state = f(block, accumulator[1])
        return (accumulator[0] + [transformed]), state

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
    """
    Converts all Lineables in a block to Line objects if not already, then merges
    consecutive static Wait instructions.
    """
    lines = [el if isinstance(el, Line) else Line.instr(el) for el in block]
    return _merge_wait(lines)


def transpile(prog: Block) -> Program:
    block_replaced = _block_transform(prog)
    blocks_first_pass, first_pass_state = reduce(
        _line_transform_apply(_first_pass),
        [[line] for line in block_replaced],
        ([], State()),
    )
    # since we subtract the duration upd_params that come after the RT instruction,
    # the count_rt_instr is initialized at 1.
    blocks_second_pass, _state = reduce(
        _line_transform_apply(_second_pass),
        blocks_first_pass,
        ([], first_pass_state.model_copy(update={"count_rt_instr": 1})),
    )
    return Program(elements=[line for block in blocks_second_pass for line in block])
