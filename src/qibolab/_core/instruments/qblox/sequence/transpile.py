import typing
from collections import defaultdict
from functools import reduce
from typing import Callable, Optional, TypeGuard

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

LineBlock = list[Line]


class State(BaseModel):
    """
    - **last_encountered_static**: Tracks the number of Wait or Play instructions
      encountered that have a static duration. This is used to provide unique labels for
      loops around waits, and to coordinate duration adjustments due to UpdParam
      instructions.
    - **subtract_from_static**: Maps each real-time instruction with a static duration,
      to the total duration that should be subtracted from that instruction due to
      UpdParam instructions that _follow_ it. This subtraction is done in the second
      pass.
    - **last_encountered_register**: Tracks the register number most recently seen in a
      Wait(Register) or Play(Register) instruction, used to associate following UpdParam
      durations with the correct register for shifting.
    - **register_shifts**: Maps register numbers to the total duration that should be
      subtracted from their initialization, for Waits or Plays using registers.
    """

    last_encountered_static: int = 0
    subtract_from_static: defaultdict[int, int] = defaultdict(int)
    last_encountered_register: Optional[int] = None
    register_shifts: defaultdict[int, int] = defaultdict(int)

    def increment_rt_counter(self) -> "State":
        return self.set_last_encountered_register(None).model_copy(
            update={"last_encountered_static": self.last_encountered_static + 1}
        )

    def record_static_shift(self, duration: int) -> "State":
        key = self.last_encountered_static
        return self.model_copy(
            update={
                "subtract_from_static": self.subtract_from_static
                | {key: self.subtract_from_static[key] + duration}
            }
        )

    def set_last_encountered_register(self, regnum: int | None) -> "State":
        return self.model_copy(update={"last_encountered_register": regnum})

    def record_register_shift(self, shift: int) -> "State":
        key = self.last_encountered_register
        assert key is not None
        return self.model_copy(
            update={
                "register_shifts": self.register_shifts
                | {key: self.register_shifts[key] + shift}
            }
        )

    @property
    def duration_to_subtract(self) -> int:
        return self.subtract_from_static[self.last_encountered_static]


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


def _convert_to_lines(decomposed: Block, origin: Line) -> LineBlock:
    """Convert a Block into Lines, applying the origin's label and comment
    to the first instruction."""
    assert isinstance(decomposed[0], Instruction)
    return [
        Line(instruction=decomposed[0], label=origin.label, comment=origin.comment),
        *[el if isinstance(el, Line) else Line.instr(el) for el in decomposed[1:]],
    ]


def _negative_move(instr: Move) -> Block:
    """Compile negative value sets.

    Apparently, the only place where negative numbers are not allowed
    are registers, otherwise they are handled by the internal compiler.

    https://docs.qblox.com/en/main/tutorials/q1asm_tutorials/intermediate/nco_control_adv.html#:~:text=Internally,%20the%20processor%20stores

    Thus, we compile instructions setting negative values as suggested:
    first setting them to 0, then subtracting the desired amount. This
    is more reliable than manually complementing the number, since it
    makes no assumption about the registers size.

    https://docs.qblox.com/en/main/cluster/troubleshooting.html#:~:text=How%20do%20I%20set%20negative%20numbers
    """
    src = instr.source
    assert isinstance(src, int) and src < 0
    dest = instr.destination
    return [Move(source=0, destination=dest), Sub(a=dest, b=abs(src), destination=dest)]


def _decompose_move(instr: Move) -> Optional[Block]:
    src = instr.source
    if isinstance(src, Register) or (isinstance(src, int) and src >= 0):
        return None
    return _negative_move(instr)


class _StaticRealTimeInstruction(typing.Protocol):
    duration: int


def _is_static_wait(instr: Instruction) -> TypeGuard[_StaticRealTimeInstruction]:
    return isinstance(instr, Wait) and isinstance(instr.duration, int)


def _is_static_play(instr: Instruction) -> TypeGuard[_StaticRealTimeInstruction]:
    return isinstance(instr, Play) and isinstance(instr.duration, int)


def _first_pass(block: LineBlock, state: State) -> tuple[LineBlock, State]:
    """Decomposes long Wait and negative Move instructions into valid Q1ASM blocks if
    needed, updating the state (e.g., wait counter). Returns the transformed lines and
    updated state. All other instructions are returned unchanged.
    """
    # _first_pass is called through _line_transform_apply and therefore receives as
    # input a block, even though prior to the first pass each block contains only a
    # single Line.
    assert len(block) == 1
    line = block[0]
    instr = line.instruction

    if _is_static_wait(instr) or _is_static_play(instr):
        return block, state.increment_rt_counter()

    if isinstance(instr, UpdParam):
        if state.last_encountered_register is not None:
            return block, state.record_register_shift(instr.duration)
        return block, state.record_static_shift(instr.duration)

    if isinstance(instr, Move):
        decomposed = _decompose_move(instr)
        if decomposed is None:
            return block, state
        decomposed_block = _convert_to_lines(decomposed, line)
        return decomposed_block, state

    if (isinstance(instr, Wait) or isinstance(instr, Play)) and isinstance(
        instr.duration, Register
    ):
        regnum = instr.duration.number
        state = state.set_last_encountered_register(regnum)
        return block, state

    return block, state


def _adjust_realtime_instruction_duration(block: LineBlock, state: State) -> LineBlock:
    """
    Adjusts the duration of Wait or Play instructions by subtracting any additional
    duration incurred due to following UpdParam instructions, ensuring correct timing.
    """

    # Wait can either be in a block generated by _decompose_wait or in a block on
    # its own. In the former case the block consists of a remainder (index 0) and a
    # loop (indices 1-3). In that case we want to subtract the duration from the
    # wait at index 0 since the loop is repeated 3 times and duration/3 might not be
    # an integer value.
    instr = block[0].instruction
    assert isinstance(instr, Play) or isinstance(instr, Wait)
    assert isinstance(instr.duration, int)
    new_duration = instr.duration - state.duration_to_subtract
    if new_duration < 0:
        raise ValueError(
            "Wait/Play duration underflow: computed duration is negative: "
            f"{new_duration}. Possibly there are too many uninterrupted virtual-Z "
            " gates. If so, consider merging them."
        )
    updated_block = [
        block[0].model_copy(
            update={"instruction": instr.model_copy(update={"duration": new_duration})}
        ),
        *block[1:],
    ]
    return updated_block


def _second_pass(block: LineBlock, state: State) -> tuple[LineBlock, State]:
    """Subtracts the additional duration incurred due to UpdParam from Wait or Play
    real-time instructions to ensure alignment between channels.
    """
    instr = block[0].instruction

    if _is_static_wait(instr) or _is_static_play(instr):
        block = _adjust_realtime_instruction_duration(block, state)

        # If needed, decompose long waits after duration has been corrected
        instr = block[0].instruction
        if _is_static_wait(instr) and instr.duration > MAX_WAIT:
            block = _convert_to_lines(
                _long_wait(instr.duration, state.last_encountered_static), block[0]
            )

        return block, state.increment_rt_counter()

    if (
        isinstance(instr, Move)
        and isinstance(instr.destination, Register)
        and instr.destination.number in state.register_shifts
    ):
        assert isinstance(instr.source, int)
        block_ = [
            instr.model_copy(
                update={
                    "source": instr.source
                    - state.register_shifts[instr.destination.number]
                }
            )
        ]
        return _convert_to_lines(block_, block[0]), state

    return block, state


def _line_transform_apply(
    f: Callable[[LineBlock, State], tuple[LineBlock, State]],
) -> Callable[
    [tuple[list[LineBlock], State], LineBlock],
    tuple[list[LineBlock], State],
]:
    def reduction(
        accumulator: tuple[list[LineBlock], State], block: LineBlock
    ) -> tuple[list[LineBlock], State]:
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
    def lines(self) -> LineBlock:
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


def _merge_wait(block: LineBlock) -> LineBlock:
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


def _block_transform(block: Block) -> LineBlock:
    """
    Converts all Lineables in a block to Line objects if not already, then merges
    consecutive static Wait instructions.
    """
    lines = [el if isinstance(el, Line) else Line.instr(el) for el in block]
    return _merge_wait(lines)


def transpile(prog: Block) -> Program:
    lines = _block_transform(prog)
    blocks_first_pass, first_pass_state = reduce(
        _line_transform_apply(_first_pass),
        [[line] for line in lines],
        ([], State()),
    )
    # since we subtract the duration upd_params that come after the RT instruction,
    # the last_encountered_static is initialized at 1.
    blocks_second_pass, _state = reduce(
        _line_transform_apply(_second_pass),
        blocks_first_pass,
        ([], first_pass_state.model_copy(update={"last_encountered_static": 1})),
    )
    return Program(elements=[line for block in blocks_second_pass for line in block])
