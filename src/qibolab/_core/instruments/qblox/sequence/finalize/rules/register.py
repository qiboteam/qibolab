from qibolab._core.sweeper import Parameter

from ....q1asm.ast_ import Add, Arithmetic, Block, Jlt, Line, Nop, Reference, Sub
from ...asm import MAX_PARAM, Registers
from ..components import LineRule

__all__ = ["update_nop", "wrap_phase_delta"]


def _match_update_nop(line: Line, state: None) -> tuple[bool, None]:
    return isinstance(line.instruction, Arithmetic), state


def _map_update_nop(line: Line, state: None) -> tuple[Block, None]:
    return [line, Nop()], state


update_nop = LineRule(
    init_state=lambda: None, match=_match_update_nop, map=_map_update_nop
)
"""Wait one clock cycle after arithmetic instructions for register propagation.

Apply after instruction expansion so generated register updates are covered too.

https://docs.qblox.com/en/main/products/architecture/sequencers/sequencer.html#registers
"""


_PERIOD = int(MAX_PARAM[Parameter.phase])


def _match_phase_delta_add(line: Line, state: int) -> tuple[bool, int]:
    instr = line.instruction
    return (
        isinstance(instr, Add) and instr.destination == Registers.phase_delta.value,
        state,
    )


def _map_phase_delta_add(line: Line, state: int) -> tuple[Block, int]:
    reg = Registers.phase_delta.value
    skip = f"phase_wrap_r{reg.number}_{state}"
    return [
        line,
        Line(
            instruction=Jlt(a=reg, b=_PERIOD, address=Reference(label=skip)),
            comment="wrap phase delta within a period",
        ),
        Sub(a=reg, b=_PERIOD, destination=reg),
        Line(instruction=Nop(), label=skip),
    ], state + 1


wrap_phase_delta = LineRule(
    init_state=lambda: 0,
    match=_match_phase_delta_add,
    map=_map_phase_delta_add,
)
"""Wrap the accumulated phase delta back below one period after each addition.

The register collects the previous undo value, virtual Zs and the relative phase of the
next pulse. Each addend is within a period, so one conditional subtraction is enough to
keep it in the range accepted by ``set_ph_delta`` and away from 32-bit overflow.

This rule is not part of the default pipeline, because it adds four instructions
(``jlt``, ``sub``, the propagation ``nop`` and the labelled ``nop``) for every addition
to the phase delta register. In long sequences with many virtual Zs and relative phases
(e.g. randomized benchmarking) this can significantly increase the program size, which
is bounded by the sequencer instruction memory.

It may be used when the accumulated phase delta can exceed one period before being
passed to ``set_ph_delta``, where values above the period are undefined behaviour, and
the program comfortably fits the instruction memory. In that case, add it to a custom
pipeline before :data:`update_nop`, which takes care of the register propagation after
the addition and the subtraction. Labels are numbered per traversal.
"""
