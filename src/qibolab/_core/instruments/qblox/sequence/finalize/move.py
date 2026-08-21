from typing import cast

from ...q1asm.ast_ import Block, Line, Move, Sub
from .components import LineRule

__all__ = ["negative_immediate"]


def _match_negative_immediate(line: Line, state: None) -> tuple[bool, None]:
    instr = line.instr
    match = (
        isinstance(instr, Move) and isinstance(instr.source, int) and instr.source < 0
    )
    return match, state


def _map_negative_immediate(line: Line, state: None) -> tuple[Block, None]:
    instr = line.instr
    assert isinstance(instr, Move)
    src = cast(int, instr.source)
    dest = instr.destination
    return [
        Move(source=0, destination=dest),
        Sub(a=dest, b=abs(src), destination=dest),
    ], None


negative_immediate = LineRule[None](
    match=_match_negative_immediate, map=_map_negative_immediate
)
"""Compile negative value sets.

Apparently, the only place where negative numbers are not allowed are registers,
otherwise they are handled by the internal compiler.

https://docs.qblox.com/en/main/tutorials/q1asm_tutorials/intermediate/nco_control_adv.html#:~:text=Internally,%20the%20processor%20stores

Thus, we compile instructions setting negative values as suggested: first setting them
to 0, than subtracting the desired amount. This is more reliable than manually
complementing the number, since it makes no assumption about the registers size.

https://docs.qblox.com/en/main/cluster/troubleshooting.html#:~:text=How%20do%20I%20set%20negative%20numbers
"""
