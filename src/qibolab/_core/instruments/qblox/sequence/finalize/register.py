from ...q1asm.ast_ import Arithmetic, Block, Line, Nop
from .components import LineRule

__all__ = ["update_nop"]


def _match_update_nop(line: Line, state: None) -> tuple[bool, None]:
    return isinstance(line.instruction, Arithmetic), state


def _map_update_nop(line: Line, state: None) -> tuple[Block, None]:
    return [line, Nop()], state


update_nop = LineRule[None](match=_match_update_nop, map=_map_update_nop)
"""Wait one clock cycle after arithmetic instructions for register propagation.

Apply after instruction expansion so generated register updates are covered too.

https://docs.qblox.com/en/main/products/architecture/sequencers/sequencer.html#registers
"""
