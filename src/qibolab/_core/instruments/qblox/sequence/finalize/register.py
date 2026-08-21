from ...q1asm.ast_ import Block, Line
from .components import LineRule

__all__ = ["update_nop"]


def _match_update_nop(line: Line, state: None) -> tuple[bool, None]:
    # TODO:
    return not isinstance(line, Line), state


def _map_update_nop(line: Line, state: None) -> tuple[Block, None]:
    # TODO:
    return [line], state


update_nop = LineRule[None](match=_match_update_nop, map=_map_update_nop)
