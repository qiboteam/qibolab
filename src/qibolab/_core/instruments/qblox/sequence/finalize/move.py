from ...q1asm.ast_ import Block, Line
from .components import LineRule

__all__ = ["negative_immediate"]


def _match_negative_immediate(line: Line, state: None) -> tuple[bool, None]:
    # TODO:
    return not isinstance(line, Line), state


def _map_negative_immediate(line: Line, state: None) -> tuple[Block, None]:
    # TODO:
    return [line], state


negative_immediate = LineRule[None](
    match=_match_negative_immediate, map=_map_negative_immediate
)
