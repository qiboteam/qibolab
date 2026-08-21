from qibolab._core.serialize import Model

from ...q1asm.ast_ import Block, Line
from .components import BlockRule, LineRule

__all__ = [
    "LongWaits",
    "long_wait",
    "merge_wait",
]


class LongWaits(Model):
    n: int = 0
    """Number of long waits expanded."""


def _match_long_wait(line: Line, state: LongWaits) -> tuple[bool, LongWaits]:
    # TODO:
    return not isinstance(line, Line), state


def _map_long_wait(line: Line, state: LongWaits) -> tuple[Block, LongWaits]:
    # TODO:
    return [line], state


long_wait = LineRule[LongWaits](match=_match_long_wait, map=_map_long_wait)


def _start_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    # TODO:
    return not isinstance(line, Line), None


def _end_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    # TODO:
    return not isinstance(line, Line), None


def _merge_wait(block: Block, state: None) -> tuple[Block, None]:
    # TODO:
    return block, None


merge_wait = BlockRule[None](
    initial=_start_merge_wait, final=_end_merge_wait, map=_merge_wait
)
