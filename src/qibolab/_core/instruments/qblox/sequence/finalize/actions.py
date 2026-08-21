from functools import reduce

from typing_extensions import TypeIs

from ...q1asm.ast_ import Block, Line, Lineable, block_to_lines
from .components import BlockRule, LineRule, Pipeline, Step


def _is_block(instructions: Block | list[Block]) -> TypeIs[Block]:
    return isinstance(instructions[0], Lineable)


def _to_lines(instructions: Block | list[Block]) -> list[Line]:
    return block_to_lines(
        instructions
        if _is_block(instructions)
        else [el for block in instructions for el in block]
    )


def _line_traverse(lines: list[Line], rule: tuple[LineRule, ...]) -> list[Block]:
    """...

    .. todo::

            return [
                    for el in (
                        (
                            Line(instruction=block[0], label=line.label, comment=line.comment),
                            *(el for el in block[1:]),
                        )
                        if block is not None
                        else [line]
                    )
                ], state
    """
    return []


def _block_traverse(lines: list[Line], rule: BlockRule) -> list[Block]:
    return []


def traverse(instructions: Block | list[Block], step: Step) -> list[Block]:
    lines = _to_lines(instructions)
    return (
        _block_traverse(lines, step)
        if isinstance(step, BlockRule)
        else _line_traverse(lines, step)
    )


def transform(block: Block, pipeline: Pipeline) -> list[Line]:
    """...

    .. todo::
        return ``list[Line]`` to be more general -> still need to wrap into a
        ``Program(elements=...)`` at call site.
    """
    return _to_lines(reduce(traverse, pipeline, block))
