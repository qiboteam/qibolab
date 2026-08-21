from functools import reduce
from typing import get_args

from typing_extensions import TypeIs

from ...q1asm.ast_ import Block, Line, Lineable, block_to_lines
from .components import BlockRule, LineRule, Pipeline, State, Step


def _is_block(instructions: Block | list[Block]) -> TypeIs[Block]:
    return isinstance(instructions[0], Lineable)


def _to_lines(instructions: Block | list[Block]) -> list[Line]:
    return block_to_lines(
        instructions
        if _is_block(instructions)
        else [el for block in instructions for el in block]
    )


def _init_state(rule: LineRule[State] | BlockRule[State]) -> State:
    map_type = type(rule).model_fields["map"]
    return_type = get_args(map_type)[1]
    state_type = get_args(return_type)[1]
    return state_type()


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
    state = _init_state(rule)
    result = []

    # current block
    block = []

    for line in lines:
        if len(block) == 0:
            match, state = rule.initial(line, state)
            if match:
                block.append(line)
            else:
                result.append([line])
        else:
            match, state = rule.final(line, state)
            block.append(line)
            if match:
                mapped, state = rule.map(block, state)
                result.append(mapped)

    return result


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
