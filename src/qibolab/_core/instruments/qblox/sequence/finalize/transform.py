from functools import reduce

from ...q1asm.ast_ import Block, BlockList, Line, block_to_lines
from .components import BlockRule, LineRule, Pipeline, State, Step


def _line_traverse(
    lines: list[Line], rules: tuple[LineRule, ...] | LineRule
) -> list[Line]:
    if isinstance(rules, LineRule):
        rules = (rules,)
    states = [rule.init_state() for rule in rules]
    result: BlockList = []

    for line in lines:
        for index, rule in enumerate(rules):
            match, state = rule.match(line, states[index])
            states[index] = state
            if match:
                mapped, state = rule.map_annotate(line, state)
                states[index] = state
                result.extend(mapped)
                break
        else:
            result.append(line)

    return block_to_lines(result)


def _block_traverse(lines: list[Line], rule: BlockRule[State]) -> list[Line]:
    state = rule.init_state()
    result: BlockList = []

    # current block
    block: list[Line] = []

    for line in lines:
        if len(block) == 0:
            match, state = rule.start(line, state)
            if match:
                block.append(line)
            else:
                result.append(line)
        else:
            match, state = rule.end(line, state)
            if match:
                starts_next, state = rule.start(line, state)
                mapped, state = rule.map(block, state)
                result.extend(mapped)
                if starts_next:
                    block = [line]
                else:
                    result.append(line)
                    block = []
            else:
                block.append(line)

    # flush the block left open at the end of the program
    if len(block) > 0:
        mapped, state = rule.map(block, state)
        result.extend(mapped)

    return block_to_lines(result)


def traverse(instructions: Block, step: Step) -> list[Line]:
    """Apply one step, with fresh rule state, and return a flat list of lines."""
    lines = block_to_lines(instructions)
    return (
        _block_traverse(lines, step)
        if isinstance(step, BlockRule)
        else _line_traverse(lines, step)
    )


def transform(block: Block, pipeline: Pipeline) -> list[Line]:
    """Apply the pipeline in order and return a flat list of annotated lines."""
    return reduce(traverse, pipeline, block_to_lines(block))
