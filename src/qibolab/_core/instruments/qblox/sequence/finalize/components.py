from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

from qibolab._core.serialize import Model

from ...q1asm.ast_ import Block, Line

State = TypeVar("State")
"""Generic carried-on state."""
Match: TypeAlias = Callable[[Line, State], tuple[bool, State]]
"""Matching condition."""
LineMap: TypeAlias = Callable[[Line, State], tuple[Block, State]]
"""Transformation acting on individual lines."""
BlockMap: TypeAlias = Callable[[list[Line], State], tuple[Block, State]]
"""Transformation acting on a block of instructions."""


class LineRule(Model, Generic[State]):
    """Line-matching transformation.

    ``init_state`` creates fresh state for each traversal. For every input line,
    ``match`` returns whether to apply ``map`` and the updated state. Both callbacks'
    state updates are carried to the next line, even when ``match`` returns false.
    The replacement is not matched again during the same step.
    """

    init_state: Callable[[], State]
    """Create independent state for each traversal."""
    match: Match[State]
    """Matching condition."""
    map: LineMap[State]
    "The transformation applied."

    def map_annotate(self, line: Line, state: State) -> tuple[Block, State]:
        """Apply map preserving annotations, i.e. label and comment."""
        block, state = self.map(line, state)
        if len(block) == 0:
            return block, state
        lineable = block[0]
        instr = lineable.instruction if isinstance(lineable, Line) else lineable
        annotated = [
            Line(instruction=instr, label=line.label, comment=line.comment),
            *block[1:],
        ]
        return annotated, state


class BlockRule(Model, Generic[State]):
    """Block-matching transformation.

    ``init_state`` creates fresh state for each traversal. Outside a block, ``start``
    is tested on each line; a matching line becomes the first line of the block.
    While collecting a block, only ``end`` is tested on subsequent lines. A matching
    end line is excluded from the block passed to ``map``.

    That end line is also tested with ``start``, before mapping the completed block:
    it either starts the next block or passes through unchanged. State returned by
    every condition and by ``map`` is retained, including when a condition does not
    match. At end of input, any open block is mapped as well. Replacement lines are
    not matched again in this step.

    For example, consecutive waits can be collected by starting on an immediate wait
    and ending before a non-wait or a labeled line. A labeled wait then starts a new
    block, preserving its role as a jump target.
    """

    init_state: Callable[[], State]
    """Create independent state for each traversal."""
    start: Match[State]
    """Start a block, including the matching line."""
    end: Match[State]
    """End a block before the matching line."""
    map: BlockMap[State]
    "The transformation applied."


Step: TypeAlias = tuple[LineRule, ...] | LineRule | BlockRule
"""Transformation step.

It is supposed to iterate through the entire set of instructions exactly once.

It can be composed of:

- a single line transformation or a tuple of alternative line transformations
- a single block-matching transformation

Semantics
---------

The reason for the type of allowed steps is to unambiguously define how they match the
instructions. Keeping the intuition straightforward is explicitly favored over efficiency.

If multiple line-oriented transformations are specified, they are tested first-to-last
against each instruction line. The first one which matches, applies the transformation,
otherwise, the line passes through unchanged. Replacements are concatenated into a
flat list of lines.

A block-oriented step uses exactly one rule. Only the first matching block is
collected: new starts within it are not tested until its end condition matches.
This avoids ambiguity between overlapping blocks or competing block rules.

.. note::
    It would be possible to combine multiple block-matching rules and even line-matching
    ones just applying the algorithmic choice mentioned last. But this is intentionally
    not supported.
"""


Pipeline: TypeAlias = tuple[Step, ...]
"""Series of transformation from Q1ASM-like code into executable.

A pipeline is transforming a possibly internal extension of Q1ASM into a different one.

It can be composed by multiple :class:`Step`, which are expected to sequentially read
the entire list of instructions.

The main purposes are intended to be:

- transform minor extensions into valid instructions

    - e.g. turning long waits into equivalent loops, which are respecting the
      limitations for the wait instructions

- supplying predictable instructions, that are consciously neglected in the code
  generation, assuming a downstream pipeline centralization

    - e.g. adding ``nop`` instructions after registers' manipulations, to guarantee the
      validity at consumption site

- optimizing the execution, when known patterns are identified

    - some optimizations are already possible at the level of the generation, but
      sometimes it may be convenient to inspect the code a posteriori, since some
      patterns may arise from close combinations that are generated by independent
      branches in the call stack
    - e.g. phases shifts by immediates can be generated by multiple instructions, and it
      can not be possible to simply combine them ahead of time, but it could be easier
      during a post-inspection

.. note::
    Some transformations could be locally disabled by providing markers in the comments
    which are recognized during the rule applications, and just stripped, with no other
    consequent action.

    In general, comments provide free space for optimization hints. Both at line and
    block level.

Each step consumes and returns a flat block. Steps run once in the specified order;
the pipeline does not repeat until a fixed point is reached.
"""
