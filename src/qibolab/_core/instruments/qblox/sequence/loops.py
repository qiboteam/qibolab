from collections.abc import Sequence
from enum import Enum
from typing import Optional

from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, iteration_length

from ..q1asm.ast_ import Register

__all__ = []


class Registers(Enum):
    bin = Register(number=0)
    bin_reset = Register(number=1)
    shots = Register(number=2)


class Loop(Model):
    """Loop descriptor.

    Created by the :func:`loops` function.
    """

    reg: Register
    """Register used for the loop counter."""
    length: int
    """Iteration length."""
    id: Optional[int]
    """Iteration index.

    `None` marks the loop as the shots loop.
    """

    @property
    def description(self) -> str:
        """Sweeper textual description."""
        return f"sweeper {self.id + 1}" if self.id is not None else "shots"


def loops(
    sweepers: list[ParallelSweepers], nshots: int, inner_shots: bool
) -> Sequence[Loop]:
    """Initialize registers for loop counters.

    The counters implement the ``length`` of the iteration, which, for a general
    sweeper, is fully characterized by a ``(start, step, length)`` tuple.

    Those related to :attr:`Registers.bin` and :attr:`Registers.bin_reset` are actually
    not loop counter on their own, but they are required to properly store the
    acquisitions in different bins.
    """
    shots = Loop(reg=Registers.shots.value, length=nshots, id=None)
    first_sweeper = max(r.value.number for r in Registers) + 1
    sweep = [
        Loop(
            reg=Register(number=i + first_sweeper),
            length=iteration_length(parsweep),
            id=i,
        )
        for i, parsweep in enumerate(sweepers)
    ]
    return [shots] + sweep if inner_shots else sweep + [shots]
