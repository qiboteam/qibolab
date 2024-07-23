from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from qm import qua

from qibolab.pulses import Delay, PulseSequence

from .acquisition import Acquisition
from .config import operation


@dataclass
class Parameters:
    # TODO: Change the following types to QUA variables
    duration: Optional[int] = None
    amplitude: Optional[float] = None
    phase: Optional[float] = None


def _delay(pulse: Delay, element: str, parameters: Parameters):
    # TODO: How to play delays on multiple elements?
    if parameters.duration is None:
        duration = pulse.duration // 4 + 1
    else:
        duration = parameters.duration
    qua.wait(duration, element)


def _play(
    op: str,
    element: str,
    parameters: Parameters,
    acquisition: Optional[Acquisition] = None,
):
    if parameters.phase is not None:
        qua.frame_rotation_2pi(parameters.phase, element)
    if parameters.amplitude is not None:
        op = op * parameters.amplitude

    if acquisition is not None:
        acquisition.measure(op)
    else:
        qua.play(op, element, duration=parameters.duration)

    if parameters.phase is not None:
        qua.reset_frame(element)


@dataclass
class ExecutionArguments:
    sequence: PulseSequence
    acquisitions: dict[tuple[str, str], Acquisition]
    relaxation_time: int = 0
    parameters: dict[str, Parameters] = field(
        default_factory=lambda: defaultdict(Parameters)
    )


def play(args: ExecutionArguments):
    """Part of QUA program that plays an arbitrary pulse sequence.

    Should be used inside a ``program()`` context.
    """
    qua.align()
    for element, pulses in args.sequence.items():
        for pulse in pulses:
            op = operation(pulse)
            params = args.parameters[op]
            if isinstance(pulse, Delay):
                _delay(pulse, element, params)
            else:
                acquisition = args.acquisitions.get((op, element))
                _play(op, element, params, acquisition)

    if args.relaxation_time > 0:
        qua.wait(args.relaxation_time // 4)
