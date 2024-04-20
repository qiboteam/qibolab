from dataclasses import dataclass
from typing import Optional

from qm import qua

from qibolab.pulses import Delay, PulseType

from .acquisition import Acquisition
from .config import element, operation


@dataclass
class Parameters:
    # TODO: Split acquisition and sweep parameters
    acquisition: Optional[Acquisition] = None
    # TODO: Change the following types to QUA variables
    duration: Optional[int] = None
    amplitude: Optional[float] = None
    phase: Optional[float] = None


def _delay(pulse):
    # TODO: How to play delays on multiple elements?
    qua.wait(pulse.duration // 4 + 1, element(pulse))


def _play(pulse, parameters):
    el = element(pulse)
    if parameters.phase is not None:
        qua.frame_rotation_2pi(parameters.phase, el)
    if parameters.amplitude is not None:
        op = operation(pulse) * parameters.amplitude
    else:
        op = operation(pulse)

    if pulse.type is PulseType.READOUT:
        parameters.acquisition.measure(op)
    else:
        qua.play(op, el, duration=parameters.duration)

    if parameters.phase is not None:
        qua.reset_frame(el)


def play(sequence, parameters, relaxation_time=0):
    """Part of QUA program that plays an arbitrary pulse sequence.

    Should be used inside a ``program()`` context.
    """
    qua.align()
    for pulse in sequence:
        if isinstance(pulse, Delay):
            _delay(pulse)
        else:
            _play(pulse, parameters[operation(pulse)])

    if relaxation_time > 0:
        qua.wait(relaxation_time // 4)
