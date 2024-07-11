from dataclasses import dataclass
from typing import Optional

from qm import qua

from qibolab.pulses import Delay, PulseType

from .acquisition import Acquisition
from .config import operation


@dataclass
class Parameters:
    # TODO: Split acquisition and sweep parameters
    acquisition: Optional[Acquisition] = None
    # TODO: Change the following types to QUA variables
    duration: Optional[int] = None
    amplitude: Optional[float] = None
    phase: Optional[float] = None


def _delay(pulse, element, parameters):
    # TODO: How to play delays on multiple elements?
    if parameters.duration is None:
        duration = pulse.duration // 4 + 1
    else:
        duration = parameters.duration
    qua.wait(duration, element)


def _play(pulse, element, parameters):
    if parameters.phase is not None:
        qua.frame_rotation_2pi(parameters.phase, element)
    if parameters.amplitude is not None:
        op = operation(pulse) * parameters.amplitude
    else:
        op = operation(pulse)

    if pulse.type is PulseType.READOUT:
        parameters.acquisition.measure(op)
    else:
        qua.play(op, element, duration=parameters.duration)

    if parameters.phase is not None:
        qua.reset_frame(element)


def play(sequence, parameters, relaxation_time=0):
    """Part of QUA program that plays an arbitrary pulse sequence.

    Should be used inside a ``program()`` context.
    """
    qua.align()
    for element, pulses in sequence.items():
        for pulse in pulses:
            params = parameters[operation(pulse)]
            if isinstance(pulse, Delay):
                _delay(pulse, element, params)
            else:
                _play(pulse, element, params)

    if relaxation_time > 0:
        qua.wait(relaxation_time // 4)
