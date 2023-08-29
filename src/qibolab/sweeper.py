import operator
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Optional

import numpy.typing as npt


class Parameter(Enum):
    """Sweeping parameters."""

    frequency = auto()
    amplitude = auto()
    duration = auto()
    relative_phase = auto()
    start = auto()

    attenuation = auto()
    gain = auto()
    bias = auto()
    lo_frequency = auto()


FREQUENCY = Parameter.frequency
AMPLITUDE = Parameter.amplitude
DURATION = Parameter.duration
RELATIVE_PHASE = Parameter.relative_phase
START = Parameter.start
ATTENUATION = Parameter.attenuation
GAIN = Parameter.gain
BIAS = Parameter.bias


class SweeperType(Enum):
    """Type of the Sweeper"""

    ABSOLUTE = partial(lambda x, y=None: x)
    FACTOR = operator.mul
    OFFSET = operator.add


QubitParameter = {Parameter.bias, Parameter.attenuation, Parameter.gain}


@dataclass
class Sweeper:
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.platforms.abstract.Platform.sweep`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.platforms.abstract.Platform.sweep`.

    Example:
        .. testcode::

            import numpy as np
            from qibolab.dummy import create_dummy
            from qibolab.sweeper import Sweeper, Parameter
            from qibolab.pulses import PulseSequence
            from qibolab import ExecutionParameters


            platform = create_dummy()
            sequence = PulseSequence()
            parameter = Parameter.frequency
            pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
            sequence.add(pulse)
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(parameter, parameter_range, [pulse])
            platform.sweep(sequence, ExecutionParameters(), sweeper)

    Args:
        parameter (`qibolab.sweeper.Parameter`): parameter to be swept, possible choices are frequency, attenuation, amplitude, current and gain.
        _values (np.ndarray): sweep range. If the parameter of the sweep is a pulse parameter, if the sweeper type is not ABSOLUTE, the base value
            will be taken from the runcard pulse parameters. If the sweep parameter is Bias, the base value will be the sweetspot of the qubits.
        pulses (list) : list of `qibolab.pulses.Pulse` to be swept (optional).
        qubits (list): list of `qibolab.platforms.abstract.Qubit` to be swept (optional).
        type (SweeperType): can be ABSOLUTE (the sweeper range is swept directly),
            FACTOR (sweeper values are multiplied by base value), OFFSET (sweeper values are added
            to base value)
    """

    parameter: Parameter
    values: npt.NDArray
    pulses: Optional[list] = None
    qubits: Optional[list] = None
    couplers: Optional[list] = None
    type: Optional[SweeperType] = SweeperType.ABSOLUTE

    def __post_init__(self):
        if self.pulses is not None and self.qubits is not None:
            raise ValueError("Cannot use a sweeper on both pulses and qubits.")
        if self.pulses is not None and self.parameter in QubitParameter:
            raise ValueError(f"Cannot sweep {self.parameter} without specifying qubits or couplers.")
        if self.qubits and self.couplers is not None and self.parameter not in QubitParameter:
            raise ValueError(f"Cannot sweep {self.parameter} without specifying pulses.")
        if self.pulses is None and self.qubits is None and self.couplers is None:
            raise ValueError("Cannot use a sweeper without specifying pulses, qubits or couplers.")

    def get_values(self, base_value):
        """Convert sweeper values depending on the sweeper type"""
        return self.type.value(self.values, base_value)
