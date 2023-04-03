from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy.typing as npt


class Parameter(Enum):
    """Sweeping parameters."""

    frequency = auto()
    amplitude = auto()
    relative_phase = auto()

    attenuation = auto()
    gain = auto()
    bias = auto()


QubitParameter = {Parameter.bias, Parameter.attenuation, Parameter.gain}


@dataclass
class Sweeper:
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`.

    Example:
        .. testcode::

            import numpy as np
            from qibolab.platform import Platform
            from qibolab.sweeper import Sweeper, Parameter
            from qibolab.pulses import PulseSequence


            platform = Platform("dummy")
            sequence = PulseSequence()
            parameter = Parameter.frequency
            pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
            sequence.add(pulse)
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(parameter, parameter_range, [pulse])
            platform.sweep(sequence, sweeper)

    Args:
        parameter (`qibolab.sweeper.Parameter`): parameter to be swept, possible choices are frequency, attenuation, amplitude, current and gain.
        values (np.ndarray): sweep range. If the parameter is `frequency` the sweep will be a shift around the readout frequency
            in case of a `ReadoutPulse` or around the drive frequency for a generic `Pulse`. If the parameter is `amplitude` the range is
            normalized with the current amplitude of the pulse. For other parameters the sweep will be performed directly over the range specified.
        pulses (list) : list of `qibolab.pulses.Pulse` to be swept (optional).
        qubits (list): list of `qibolab.platforms.abstract.Qubit` to be swept (optional).
    """

    parameter: Parameter
    values: npt.NDArray
    pulses: Optional[list] = None
    qubits: Optional[list] = None

    def __post_init__(self):
        if self.pulses is not None and self.qubits is not None:
            raise ValueError("Cannot use a sweeper on both pulses and qubits.")
        elif self.pulses is not None and self.parameter in QubitParameter:
            raise ValueError(f"Cannot sweep {self.parameter} without specifying qubits.")
        elif self.qubits is not None and self.parameter not in QubitParameter:
            raise ValueError(f"Cannot sweep {self.parameter} without specifying pulses.")
        elif self.pulses is None and self.qubits is None:
            raise ValueError("Cannot use a sweeper without specifying pulses or qubits.")
