from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy.typing as npt

from qibolab.pulses import PulseType


class Parameter(Enum):
    """Sweeping parameters."""

    attenuation = auto()
    gain = auto()
    frequency = auto()
    amplitude = auto()
    current = auto()


@dataclass
class Sweeper:
    """Data structure for Sweeper object.

    This object is passed as an argument to the method :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`
    which enables the user to sweep a specific parameter for one or more pulses. For information on how to
    perform sweeps see :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`.

    Example:
        .. code-block:: python

            import numpy as np
            from qibolab.platform import Platform
            from qibolab.sweeper import Sweeper
            from qibolab.pulses import PulseSequence


            platform = Platform("dummy")
            sequence = PulseSequence()
            parameter = "frequency"
            pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
            sequence.add(pulse)
            parameter_range = np.random.randint(10, size=10)
            sweeper = Sweeper(parameter, parameter_range, [pulse])
            platform.sweep(sequence, sweeper)

    Args:
        parameter (`qibolab.sweeper.Parameter`): parameter to be swept, possible choices are frequency, attenuation, amplitude, current and gain.
        values (np.ndarray): sweep range. If the parameter is `frequency` the sweep will be a shift around the readout frequency
            in case of a `ReadoutPulse` or around the drive frequency for a generic `Pulse`. For other parameters the sweep will be
            performed directly over the range specified.
        pulses (list) : list of `qibolab.pulses.Pulse` to be swept (optional).
        qubits (lilst): list of `qibolab.platforms.abstract.Qubit` to be swept (optional).
    """

    parameter: Parameter
    values: npt.NDArray
    pulses: Optional[list] = None
    qubits: Optional[list] = None

    @property
    def pulse_type(self) -> Optional[PulseType]:
        types = {p.type for p in self.pulses}
        if len(types) > 1:
            raise RuntimeError("Not homogeneous pulses")
        return next(iter(types), None)
