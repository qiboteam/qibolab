import numpy as np
from .operators import QUBIT_NUMBER, QUBIT_DRIVE
from dataclasses import dataclass, field
from typing import Literal, Callable
from qibolab._core.components import Config
from qibolab import Pulse, Delay


class QubitConfig(Config):
    """Hamiltonian parameters for single qubit."""

    kind: Literal["qubit"] = "qubit"
    frequency: float = 0
    anharmonicity: float = 0

    @property
    def operator(self):
        #TODO: add anharmonicity
        return 2*np.pi*self.frequency*QUBIT_NUMBER


@dataclass
class QubitDrive:
    """Hamiltonian parameters for qubit drive."""
    pulse: Pulse
    frequency: float
    sampling_rate: float = 1


    @property
    def envelopes(self):
        if isinstance(self.pulse, Delay):
            return [np.zeros(len(self)), np.zeros(len(self))]
        return self.pulse.envelopes(self.sampling_rate)

    @property
    def operator(self):
        return QUBIT_DRIVE


    def __len__(self):
        return int(self.pulse.duration)

    def __call__(self, t, sample):
        i, q = self.envelopes
        if isinstance(self.pulse, Delay):
            return i[sample]
        return self.pulse.amplitude*(np.cos(2*np.pi*self.frequency*t + self.pulse.relative_phase)*i[sample] +
                               np.sin(2*np.pi*self.frequency*t+self.pulse.relative_phase)*q[sample])
