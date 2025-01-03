from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from qibolab import Delay, Pulse
from qibolab._core.components import Config

from .operators import HZ_TO_GHZ, QUBIT_DESTROY, QUBIT_DRIVE, QUBIT_NUMBER, SIGMAZ


class Qubit(Config):
    """Hamiltonian parameters for single qubit."""

    frequency: float = 0
    anharmonicity: float = 0
    t1: float = 0
    t2: float = 0

    @property
    def operator(self):
        # TODO: add anharmonicity
        return 2 * np.pi * self.frequency * HZ_TO_GHZ * QUBIT_NUMBER

    @property
    def t_phi(self):
        return 1 / (1 / self.t2 - 1 / self.t1 / 2)

    @property
    def decoherence(self):
        assert self.t1 > 0 and self.t2 > 0
        return np.sqrt(1 / self.t1) * QUBIT_DESTROY + np.sqrt(1 / self.t_phi) * SIGMAZ


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
        return self.pulse.amplitude * (
            np.cos(2 * np.pi * self.frequency * t + self.pulse.relative_phase)
            * i[sample]
            + np.sin(2 * np.pi * self.frequency * t + self.pulse.relative_phase)
            * q[sample]
        )


class HamiltonianConfig(Config):
    """Hamiltonian configuration."""

    kind: Literal["hamiltonian"] = "hamiltonian"
    single_qubit: dict[str, Qubit] = Field(default_factory=dict)

    @property
    def hamiltonian(self):
        return [qubit.operator for qubit in self.single_qubit.values()]

    @property
    def decoherence(self):
        ops = []
        for qubit in self.single_qubit.values():
            if isinstance(qubit, list):
                continue
            else:
                ops.append(qubit.decoherence)
        return ops
