from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Optional

import numpy as np
from pydantic import Field

from qibolab import Delay, Pulse
from qibolab._core.components import Config

from ..components import IqConfig
from .operators import L1, L2, QUBIT_CREATE, QUBIT_DESTROY, QUBIT_DRIVE, QUBIT_NUMBER
from .utils import HZ_TO_GHZ


class Qubit(Config):
    """Hamiltonian parameters for single qubit."""

    frequency: float = 0
    """Qubit frequency for 0->1."""
    anharmonicity: float = 0
    """Qubit anharmonicity."""
    t1: float = 0
    """Relaxation time."""
    t2: float = 0
    """Coherence time."""

    @property
    def operator(self):
        """Time independent operator."""
        return (
            2 * np.pi * self.frequency * HZ_TO_GHZ * QUBIT_NUMBER
            + np.pi
            * self.anharmonicity
            * HZ_TO_GHZ
            * QUBIT_CREATE
            * QUBIT_CREATE
            * QUBIT_DESTROY
            * QUBIT_DESTROY
        )

    @property
    def t_phi(self):
        """T_phi computed from T1 and T2."""
        return 1 / (1 / self.t2 - 1 / self.t1 / 2)

    @property
    def decoherence(self):
        """Decoherence operator."""
        assert self.t1 > 0 and self.t2 > 0
        return np.sqrt(1 / self.t1) * L1 + np.sqrt(1 / self.t_phi) * L2


@dataclass
class QubitDrive:
    """Hamiltonian parameters for qubit drive."""

    pulse: Pulse
    """Drive pulse."""
    frequency: float
    """Drive frequency."""
    sampling_rate: float = 1
    """Sampling rate."""

    @cached_property
    def envelopes(self):
        if isinstance(self.pulse, Delay):
            return [np.zeros(len(self)), np.zeros(len(self))]
        return self.pulse.envelopes(self.sampling_rate)

    @cached_property
    def operator(self):
        """Time independent operator."""
        return QUBIT_DRIVE

    def __len__(self):
        return int(self.pulse.duration)

    def __call__(self, t, sample):
        i, q = self.envelopes
        if isinstance(self.pulse, Delay):
            return 0
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


def waveform(pulse, channel, configs, updates=None) -> Optional[QubitDrive]:
    """Convert pulse to hamiltonian."""
    if updates is None:
        updates = {}
    # mapping IqConfig -> QubitDrive
    if isinstance(configs[channel], IqConfig):
        if channel in updates:
            config = configs[channel].model_copy(update=updates[channel])
            frequency = config.frequency
        else:
            frequency = configs[channel].frequency
        if pulse.id in updates:
            pulse = pulse.model_copy(update=updates[pulse.id])
        return QubitDrive(pulse=pulse, frequency=frequency * HZ_TO_GHZ)
