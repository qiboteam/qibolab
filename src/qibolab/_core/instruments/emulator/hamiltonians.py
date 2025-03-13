from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Optional

import numpy as np
from pydantic import Field
from scipy.constants import giga

from ...components import Config, IqConfig
from ...pulses import Delay, Pulse
from .operators import L1, L2, QUBIT_DRIVE, SIGMAZ


class Qubit(Config):
    """Hamiltonian parameters for single qubit."""

    frequency: float = 0
    """Qubit frequency for 0->1."""
    t1: float = 0
    """Relaxation time."""
    t2: float = 0
    """Coherence time."""

    @property
    def operator(self):
        """Time independent operator."""
        return -np.pi * (self.frequency / giga) * SIGMAZ

    @property
    def t_phi(self):
        """T_phi computed from T1 and T2."""
        return 1 / (1 / self.t2 - 1 / self.t1 / 2)

    @property
    def decoherence(self):
        """Decoherence operator."""
        assert self.t1 > 0 and self.t2 > 0
        return np.sqrt(1 / self.t1) * L1 + np.sqrt(1 / self.t_phi / 2) * L2


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
        return int(self.pulse.duration * self.sampling_rate)

    def __call__(self, t, sample):
        if isinstance(self.pulse, Delay):
            return 0
        i, q = self.envelopes
        omega = 2 * np.pi * self.frequency * t + self.pulse.relative_phase
        return self.pulse.amplitude * (
            np.cos(omega) * i[sample] + np.sin(omega) * q[sample]
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
        return [
            qubit.decoherence
            for qubit in self.single_qubit.values()
            if not isinstance(qubit, list)
        ]


def waveform(pulse, channel, configs, updates=None) -> Optional[QubitDrive]:
    """Convert pulse to hamiltonian."""
    if updates is None:
        updates = {}
    # mapping IqConfig -> QubitDrive

    if not isinstance(configs[channel], IqConfig):
        return None

    config = configs[channel].model_copy(update=updates.get(channel, {}))
    frequency = config.frequency
    pulse = pulse.model_copy(update=updates.get(pulse.id, {}))
    return QubitDrive(pulse=pulse, frequency=frequency / giga)
