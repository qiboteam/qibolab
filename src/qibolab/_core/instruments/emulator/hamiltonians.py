from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Optional

import numpy as np
from pydantic import Field
from scipy.constants import giga

from ...components import Config, IqConfig
from ...identifier import QubitId, TransitionId
from ...pulses import Delay, Pulse
from .operators import (
    dephasing,
    probability,
    relaxation,
    state,
    transmon_create,
    transmon_destroy,
)


class Qubit(Config):
    """Hamiltonian parameters for single qubit."""

    frequency: float = 0
    """Qubit frequency for 0->1."""
    anharmonicity: float = 0
    """Qubit anharmonicity."""
    t1: dict[TransitionId, float] = Field(default_factory=dict)
    """Dictionary with relaxation times per transition."""
    t2: dict[TransitionId, float] = Field(default_factory=dict)
    """Dictionary with dephasing time per transition."""

    @property
    def omega(self) -> float:
        """Angular velocity."""
        return 2 * np.pi * self.frequency

    def operator(self, n: int):
        """Time independent operator."""
        quadratic_term = transmon_create(n) * transmon_destroy(n) * self.omega / giga
        quartic_term = (
            self.anharmonicity
            * np.pi
            / giga
            * transmon_create(n)
            * transmon_create(n)
            * transmon_destroy(n)
            * transmon_destroy(n)
        )
        return quadratic_term + quartic_term

    def t_phi(self, transition: TransitionId) -> float:
        """T_phi computed from T1 and T2 per transition."""
        return 1 / (1 / self.t2[transition] - 1 / self.t1[transition] / 2)

    def relaxation(self, n: int):
        return sum(
            np.sqrt(1 / t1) * relaxation(pair[0], pair[1], n)
            for pair, t1 in self.t1.items()
        )

    def dephasing(self, n: int):
        return sum(
            np.sqrt(1 / self.t_phi(pair) / 2) * dephasing(pair[0], pair[1], n)
            for pair in self.t2
        )

    def dissipation(self, n: int):
        """Decoherence operator."""
        return self.relaxation(n) + self.dephasing(n)


@dataclass
class QubitDrive:
    """Hamiltonian parameters for qubit drive."""

    pulse: Pulse
    """Drive pulse."""
    frequency: float
    """Drive frequency."""
    n: int
    """Transmon levels."""
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
        return -1.0j * (transmon_destroy(self.n) - transmon_create(self.n))

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
    transmon_levels: int = 2
    single_qubit: dict[QubitId, Qubit] = Field(default_factory=dict)

    @property
    def initial_state(self):
        return state(0, self.transmon_levels)

    def probability(self, state: int):
        return probability(state=state, n=self.transmon_levels)

    @property
    def hamiltonian(self):
        return [
            qubit.operator(self.transmon_levels) for qubit in self.single_qubit.values()
        ]

    @property
    def dissipation(self):
        return [
            qubit.dissipation(self.transmon_levels)
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
    return QubitDrive(
        pulse=pulse,
        frequency=frequency / giga,
        n=configs["hamiltonian"].transmon_levels,
    )
