from dataclasses import dataclass
from functools import cache
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from qutip import Qobj
from scipy.constants import giga

from qibolab._core.serialize import Model

from ...components import Config, IqConfig
from ...identifier import QubitId, TransitionId
from ...pulses import Delay, Pulse, VirtualZ
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

    @property
    def duration(self):
        """Duration of the pulse."""
        return self.pulse.duration

    @property
    def phase(self):
        """Virtual Z phase."""
        return 0

    def __call__(self, times: NDArray, phase: float) -> NDArray:
        i, q = self.pulse.amplitude * self.pulse.envelope.envelopes(times.size)
        omega = 2 * np.pi * self.frequency * times + self.pulse.relative_phase + phase
        return np.cos(omega) * i + np.sin(omega) * q


@cache
def channel_operator(n: int) -> Qobj:
    """Time independent operator for channel coupling."""
    # TODO: add distinct operators for distinct channel types
    return -1.0j * (transmon_destroy(n) - transmon_create(n))


class ModulatedDelay(Model):
    """Modulated delay."""

    duration: float
    """Delay duration."""
    phase: float = 0
    """Delay has 0 virtual z phase."""

    def __call__(self, times: NDArray, phase: float) -> NDArray:
        return np.zeros_like(times)


class ModulatedVirtualZ(Model):
    """Modulated Virtual Z pulse."""

    phase: float
    """Virtual Z phase."""
    duration: float = 0
    """Duration is 0 for virtual Z."""

    def __call__(self, times: NDArray, phase: float) -> NDArray:
        return np.array([])


Modulated = Union[QubitDrive, ModulatedDelay, ModulatedVirtualZ]


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


def waveform(
    pulse: Union[Pulse, Delay, VirtualZ], channel: Config, level: int
) -> Optional[Modulated]:
    """Convert pulse to hamiltonian."""
    # mapping IqConfig -> QubitDrive
    if not isinstance(channel, IqConfig):
        return None
    if isinstance(pulse, Pulse):
        frequency = channel.frequency
        return QubitDrive(pulse=pulse, frequency=frequency / giga, n=level)
    if isinstance(pulse, Delay):
        return ModulatedDelay(duration=pulse.duration)
    if isinstance(pulse, VirtualZ):
        return ModulatedVirtualZ(phase=pulse.phase)
