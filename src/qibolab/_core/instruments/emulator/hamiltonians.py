from dataclasses import dataclass
from functools import cache, cached_property
from itertools import product
from typing import Literal, Optional, Union

import numpy as np
from pydantic import Field
from qibo.config import raise_error
from qutip import Qobj, qeye, tensor
from scipy.constants import giga

from ...components import Config
from ...identifier import QubitId, QubitPairId, TransitionId
from ...pulses import Delay, Pulse, PulseLike, VirtualZ
from ...serialize import Model
from .operators import (
    dephasing,
    probability,
    relaxation,
    state,
    transmon_create,
    transmon_destroy,
)


class DriveConfig(Config):
    """Configuration for an IQ channel."""

    kind: Literal["drive"] = "drive"

    frequency: float
    """Frequency of drive."""
    rabi_frequency: float = 1
    """Rabi frequency [GHz]"""
    scale_factor: float = 10
    """Scaling factor."""


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


class QubitPair(Config):
    """Hamiltonian parameters for qubit pair."""

    coupling: float
    """Qubit-qubit coupling."""

    def operator(self, n: int):
        """Time independent operator."""
        # TODO: pass index of qubits to assign position in the tensor product
        return (
            2
            * np.pi
            * self.coupling
            / giga
            * tensor(channel_operator(n), channel_operator(n))
        )


@dataclass
class ModulatedDrive:
    """Hamiltonian parameters for qubit drive."""

    pulse: Pulse
    """Drive pulse."""
    frequency: float
    """Drive frequency."""
    rabi_frequency: float
    """Rabi frequency."""
    scale_factor: float
    """Scaling factor."""
    n: int
    """Transmon levels."""
    phase: float = 0
    """Drive has zero virtual z phase."""
    sampling_rate: float = 1
    """Sampling rate."""

    @cached_property
    def envelopes(self):
        """Pulse envelopes."""
        return self.pulse.envelopes(self.sampling_rate)

    @property
    def duration(self):
        """Duration of the pulse."""
        return self.pulse.duration

    @property
    def phase(self):
        """Virtual Z phase."""
        return 0

    @property
    def omega(self):
        return 2 * np.pi * self.rabi_frequency

    def __call__(self, t, sample, phase):
        i, q = self.envelopes
        phi = 2 * np.pi * self.frequency * t + self.pulse.relative_phase + phase
        return (
            self.omega
            * self.scale_factor
            * (np.cos(phi) * i[sample] + np.sin(phi) * q[sample])
        )


@cache
def channel_operator(n: int) -> Qobj:
    """Time independent operator for channel coupling."""
    # TODO: add distinct operators for distinct channel types
    return -1j * (transmon_destroy(n) - transmon_create(n))


class ModulatedDelay(Model):
    """Modulated delay."""

    duration: float
    """Delay duration."""
    phase: float = 0
    """Delay has 0 virtual z phase."""

    def __call__(self, t: float, sample: int, phase: float) -> float:
        """Delay waveform."""
        return 0


class ModulatedVirtualZ(Model):
    """Modulated Virtual Z pulse."""

    phase: float
    """Virtual Z phase."""
    duration: float = 0
    """Duration is 0 for virtual Z."""

    def __call__(self, t: float, sample: int, phase: float) -> float:
        """Delay waveform."""
        raise_error(ValueError, "VirtualZ doesn't have waveform.")


Modulated = Union[ModulatedDrive, ModulatedDelay, ModulatedVirtualZ]


class HamiltonianConfig(Config):
    """Hamiltonian configuration."""

    kind: Literal["hamiltonian"] = "hamiltonian"
    transmon_levels: int = 2
    single_qubit: dict[QubitId, Qubit] = Field(default_factory=dict)
    pairs: dict[QubitPairId, QubitPair] = Field(default_factory=dict)

    @property
    def nqubits(self):
        return len(self.single_qubit)

    @property
    def identity(self):
        return self.nqubits * [qeye(self.transmon_levels)]

    def _embed_operator(self, operator: Qobj, index: int) -> Qobj:
        """Embed operator in the tensor product space."""
        space = self.identity
        space[index] = operator
        return tensor(space)

    @property
    def initial_state(self):
        return tensor(state(0, self.transmon_levels) for i in range(self.nqubits))

    @property
    def outcomes(self) -> list[str]:
        """Compute all possible outcomes."""
        if self.nqubits > 1:
            return [
                f"{i}{j}"
                for i, j in product(
                    list(range(self.transmon_levels)), repeat=self.nqubits
                )
            ]
        return [f"{i}" for i in range(self.transmon_levels)]

    def probability(self, state: int, index: int) -> Qobj:
        """Probability of having qubit at `index` with state `state`."""
        return self._embed_operator(
            probability(state=int(state), n=self.transmon_levels), index
        )

    @property
    def observable(self) -> list[Qobj]:
        operators = []
        for i in range(self.nqubits):
            for j in range(self.transmon_levels):
                operators.append(self.probability(j, i))
        return operators

    @property
    def hamiltonian(self):
        ham = sum(
            [
                self._embed_operator(qubit.operator(self.transmon_levels), i)
                for i, qubit in self.single_qubit.items()
            ]
        )
        ham += sum(pair.operator(self.transmon_levels) for pair in self.pairs.values())
        return ham

    @property
    def dissipation(self):
        return sum(
            [
                self._embed_operator(qubit.dissipation(self.transmon_levels), i)
                for i, qubit in self.single_qubit.items()
                if not isinstance(qubit, list)
            ]
        )


def waveform(pulse: PulseLike, config: Config, level: int) -> Optional[Modulated]:
    """Convert pulse to hamiltonian."""
    # mapping IqConfig -> QubitDrive
    if not isinstance(config, DriveConfig):
        return None
    if isinstance(pulse, Pulse):
        return ModulatedDrive(
            pulse=pulse,
            frequency=config.frequency / giga,
            rabi_frequency=config.rabi_frequency / giga,
            scale_factor=config.scale_factor,
            n=level,
        )
    if isinstance(pulse, Delay):
        return ModulatedDelay(duration=pulse.duration)
    if isinstance(pulse, VirtualZ):
        return ModulatedVirtualZ(phase=pulse.phase)
