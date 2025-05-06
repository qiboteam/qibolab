from functools import cached_property
from typing import Literal, Optional, Union

import numpy as np
from pydantic import Field
from qibo.config import raise_error
from scipy.constants import giga

from ...components import Config
from ...identifier import QubitId, QubitPairId, TransitionId
from ...pulses import Delay, Pulse, PulseLike, VirtualZ
from ...serialize import Model
from .engine import Operator, SimulationEngine

__all__ = ["DriveEmulatorConfig", "HamiltonianConfig"]


class DriveEmulatorConfig(Config):
    """Configuration for an IQ channel."""

    kind: Literal["drive-emulator"] = "drive-emulator"

    frequency: float
    """Frequency of drive."""
    rabi_frequency: float = 1e9
    """Rabi frequency [Hz]"""
    scale_factor: float = 1
    """Scaling factor."""

    @staticmethod
    def operator(n: int, engine: SimulationEngine) -> Operator:
        return -1j * (engine.destroy(n) - engine.create(n))


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

    def operator(self, n: int, engine: SimulationEngine) -> Operator:
        """Time independent operator."""
        quadratic_term = engine.create(n) * engine.destroy(n) * self.omega / giga
        quartic_term = (
            self.anharmonicity
            * np.pi
            / giga
            * engine.create(n)
            * engine.create(n)
            * engine.destroy(n)
            * engine.destroy(n)
        )
        return quadratic_term + quartic_term

    def t_phi(self, transition: TransitionId) -> float:
        """T_phi computed from T1 and T2 per transition."""
        return 1 / (1 / self.t2[transition] - 1 / self.t1[transition] / 2)

    def relaxation(self, n: int, engine: SimulationEngine) -> Operator:
        return sum(
            np.sqrt(1 / t1)
            * engine.basis(state=transition[0], dim=n)
            * engine.basis(state=transition[1], dim=n).dag()
            for transition, t1 in self.t1.items()
        )

    def dephasing(self, n: int, engine: SimulationEngine) -> Operator:
        return sum(
            np.sqrt(1 / self.t_phi(pair) / 2)
            * (
                engine.basis(state=pair[0], dim=n)
                * engine.basis(state=pair[0], dim=n).dag()
                - engine.basis(state=pair[1], dim=n)
                * engine.basis(state=pair[1], dim=n).dag()
            )
            for pair in self.t2
        )

    def dissipation(self, n: int, engine: SimulationEngine) -> Operator:
        """Decoherence operator."""
        return self.relaxation(n=n, engine=engine) + self.dephasing(n=n, engine=engine)


class QubitPair(Config):
    """Hamiltonian parameters for qubit pair."""

    coupling: float
    """Qubit-qubit coupling."""

    def operator(self, n: int, engine: SimulationEngine) -> Operator:
        """Time independent operator."""
        op = engine.tensor(
            engine.destroy(n),
            engine.create(n),
        ) + engine.tensor(
            engine.create(n),
            engine.destroy(n),
        )
        return 2 * np.pi * self.coupling / giga * op


class ModulatedDrive(Model):
    """Hamiltonian parameters for qubit drive."""

    pulse: Pulse
    """Drive pulse."""
    config: DriveEmulatorConfig
    """Drive emulator configuration."""
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
    def omega(self):
        return 2 * np.pi * self.config.frequency / giga

    @property
    def rabi_omega(self):
        return 2 * np.pi * self.config.rabi_frequency / giga

    def __call__(self, t, sample, phase):
        i, q = self.envelopes
        phi = self.omega * t + self.pulse.relative_phase + phase
        return (
            self.rabi_omega
            * self.config.scale_factor
            * (np.cos(phi) * i[sample] + np.sin(phi) * q[sample])
        )


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

    def initial_state(self, engine: SimulationEngine) -> Operator:
        """Initial state as ground state of the system."""
        return engine.tensor(
            engine.basis(state=0, dim=self.transmon_levels) for i in range(self.nqubits)
        )

    @property
    def dims(self) -> list[int]:
        """Dimensions of the system."""
        return [self.transmon_levels] * self.nqubits

    def hamiltonian(self, engine: SimulationEngine) -> Operator:
        """Time independent part of Hamiltonian."""
        single_qubit_terms = sum(
            engine.expand(
                qubit.operator(n=self.transmon_levels, engine=engine), self.dims, i
            )
            for i, qubit in self.single_qubit.items()
        )
        two_qubit_terms = sum(
            engine.expand(
                pair.operator(n=self.transmon_levels, engine=engine),
                self.dims,
                list(pair_id),
            )
            for pair_id, pair in self.pairs.items()
        )
        return single_qubit_terms + two_qubit_terms

    def dissipation(self, engine: SimulationEngine) -> Operator:
        """Dissipation operators for the hamiltonian.

        They are going to be passed to mesolve as collapse operators."""
        return sum(
            engine.expand(
                qubit.dissipation(n=self.transmon_levels, engine=engine), self.dims, i
            )
            for i, qubit in self.single_qubit.items()
            if not isinstance(qubit, list)
        )


def waveform(pulse: PulseLike, config: Config) -> Optional[Modulated]:
    """Convert pulse to hamiltonian."""
    if not isinstance(config, DriveEmulatorConfig):
        return None
    if isinstance(pulse, Pulse):
        return ModulatedDrive(pulse=pulse, config=config)
    if isinstance(pulse, Delay):
        return ModulatedDelay(duration=pulse.duration)
    if isinstance(pulse, VirtualZ):
        return ModulatedVirtualZ(phase=pulse.phase)
