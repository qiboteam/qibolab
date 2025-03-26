from functools import cached_property
from typing import Literal, Optional, Union

import numpy as np
from pydantic import Field
from qibo.config import raise_error
from scipy.constants import giga

from ...components import Config
from ...identifier import QubitId, QubitPairId, TransitionId
<<<<<<< HEAD
from ...pulses import Delay, Pulse, PulseLike, VirtualZ
from ...serialize import Model
=======
from ...parameters import Update, _setvalue
from ...pulses import Delay, Pulse, VirtualZ
>>>>>>> b30529ae (feat: Dummy prototype for offset sweeper)
from .operators import (
    Operator,
    dephasing,
    expand,
    relaxation,
    state,
    tensor_product,
    transmon_create,
    transmon_destroy,
)

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
    def operator(n: int) -> Operator:
        return -1j * (transmon_destroy(n) - transmon_create(n))


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
    dynamical_frequency: float = 0
    """Frequency to be used during evolution (could be different from frequency due to static offset.)"""
    anharmonicity: float = 0
    """Qubit anharmonicity."""
    sweetspot: float = 0
    """Sweetspot point."""
    asymmetry: float = 0
    """Asymmetry."""
    t1: dict[TransitionId, float] = Field(default_factory=dict)
    """Dictionary with relaxation times per transition."""
    t2: dict[TransitionId, float] = Field(default_factory=dict)
    """Dictionary with dephasing time per transition."""

    @property
    def omega(self) -> float:
        """Angular velocity."""
        return 2 * np.pi * self.dynamical_frequency

    def detuned_frequency(self, flux: float) -> float:
        """Return frequency of the qubit modified by the flux."""
        return (self.frequency - self.anharmonicity) * (
            self.asymmetry**2
            + (1 - self.asymmetry**2) * np.cos(np.pi * (flux - self.sweetspot)) ** 2
        ) ** (1 / 4) + self.anharmonicity

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

    def operator(self, n: int) -> Operator:
        """Time independent operator."""
        op = tensor_product(
            transmon_destroy(n),
            transmon_create(n),
        ) + tensor_product(
            transmon_create(n),
            transmon_destroy(n),
        )
        return 2 * np.pi * self.coupling / giga * op


@dataclass
class FluxPulse:
    pulse: Pulse
    """Flux pulse to be played."""
    offset: float
    """Static bias offset."""
    flux_freq_dependence: callable
    """Flux frequency dep."""
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

    def __call__(self, t, sample, phase):
        i, _ = self.envelopes
        # we are passing the relative frequency because the term with the offset
        # is already included in the time-independent part of the Hamiltonian
        # and it corresponds to changing the static bias
        return (
            2
            * np.pi
            * (
                self.flux_freq_dependence(i[sample] + self.offset)
                - self.flux_freq_dependence(self.offset)
            )
            / giga
        )


@dataclass
class QubitDrive:
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

    @property
    def omega(self):
        return 2 * np.pi * self.rabi_frequency

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

    def replace(self, update: Update) -> "HamiltonianConfig":
        """Update parameters' values."""
        d = self.model_dump()
        for path, val in update.items():
            _setvalue(d, path, val)

        return self.model_validate(d)

    @property
    def nqubits(self):
        return len(self.single_qubit)

    @property
    def initial_state(self):
        """Initial state as ground state of the system."""
        return tensor_product(
            state(0, self.transmon_levels) for i in range(self.nqubits)
        )

    @property
    def dims(self) -> list[int]:
        """Dimensions of the system."""
        return [self.transmon_levels] * self.nqubits

    @property
    def hamiltonian(self) -> Operator:
        """Time independent part of Hamiltonian."""
        single_qubit_terms = sum(
            expand(qubit.operator(self.transmon_levels), self.dims, i)
            for i, qubit in self.single_qubit.items()
        )
        two_qubit_terms = sum(
            expand(pair.operator(self.transmon_levels), self.dims, list(pair_id))
            for pair_id, pair in self.pairs.items()
        )
        return single_qubit_terms + two_qubit_terms

    @property
    def dissipation(self) -> Operator:
        """Dissipation operators for the hamiltonian.

        They are going to be passed to mesolve as collapse operators."""
        return sum(
            expand(qubit.dissipation(self.transmon_levels), self.dims, i)
            for i, qubit in self.single_qubit.items()
            if not isinstance(qubit, list)
        )


def waveform(
    pulse: Union[Pulse, Delay], channel: Config, level: int, flux_dependence: callable
) -> Optional[Modulated]:
    """Convert pulse to hamiltonian."""
    if not isinstance(config, DriveEmulatorConfig):
        return None

    if isinstance(pulse, Pulse):
        if isinstance(channel, DriveConfig):
            return QubitDrive(
                pulse=pulse,
                frequency=channel.frequency / giga,
                rabi_frequency=channel.rabi_frequency / giga,
                scale_factor=channel.scale_factor,
                n=level,
            )
        if isinstance(channel, DcConfig):
            return FluxPulse(
                pulse=pulse,
                offset=channel.offset,
                flux_freq_dependence=flux_dependence,
            )
    if isinstance(pulse, Delay):
        return ModulatedDelay(duration=pulse.duration)
    if isinstance(pulse, VirtualZ):
        return ModulatedVirtualZ(phase=pulse.phase)
