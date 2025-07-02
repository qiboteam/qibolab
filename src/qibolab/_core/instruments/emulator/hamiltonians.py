from functools import cached_property
from typing import Literal, Optional, Union

import numpy as np
from pydantic import Field
from qibo.config import raise_error
from scipy.constants import giga

from ...components import Config
from ...identifier import QubitId, QubitPairId, TransitionId
from ...parameters import Update, _setvalue
from ...pulses import Delay, Pulse, PulseLike, VirtualZ
from ...serialize import Model
from .engine import Operator, SimulationEngine

__all__ = ["DriveEmulatorConfig", "FluxEmulatorConfig", "HamiltonianConfig"]


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


class FluxEmulatorConfig(Config):
    """Configuration for a flux line."""

    kind: Literal["flux-emulator"] = "flux-emulator"

    offset: float
    """DC offset of the channel."""
    voltage_to_flux: float = 1
    """Convert voltarget to flux."""

    @staticmethod
    def operator(n: int, engine: SimulationEngine) -> Operator:
        return engine.create(n) * engine.destroy(n)


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


class CapacitiveCoupling(Config):
    """Hamiltonian parameters for qubit pair."""

    coupling: float
    """Qubit-qubit coupling."""

    @staticmethod
    def _operator(n: int, engine: SimulationEngine) -> Operator:
        """Time independent operator."""
        op = engine.tensor(
            [
                engine.destroy(n),
                engine.create(n),
            ]
        ) + engine.tensor(
            [
                engine.create(n),
                engine.destroy(n),
            ]
        )
        return 2 * np.pi * op / giga

    def operator(self, n: int, engine: SimulationEngine) -> Operator:
        """Time independent operator."""
        return self.coupling * self._operator(n, engine)


class FluxPulse(Model):
    """Flux pulse term in Hamiltonian."""

    pulse: Pulse
    """Flux pulse to be played."""
    config: FluxEmulatorConfig
    """Flux emulator configuration."""
    qubit: Qubit
    """Qubit affected by the flux pulse."""
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
                self.qubit.detuned_frequency(
                    self.config.voltage_to_flux * (i[sample] + self.config.offset)
                )
                - self.qubit.dynamical_frequency
            )
            / giga
        )


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
ControlLine = Union[Modulated, FluxPulse]


class HamiltonianConfig(Config):
    """Hamiltonian configuration."""

    kind: Literal["hamiltonian"] = "hamiltonian"
    transmon_levels: int = 2
    qubits: dict[QubitId, Qubit] = Field(default_factory=dict)
    pairs: dict[QubitPairId, CapacitiveCoupling] = Field(default_factory=dict)

    @property
    def nqubits(self):
        """Number of qubits."""
        return len(self.qubits)

    def replace(self, update: Update) -> "HamiltonianConfig":
        """Update parameters' values."""
        d = self.model_dump()
        for path, val in update.items():
            _setvalue(d, path, val)
        return self.model_validate(d)

    def update_from_configs(
        self, config: dict[str, Config]
    ) -> tuple["HamiltonianConfig", dict[str, Config]]:
        """Update hamiltonian parameters from configs.
        Also the configs itself are updated if they contain an HamiltonianConfig."""

        config_update = {}
        for qubit in self.qubits:
            flux = config.get(f"{qubit}/flux")
            config_update.update(
                {
                    f"qubits.{qubit}.dynamical_frequency": self.qubits[
                        qubit
                    ].detuned_frequency(
                        flux.offset * flux.voltage_to_flux if flux is not None else 0
                    )
                }
            )

        new_hamiltonian_config = self.replace(update=config_update)
        if "hamiltonian" in config:
            config["hamiltonian"] = new_hamiltonian_config
        return new_hamiltonian_config, config

    def initial_state(self, engine: SimulationEngine) -> Operator:
        """Initial state as ground state of the system."""
        return engine.basis(self.dims, self.nqubits * [0])

    def hilbert_space_index(self, qubit: QubitId) -> int:
        """Return Hilbert space index from qubit id."""
        return list(self.qubits).index(qubit)

    @property
    def dims(self) -> list[int]:
        """Dimensions of the system."""
        return [self.transmon_levels] * len(self.qubits)

    def hamiltonian(self, engine: SimulationEngine) -> Operator:
        """Time independent part of Hamiltonian."""
        qubit_terms = sum(
            engine.expand(
                qubit.operator(self.transmon_levels, engine),
                self.dims,
                self.hilbert_space_index(i),
            )
            for i, qubit in self.qubits.items()
        )
        coupling = sum(
            engine.expand(
                pair.operator(self.transmon_levels, engine),
                self.dims,
                [
                    self.hilbert_space_index(pair_id[0]),
                    self.hilbert_space_index(pair_id[1]),
                ],
            )
            for (pair_id, pair) in self.pairs.items()
        )
        return qubit_terms + coupling

    def dissipation(self, engine: SimulationEngine) -> Operator:
        """Dissipation operators for the hamiltonian.

        They are going to be passed to mesolve as collapse operators."""
        collapse_operators = []
        for i, qubit in self.qubits.items():
            if len(qubit.t1) > 0:
                collapse_operators.append(
                    engine.expand(
                        qubit.relaxation(self.transmon_levels, engine),
                        self.dims,
                        self.hilbert_space_index(i),
                    )
                )
            if len(qubit.t2) > 0:
                collapse_operators.append(
                    engine.expand(
                        qubit.dephasing(self.transmon_levels, engine),
                        self.dims,
                        self.hilbert_space_index(i),
                    )
                )
        return collapse_operators


def waveform(
    pulse: PulseLike,
    config: Config,
    qubit: Qubit,
) -> Optional[ControlLine]:
    """Convert pulse to hamiltonian."""
    if not isinstance(config, (DriveEmulatorConfig, FluxEmulatorConfig)):
        return None

    if isinstance(pulse, Pulse):
        if isinstance(config, DriveEmulatorConfig):
            return ModulatedDrive(pulse=pulse, config=config)
        if isinstance(config, FluxEmulatorConfig):
            return FluxPulse(
                pulse=pulse,
                config=config,
                qubit=qubit,
            )
    if isinstance(pulse, Delay):
        return ModulatedDelay(duration=pulse.duration)
    if isinstance(pulse, VirtualZ):
        return ModulatedVirtualZ(phase=pulse.phase)
