from functools import cached_property, reduce
from typing import Literal, Optional, Union
from operator import add

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
    def operator(engine: SimulationEngine, **kwargs) -> Operator:
        return -1j * (engine.destroy(**kwargs) - engine.create(**kwargs))    


class FluxEmulatorConfig(Config):
    """Configuration for a flux line."""

    kind: Literal["flux-emulator"] = "flux-emulator"

    offset: float
    """DC offset of the channel."""
    voltage_to_flux: float = 1
    """Convert voltarget to flux."""

    @staticmethod
    def operator(engine: SimulationEngine, **kwargs) -> Operator:
        return engine.create(**kwargs) * engine.destroy(**kwargs)

    @property
    def flux(self) -> float:
        """Returns flux."""
        return self.offset * self.voltage_to_flux


class Qubit(Config):
    """Hamiltonian parameters for single qubit."""

    frequency: float = 0
    """Qubit frequency for 0->1."""
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

    def omega(self, flux: float = 0) -> float:
        """Angular velocity."""
        return 2 * np.pi * self.detuned_frequency(flux)

    def detuned_frequency(self, flux: float) -> float:
        """Return frequency of the qubit modified by the flux."""
        return (self.frequency - self.anharmonicity) * (
            self.asymmetry**2
            + (1 - self.asymmetry**2) * np.cos(np.pi * (flux - self.sweetspot)) ** 2
        ) ** (1 / 4) + self.anharmonicity

    def operator(self, engine: SimulationEngine, flux: float = 0, **kwargs):
        """Time independent operator."""
        quadratic_term = engine.create(**kwargs) * engine.destroy(**kwargs) * self.omega(flux) / giga
        quartic_term = (
            self.anharmonicity
            * np.pi
            / giga
            * engine.create(**kwargs)
            * engine.create(**kwargs)
            * engine.destroy(**kwargs)
            * engine.destroy(**kwargs)
        )
        return quadratic_term + quartic_term

    def t_phi(self, transition: TransitionId) -> float:
        """T_phi computed from T1 and T2 per transition."""
        return 1 / (1 / self.t2[transition] - 1 / self.t1[transition] / 2)

    def relaxation(self, engine: SimulationEngine, **kwargs) -> Operator:
        return reduce(
            add,
            (
                np.sqrt(1 / t1) * engine.relaxation_op(transition=transition, **kwargs)
                for transition, t1 in self.t1.items()
            )
        )

    def dephasing(self, engine: SimulationEngine, **kwargs) -> Operator:
        return reduce(
            add,
            (
                np.sqrt(1 / self.t_phi(pair) / 2) * engine.dephasing_op(pair=pair, **kwargs)
                for pair in self.t2
            )
        )


class CapacitiveCoupling(Config):
    """Hamiltonian parameters for qubit pair."""

    coupling: float
    """Qubit-qubit coupling."""

    @staticmethod
    def _operator(n: int, target1: int, target2: int, engine: SimulationEngine) -> Operator:
        """Time independent operator."""
        op = engine.tensor(
            [
                engine.destroy(n=n, target=target1),
                engine.create(n=n, target=target2),
            ]
        ) + engine.tensor(
            [
                engine.create(n=n, target=target1),
                engine.destroy(n=n, target=target2),
            ]
        )
        return 2 * np.pi * op / giga

    def operator(self, n: int, target1: int, target2: int, engine: SimulationEngine) -> Operator:
        """Time independent operator."""
        return self.coupling * self._operator(n, target1, target2, engine)


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
                - self.qubit.detuned_frequency(self.config.flux)
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

    def initial_state(self, engine: SimulationEngine) -> Operator:
        """Initial state as ground state of the system."""
        return engine.basis(self.dims, self.nqubits * [0])

    def hilbert_space_index(self, qubit: QubitId, engine_has_flipped_index: bool = False) -> int:
        """Return Hilbert space index from qubit id."""
        index = list(self.qubits).index(qubit)
        if engine_has_flipped_index:
            index = self.nqubits - 1 - index
        return index

    def hilbert_space_dims(self, engine_has_flipped_index: bool = False) -> dict:
        """Construct dictionary of hilbert space index and its corresponding dimension of the system."""
        dims = {}
        for qubit_id, _ in self.qubits.items():
            dims |= {self.hilbert_space_index(qubit_id, engine_has_flipped_index): self.transmon_levels}
        return dims
    
    @property
    def dims(self) -> list[int]:
        """Dimensions of the system."""
        return [self.transmon_levels] * len(self.qubits)

    def hamiltonian(self, config: dict, engine: SimulationEngine) -> Operator:
        """Time independent part of Hamiltonian."""
        qubit_terms = reduce(
            add,
            (
                engine.expand(
                    qubit.operator(
                        n=self.transmon_levels,
                        target=self.hilbert_space_index(i, engine.has_flipped_index),
                        flux=static_flux(qubit=i, config=config),
                        engine=engine,
                    ),
                    self.dims,
                    self.hilbert_space_index(i, engine.has_flipped_index),
                )
                for i, qubit in self.qubits.items()
            )
        )
        coupling = reduce(
            add,
            (
                engine.expand(
                    pair.operator(
                        n=self.transmon_levels, 
                        target1=self.hilbert_space_index(pair_id[0], engine.has_flipped_index),
                        target2=self.hilbert_space_index(pair_id[1], engine.has_flipped_index),
                        engine=engine
                    ),
                    self.dims,
                    [
                        self.hilbert_space_index(pair_id[0], engine.has_flipped_index),
                        self.hilbert_space_index(pair_id[1], engine.has_flipped_index),
                    ],
                )
                for (pair_id, pair) in self.pairs.items()
            )
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
                        qubit.relaxation(dim=self.transmon_levels, target=self.hilbert_space_index(i, engine.has_flipped_index), engine=engine),
                        self.dims,
                        self.hilbert_space_index(i, engine.has_flipped_index),
                    )
                )
            if len(qubit.t2) > 0:
                collapse_operators.append(
                    engine.expand(
                        qubit.dephasing(dim=self.transmon_levels, target=self.hilbert_space_index(i, engine.has_flipped_index), engine=engine),
                        self.dims,
                        self.hilbert_space_index(i, engine.has_flipped_index),
                    )
                )
        return collapse_operators


def static_flux(qubit: QubitId, config: dict) -> float:
    """Get static flux for qubit given config (offset)."""
    qubit_config = config.get(f"{qubit}/flux")
    if qubit_config is not None:
        return qubit_config.flux
    coupler_config = config.get(f"coupler_{qubit}/flux")
    if coupler_config is not None:
        return coupler_config.flux
    return 0


def waveform(
    pulse: PulseLike,
    config: Config,
    qubit: Qubit,
    sampling_rate: float,
) -> Optional[ControlLine]:
    """Convert pulse to hamiltonian."""
    if not isinstance(config, (DriveEmulatorConfig, FluxEmulatorConfig)):
        return None

    if isinstance(pulse, Pulse):
        if isinstance(config, DriveEmulatorConfig):
            return ModulatedDrive(
                pulse=pulse, config=config, sampling_rate=sampling_rate
            )
        if isinstance(config, FluxEmulatorConfig):
            return FluxPulse(
                pulse=pulse,
                config=config,
                qubit=qubit,
                sampling_rate=sampling_rate,
            )
    if isinstance(pulse, Delay):
        return ModulatedDelay(duration=pulse.duration)
    if isinstance(pulse, VirtualZ):
        return ModulatedVirtualZ(phase=pulse.phase)
