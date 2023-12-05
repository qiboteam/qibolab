from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple, Union

from qibolab.channels import Channel
from qibolab.couplers import Coupler
from qibolab.native import SingleQubitNatives, TwoQubitNatives

QubitId = Union[str, int]
"""Type for qubit names."""

CHANNEL_NAMES = ("readout", "feedback", "drive", "flux", "twpa")
"""Names of channels that belong to a qubit.
Not all channels are required to operate a qubit.
"""
EXCLUDED_FIELDS = CHANNEL_NAMES + ("name", "native_gates")
"""Qubit dataclass fields that are excluded by the ``characterization`` property."""


@dataclass
class Qubit:
    """Representation of a physical qubit.

    Qubit objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.

    Args:
        name (int, str): Qubit number or name.
        readout (:class:`qibolab.platforms.utils.Channel`): Channel used to
            readout pulses to the qubit.
        feedback (:class:`qibolab.platforms.utils.Channel`): Channel used to
            get readout feedback from the qubit.
        drive (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send drive pulses to the qubit.
        flux (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send flux pulses to the qubit.
        Other characterization parameters for the qubit, loaded from the runcard.
    """

    name: QubitId
    kernel_path: Optional[Path] = None

    bare_resonator_frequency: int = 0
    readout_frequency: int = 0
    """ Readout dressed frequency"""
    drive_frequency: int = 0
    anharmonicity: int = 0
    sweetspot: float = 0.0
    flux_to_bias: float = 0.0
    asymmetry: float = 0.0
    bare_resonator_frequency_sweetspot: float = 0.0
    """Bare resonator frequency at sweetspot"""
    ssf_brf: float = 0.0
    """Estimated sweetspot qubit frequency divided by the bare_resonator_frequency"""
    Ec: float = 0.0
    """Readout Charge Energy"""
    Ej: float = 0.0
    """Readout Josephson Energy"""
    g: float = 0.0
    """Readout coupling"""
    assignment_fidelity: float = 0.0
    """Assignment fidelity."""
    readout_fidelity: float = 0.0
    """Readout fidelity."""
    effective_temperature: float = 0.0
    """Effective temperature."""
    peak_voltage: float = 0
    pi_pulse_amplitude: float = 0
    T1: int = 0
    T2: int = 0
    T2_spin_echo: int = 0
    state0_voltage: int = 0
    state1_voltage: int = 0
    mean_gnd_states: List[float] = field(default_factory=lambda: [0, 0])
    mean_exc_states: List[float] = field(default_factory=lambda: [0, 0])

    # parameters for single shot classification
    threshold: Optional[float] = None
    iq_angle: float = 0.0
    # required for mixers (not sure if it should be here)
    mixer_drive_g: float = 0.0
    mixer_drive_phi: float = 0.0
    mixer_readout_g: float = 0.0
    mixer_readout_phi: float = 0.0

    readout: Optional[Channel] = None
    feedback: Optional[Channel] = None
    twpa: Optional[Channel] = None
    drive: Optional[Channel] = None
    flux: Optional[Channel] = None

    native_gates: SingleQubitNatives = field(default_factory=SingleQubitNatives)

    @property
    def channels(self):
        for name in CHANNEL_NAMES:
            channel = getattr(self, name)
            if channel is not None:
                yield channel

    @property
    def characterization(self):
        """Dictionary containing characterization parameters."""
        return {fld.name: getattr(self, fld.name) for fld in fields(self) if fld.name not in EXCLUDED_FIELDS}


QubitPairId = Tuple[QubitId, QubitId]
"""Type for holding ``QubitPair``s in the ``platform.pairs`` dictionary."""


@dataclass
class QubitPair:
    """Data structure for holding the native two-qubit gates acting on a pair of qubits.

    This is needed for symmetry to the single-qubit gates which are storred in the
    :class:`qibolab.platforms.abstract.Qubit`.

    Qubits are sorted according to ``qubit.name`` such that
    ``qubit1.name < qubit2.name``.
    """

    qubit1: Qubit
    qubit2: Qubit

    coupler: Optional[Coupler] = None

    native_gates: TwoQubitNatives = field(default_factory=TwoQubitNatives)
