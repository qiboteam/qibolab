from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple, Union

import numpy as np

from qibolab.channel_type import Channel
from qibolab.couplers import Coupler
from qibolab.native import SingleQubitNatives, TwoQubitNatives

QubitId = Union[str, int]
"""Type for qubit names."""

CHANNEL_NAMES = ("readout", "acquisition", "drive", "flux", "twpa")
"""Names of channels that belong to a qubit.

Not all channels are required to operate a qubit.
"""
EXCLUDED_FIELDS = CHANNEL_NAMES + ("name", "native_gates", "kernel", "flux")
"""Qubit dataclass fields that are excluded by the ``characterization``
property."""


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

    bare_resonator_frequency: int = 0
    readout_frequency: int = 0
    """Readout dressed frequency."""
    drive_frequency: int = 0
    anharmonicity: int = 0
    sweetspot: float = 0.0
    asymmetry: float = 0.0
    crosstalk_matrix: dict[QubitId, float] = field(default_factory=dict)
    """Crosstalk matrix for voltages."""
    Ec: float = 0.0
    """Readout Charge Energy."""
    Ej: float = 0.0
    """Readout Josephson Energy."""
    g: float = 0.0
    """Readout coupling."""
    assignment_fidelity: float = 0.0
    """Assignment fidelity."""
    readout_fidelity: float = 0.0
    """Readout fidelity."""
    gate_fidelity: float = 0.0
    """Gate fidelity from standard RB."""

    effective_temperature: float = 0.0
    """Effective temperature."""
    peak_voltage: float = 0
    pi_pulse_amplitude: float = 0
    resonator_depletion_time: int = 0
    T1: int = 0
    T2: int = 0
    T2_spin_echo: int = 0
    state0_voltage: int = 0
    state1_voltage: int = 0
    mean_gnd_states: List[float] = field(default_factory=lambda: [0, 0])
    mean_exc_states: List[float] = field(default_factory=lambda: [0, 0])

    # parameters for single shot classification
    threshold: float = 0.0
    iq_angle: float = 0.0
    kernel: Optional[np.ndarray] = field(default=None, repr=False)
    # required for mixers (not sure if it should be here)
    mixer_drive_g: float = 0.0
    mixer_drive_phi: float = 0.0
    mixer_readout_g: float = 0.0
    mixer_readout_phi: float = 0.0

    readout: Optional[Channel] = None
    acquisition: Optional[Channel] = None
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
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.name not in EXCLUDED_FIELDS
        }

    @property
    def mixer_frequencies(self):
        """Get local oscillator and intermediate frequencies of native gates.

        Assumes RF = LO + IF.
        """
        freqs = {}
        for gate in fields(self.native_gates):
            native = getattr(self.native_gates, gate.name)
            if native is not None:
                channel_type = native.pulse_type.name.lower()
                _lo = getattr(self, channel_type).lo_frequency
                _if = native.frequency - _lo
                freqs[gate.name] = _lo, _if
        return freqs


QubitPairId = Tuple[QubitId, QubitId]
"""Type for holding ``QubitPair``s in the ``platform.pairs`` dictionary."""


@dataclass
class QubitPair:
    """Data structure for holding the native two-qubit gates acting on a pair
    of qubits.

    This is needed for symmetry to the single-qubit gates which are storred in the
    :class:`qibolab.platforms.abstract.Qubit`.
    """

    qubit1: Qubit
    """First qubit of the pair.

    Acts as control on two-qubit gates.
    """
    qubit2: Qubit
    """Second qubit of the pair.

    Acts as target on two-qubit gates.
    """

    gate_fidelity: float = 0.0
    """Gate fidelity from standard 2q RB."""

    cz_fidelity: float = 0.0
    """Gate fidelity from CZ interleaved RB."""

    coupler: Optional[Coupler] = None

    native_gates: TwoQubitNatives = field(default_factory=TwoQubitNatives)

    @property
    def characterization(self):
        """Dictionary containing characterization parameters."""
        return {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.name not in EXCLUDED_FIELDS
        }
