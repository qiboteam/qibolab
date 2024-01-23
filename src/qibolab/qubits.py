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
EXCLUDED_FIELDS = CHANNEL_NAMES + ("name", "native_gates", "_flux")
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
    kernel_path: Optional[Path] = None
    # required for mixers (not sure if it should be here)
    mixer_drive_g: float = 0.0
    mixer_drive_phi: float = 0.0
    mixer_readout_g: float = 0.0
    mixer_readout_phi: float = 0.0

    readout: Optional[Channel] = None
    feedback: Optional[Channel] = None
    twpa: Optional[Channel] = None
    drive: Optional[Channel] = None
    _flux: Optional[Channel] = None

    native_gates: SingleQubitNatives = field(default_factory=SingleQubitNatives)

    def __post_init__(self):
        if self.flux is not None and self.sweetspot != 0:
            self.flux.offset = self.sweetspot

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, channel):
        if self.sweetspot != 0:
            channel.offset = self.sweetspot
        self._flux = channel

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
    def mz_frequencies(self):
        """Get local oscillator and intermediate frequency used for readout.

        Assumes RF = LO + IF.
        """
        _lo = self.readout.lo_frequency
        _if = self.native_gates.MZ.frequency - _lo
        return _lo, _if

    @property
    def rx_frequencies(self):
        """Get local oscillator and intermediate frequency used for drive.

        Assumes RF = LO + IF.
        """
        _lo = self.drive.lo_frequency
        _if = self.native_gates.RX.frequency - _lo
        return _lo, _if


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

    coupler: Optional[Coupler] = None

    native_gates: TwoQubitNatives = field(default_factory=TwoQubitNatives)
