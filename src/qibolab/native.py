from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

from .pulses import Drag, Gaussian, PulseSequence
from .serialize_ import replace


def _normalize_angles(theta, phi):
    """Normalize theta to (-pi, pi], and phi to [0, 2*pi)."""
    theta = theta % (2 * np.pi)
    theta = theta - 2 * np.pi * (theta > np.pi)
    phi = phi % (2 * np.pi)
    return theta, phi


class RxyFactory:
    """Factory for pulse sequences that generate single-qubit rotations around
    an axis in xy plane.

    It is assumed that the underlying sequence contains only a single pulse.
    It is assumed that the base sequence corresponds to a calibrated pi rotation around X axis.
    Other rotation angles are achieved by scaling the amplitude, assuming a linear transfer function.

    Args:
        sequence: The base sequence for the factory.
    """

    def __init__(self, sequence: PulseSequence):
        if len(sequence) != 1:
            raise ValueError(
                f"Incompatible number of channels: {len(sequence)}. "
                f"{self.__class__} expects a sequence on exactly one channel."
            )

        pulses = next(iter(sequence.values()))
        if len(pulses) != 1:
            raise ValueError(
                f"Incompatible number of pulses: {len(pulses)}. "
                f"{self.__class__} expects a sequence with exactly one pulse."
            )

        pulse = pulses[0]
        expected_envelopes = (Gaussian, Drag)
        if not isinstance(pulse.envelope, expected_envelopes):
            raise ValueError(
                f"Incompatible pulse envelope: {pulse.envelope.__class__}. "
                f"{self.__class__} expects {expected_envelopes} envelope."
            )

        self._seq = sequence

    def create_sequence(self, theta: float, phi: float) -> PulseSequence:
        """Create a sequence for single-qubit rotation.

        Args:
            theta: the angle of rotation.
            phi: the angle that rotation axis forms with x axis.
        """
        theta, phi = _normalize_angles(theta, phi)
        seq = self._seq.copy()
        channel = next(iter(seq.keys()))
        pulse = seq[channel][0]
        new_amplitude = pulse.amplitude * theta / np.pi
        seq[channel][0] = replace(pulse, amplitude=new_amplitude, relative_phase=phi)
        return seq


class FixedSequenceFactory:
    """Simple factory for a fixed arbitrary sequence."""

    def __init__(self, sequence: PulseSequence):
        self._seq = sequence

    def create_sequence(self) -> PulseSequence:
        return self._seq.copy()


@dataclass
class SingleQubitNatives:
    """Container with the native single-qubit gates acting on a specific
    qubit."""

    RX: Optional[RxyFactory] = None
    """Pulse to drive the qubit from state 0 to state 1."""
    RX12: Optional[FixedSequenceFactory] = None
    """Pulse to drive to qubit from state 1 to state 2."""
    MZ: Optional[FixedSequenceFactory] = None
    """Measurement pulse."""


@dataclass
class TwoQubitNatives:
    """Container with the native two-qubit gates acting on a specific pair of
    qubits."""

    CZ: Optional[FixedSequenceFactory] = field(
        default=None, metadata={"symmetric": True}
    )
    CNOT: Optional[FixedSequenceFactory] = field(
        default=None, metadata={"symmetric": False}
    )
    iSWAP: Optional[FixedSequenceFactory] = field(
        default=None, metadata={"symmetric": True}
    )

    @property
    def symmetric(self):
        """Check if the defined two-qubit gates are symmetric between target
        and control qubits."""
        return all(
            fld.metadata["symmetric"] or getattr(self, fld.name) is None
            for fld in fields(self)
        )
