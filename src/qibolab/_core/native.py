from typing import Annotated, Optional

import numpy as np

from .pulses import Pulse
from .sequence import PulseSequence
from .serialize import Model, replace


class Native(PulseSequence):
    def create_sequence(self) -> PulseSequence:
        """Create the sequence associated to the gate."""
        return PulseSequence([(ch, p.new()) for ch, p in self])

    def __call__(self, *args, **kwargs) -> PulseSequence:
        """Create the sequence associated to the gate.

        Alias to :meth:`create_sequence`.
        """
        return self.create_sequence(*args, **kwargs)


def _normalize_angles(theta, phi):
    """Normalize theta to (-pi, pi], and phi to [0, 2*pi)."""
    assert 0 <= theta < 2 * np.pi, f"Theta {theta} must be in [0, 2*pi)"
    theta = theta % (2 * np.pi)
    theta = theta - 2 * np.pi * (theta > np.pi)
    phi = phi % (2 * np.pi)
    return theta, phi


def rotation(
    seq: PulseSequence,
    theta: float = np.pi,
    phi: float = 0.0,
    rx90: bool = False,
) -> PulseSequence:
    """Create a sequence for single-qubit rotation.

    ``theta`` will be the angle of the rotation, while ``phi`` the angle that the rotation axis forms with x axis.
    If ``rx90`` is True the rotation will be performed starting from an RX90 pulse doubling its amplitude,
    if the amplitude is greater than 0.5 two pulses are played.
    """
    theta, phi = _normalize_angles(theta, phi)
    ch, pulse = seq[0]
    assert isinstance(pulse, Pulse)
    amplitude = pulse.amplitude * theta / np.pi
    if rx90:
        if amplitude > 0.5:
            p = replace(pulse, amplitude=amplitude, relative_phase=phi)
            return PulseSequence([(ch, p), (ch, p.model_copy())])
        else:
            return PulseSequence(
                [(ch, replace(pulse, amplitude=2 * amplitude, relative_phase=phi))]
            )
    return PulseSequence(
        [(ch, replace(pulse, amplitude=amplitude, relative_phase=phi))]
    )


class MissingNative(RuntimeError):
    """Missing native gate."""

    def __init__(self, gate: str):
        super().__init__(f"Native gate definition not found, for gate {gate}")


class NativeContainer(Model):
    def ensure(self, name: str) -> Native:
        value = getattr(self, name)
        if value is None:
            raise MissingNative(value)
        return value


class SingleQubitNatives(NativeContainer):
    """Container with the native single-qubit gates acting on a specific
    qubit."""

    RX: Optional[Native] = None
    """Pulse to drive the qubit from state 0 to state 1."""
    RX90: Optional[Native] = None
    """Pulse to drive the qubit from state 0 to state +"""
    RX12: Optional[Native] = None
    """Pulse to drive to qubit from state 1 to state 2."""
    MZ: Optional[Native] = None
    """Measurement pulse."""
    CP: Optional[Native] = None
    """Pulse to activate coupler."""

    def R(self, theta: float = np.pi, phi: float = 0.0) -> PulseSequence:
        """Create a sequence for single-qubit rotation.

        ``theta`` will be the angle of the rotation, while ``phi`` the angle that the rotation axis forms with x axis.
        """
        if self.RX90 is not None:
            return rotation(self.RX90, theta, phi, rx90=True)
        assert self.RX is not None
        return rotation(self.RX, theta, phi)


class TwoQubitNatives(NativeContainer):
    """Container with the native two-qubit gates acting on a specific pair of
    qubits."""

    CZ: Annotated[Optional[Native], {"symmetric": True}] = None
    CNOT: Annotated[Optional[Native], {"symmetric": False}] = None
    iSWAP: Annotated[Optional[Native], {"symmetric": True}] = None

    @property
    def symmetric(self):
        """Check if the defined two-qubit gates are symmetric between target
        and control qubits."""
        return all(
            info.metadata[0]["symmetric"] or getattr(self, fld) is None
            for fld, info in self.model_fields.items()
        )
