from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Annotated, Optional

import numpy as np

from .pulses import Pulse
from .sequence import PulseSequence
from .serialize import Model, replace


def _normalize_angles(theta, phi):
    """Normalize theta to (-pi, pi], and phi to [0, 2*pi)."""
    theta = theta % (2 * np.pi)
    theta = theta - 2 * np.pi * (theta > np.pi)
    phi = phi % (2 * np.pi)
    return theta, phi


class Native(ABC, PulseSequence):
    @abstractmethod
    def create_sequence(self, *args, **kwargs) -> PulseSequence:
        """Create a sequence for single-qubit rotation."""

    def __call__(self, *args, **kwargs) -> PulseSequence:
        """Create a sequence for single-qubit rotation.

        Alias to :meth:`create_sequence`.
        """
        return self.create_sequence(*args, **kwargs)


def rxy(seq: PulseSequence, theta: float = np.pi, phi: float = 0.0) -> PulseSequence:
    """Create a sequence for single-qubit rotation.

    ``theta`` will be the angle of the rotation, while ``phi`` the angle that the rotation axis forms with x axis.
    """
    theta, phi = _normalize_angles(theta, phi)
    ch, pulse = seq[0]
    assert isinstance(pulse, Pulse)
    new_amplitude = pulse.amplitude * theta / np.pi
    return PulseSequence(
        [(ch, replace(pulse, amplitude=new_amplitude, relative_phase=phi))]
    )


class FixedSequenceFactory(Native):
    """Simple factory for a fixed arbitrary sequence."""

    def create_sequence(self) -> PulseSequence:
        return deepcopy(self)


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

    RX: Optional[FixedSequenceFactory] = None
    """Pulse to drive the qubit from state 0 to state 1."""
    RX12: Optional[FixedSequenceFactory] = None
    """Pulse to drive to qubit from state 1 to state 2."""
    MZ: Optional[FixedSequenceFactory] = None
    """Measurement pulse."""
    CP: Optional[FixedSequenceFactory] = None
    """Pulse to activate coupler."""

    def RXY(self, theta: float = np.pi, phi: float = 0.0) -> PulseSequence:
        """Create a sequence for single-qubit rotation.

        ``theta`` will be the angle of the rotation, while ``phi`` the angle that the rotation axis forms with x axis.
        """
        assert self.RX is not None
        return rxy(self.RX, theta, phi)


class TwoQubitNatives(NativeContainer):
    """Container with the native two-qubit gates acting on a specific pair of
    qubits."""

    CZ: Annotated[Optional[FixedSequenceFactory], {"symmetric": True}] = None
    CNOT: Annotated[Optional[FixedSequenceFactory], {"symmetric": False}] = None
    iSWAP: Annotated[Optional[FixedSequenceFactory], {"symmetric": True}] = None

    @property
    def symmetric(self):
        """Check if the defined two-qubit gates are symmetric between target
        and control qubits."""
        return all(
            info.metadata[0]["symmetric"] or getattr(self, fld) is None
            for fld, info in self.model_fields.items()
        )
