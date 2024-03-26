from dataclasses import dataclass, field, fields, replace
from typing import Optional

from qibolab.pulses import Pulse, PulseSequence


@dataclass
class SingleQubitNatives:
    """Container with the native single-qubit gates acting on a specific
    qubit."""

    RX: Optional[Pulse] = None
    """Pulse to drive the qubit from state 0 to state 1."""
    RX12: Optional[Pulse] = None
    """Pulse to drive to qubit from state 1 to state 2."""
    MZ: Optional[Pulse] = None
    """Measurement pulse."""
    CP: Optional[Pulse] = None
    """Pulse to activate a coupler."""

    @property
    def RX90(self) -> Pulse:
        """RX90 native pulse is inferred from RX by halving its amplitude."""
        return replace(self.RX, amplitude=self.RX.amplitude / 2.0)


@dataclass
class TwoQubitNatives:
    """Container with the native two-qubit gates acting on a specific pair of
    qubits."""

    CZ: PulseSequence = field(
        default_factory=lambda: PulseSequence(), metadata={"symmetric": True}
    )
    CNOT: PulseSequence = field(
        default_factory=lambda: PulseSequence(), metadata={"symmetric": False}
    )
    iSWAP: PulseSequence = field(
        default_factory=lambda: PulseSequence(), metadata={"symmetric": True}
    )

    @property
    def symmetric(self):
        """Check if the defined two-qubit gates are symmetric between target
        and control qubits."""
        return all(
            fld.metadata["symmetric"] or len(getattr(self, fld.name)) == 0
            for fld in fields(self)
        )
