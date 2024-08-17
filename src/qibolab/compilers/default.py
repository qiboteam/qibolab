"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

import numpy as np
from qibo.gates import Align, Gate

from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.pulses import Delay, PulseSequence, VirtualZ
from qibolab.qubits import Qubit


def z_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """Z gate applied virtually."""
    return PulseSequence([(qubit.drive.name, VirtualZ(phase=math.pi))])


def rz_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """RZ gate applied virtually."""
    return PulseSequence([(qubit.drive.name, VirtualZ(phase=gate.parameters[0]))])


def identity_rule(gate: Gate, natives: SingleQubitNatives) -> PulseSequence:
    """Identity gate skipped."""
    return PulseSequence()


def gpi2_rule(gate: Gate, natives: SingleQubitNatives) -> PulseSequence:
    """Rule for GPI2."""
    assert natives.RX is not None
    return natives.RX.create_sequence(theta=np.pi / 2, phi=gate.parameters[0])


def gpi_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """Rule for GPI."""
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
    return qubit.RX.create_sequence(theta=np.pi, phi=gate.parameters[0])


def cz_rule(gate: Gate, natives: TwoQubitNatives) -> PulseSequence:
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    assert natives.CZ is not None
    return natives.CZ.create_sequence()


def cnot_rule(gate: Gate, natives: TwoQubitNatives) -> PulseSequence:
    """CNOT applied as defined in the platform runcard."""
    assert natives.CNOT is not None
    return natives.CNOT.create_sequence()


def measurement_rule(gate: Gate, natives: list[SingleQubitNatives]) -> PulseSequence:
    """Measurement gate applied using the platform readout pulse."""
    seq = PulseSequence()
    for qubit in natives:
        assert qubit.MZ is not None
        seq.concatenate(qubit.MZ.create_sequence())
    return seq


def align_rule(gate: Align, qubits: list[Qubit]) -> PulseSequence:
    """Measurement gate applied using the platform readout pulse."""
    if gate.delay == 0.0:
        return PulseSequence()
    return PulseSequence(
        [
            (ch.name, Delay(duration=gate.delay))
            for qubit in qubits
            for ch in qubit.channels
        ]
    )
