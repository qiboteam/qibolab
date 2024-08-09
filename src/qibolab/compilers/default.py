"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

import numpy as np
from qibo.gates import Gate

from qibolab.pulses import PulseSequence, VirtualZ
from qibolab.qubits import Qubit, QubitPair


def identity_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """Identity gate skipped."""
    return PulseSequence()


def z_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """Z gate applied virtually."""
    return PulseSequence([(qubit.drive.name, VirtualZ(phase=math.pi))])


def rz_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """RZ gate applied virtually."""
    return PulseSequence([(qubit.drive.name, VirtualZ(phase=gate.parameters[0]))])


def gpi2_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """Rule for GPI2."""
    return qubit.native_gates.RX.create_sequence(
        theta=np.pi / 2, phi=gate.parameters[0]
    )


def gpi_rule(gate: Gate, qubit: Qubit) -> PulseSequence:
    """Rule for GPI."""
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
    return qubit.native_gates.RX.create_sequence(theta=np.pi, phi=gate.parameters[0])


def cz_rule(gate: Gate, pair: QubitPair) -> PulseSequence:
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    return pair.native_gates.CZ.create_sequence()


def cnot_rule(gate: Gate, pair: QubitPair) -> PulseSequence:
    """CNOT applied as defined in the platform runcard."""
    return pair.native_gates.CNOT.create_sequence()


def measurement_rule(gate: Gate, qubits: list[Qubit]) -> PulseSequence:
    """Measurement gate applied using the platform readout pulse."""
    seq = PulseSequence()
    for qubit in qubits:
        seq.concatenate(qubit.native_gates.MZ.create_sequence())
    return seq
