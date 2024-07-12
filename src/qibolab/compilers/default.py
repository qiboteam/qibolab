"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

import numpy as np

from qibolab.pulses import PulseSequence, VirtualZ


def identity_rule(gate, qubit):
    """Identity gate skipped."""
    return PulseSequence()


def z_rule(gate, qubit):
    """Z gate applied virtually."""
    seq = PulseSequence()
    seq[qubit.drive.name].append(VirtualZ(phase=math.pi))
    return seq


def rz_rule(gate, qubit):
    """RZ gate applied virtually."""
    seq = PulseSequence()
    seq[qubit.drive.name].append(VirtualZ(phase=gate.parameters[0]))
    return seq


def gpi2_rule(gate, qubit):
    """Rule for GPI2."""
    return qubit.native_gates.RX.create_sequence(
        theta=np.pi / 2, phi=gate.parameters[0]
    )


def gpi_rule(gate, qubit):
    """Rule for GPI."""
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
    return qubit.native_gates.RX.create_sequence(theta=np.pi, phi=gate.parameters[0])


def cz_rule(gate, pair):
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    return pair.native_gates.CZ.create_sequence()


def cnot_rule(gate, pair):
    """CNOT applied as defined in the platform runcard."""
    return pair.native_gates.CNOT.create_sequence()


def measurement_rule(gate, qubits):
    """Measurement gate applied using the platform readout pulse."""
    seq = PulseSequence()
    for qubit in qubits:
        seq.extend(qubit.native_gates.MZ.create_sequence())
    return seq
