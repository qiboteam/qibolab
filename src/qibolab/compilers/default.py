"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math
from dataclasses import replace

import numpy as np

from qibolab.pulses import PulseSequence, VirtualZ


def identity_rule(gate, qubit):
    """Identity gate skipped."""
    return PulseSequence()


def z_rule(gate, qubit):
    """Z gate applied virtually."""
<<<<<<< HEAD
    seq = PulseSequence()
    seq[qubit.drive.name].append(VirtualZ(phase=math.pi))
    return seq
=======
    return PulseSequence([VirtualZ(phase=math.pi)])
>>>>>>> c5ef5a19 (remove channel and qubit properties from pulse)


def rz_rule(gate, qubit):
    """RZ gate applied virtually."""
<<<<<<< HEAD
    seq = PulseSequence()
    seq[qubit.drive.name].append(VirtualZ(phase=gate.parameters[0]))
    return seq
=======
    return PulseSequence([VirtualZ(phase=gate.parameters[0])])
>>>>>>> c5ef5a19 (remove channel and qubit properties from pulse)


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
<<<<<<< HEAD
    return qubit.native_gates.RX.create_sequence(theta=np.pi, phi=gate.parameters[0])
=======
    pulse = replace(qubit.native_gates.RX, relative_phase=theta)
    sequence = PulseSequence([pulse])
    return sequence


def u3_rule(gate, qubit):
    """U3 applied as RZ-RX90-RZ-RX90-RZ."""
    # Transform gate to U3 and add pi/2-pulses
    theta, phi, lam = gate.parameters
    # apply RZ(lam)
    virtual_z_phases = {qubit.name: lam}
    sequence = PulseSequence()
    sequence.append(VirtualZ(phase=lam))
    # Fetch pi/2 pulse from calibration and apply RX(pi/2)
    sequence.append(qubit.native_gates.RX90)
    # apply RZ(theta)
    sequence.append(VirtualZ(phase=theta))
    # Fetch pi/2 pulse from calibration and apply RX(-pi/2)
    sequence.append(replace(qubit.native_gates.RX90, relative_phase=-math.pi))
    # apply RZ(phi)
    sequence.append(VirtualZ(phase=phi))
    return sequence
>>>>>>> c5ef5a19 (remove channel and qubit properties from pulse)


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
