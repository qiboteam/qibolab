"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

from qibolab.pulses import PulseSequence, VirtualZ
from qibolab.serialize_ import replace


def identity_rule(gate, qubit):
    """Identity gate skipped."""
    return PulseSequence()


def z_rule(gate, qubit):
    """Z gate applied virtually."""
    return PulseSequence(
        [VirtualZ(phase=math.pi, channel=qubit.drive.name, qubit=qubit.name)]
    )


def rz_rule(gate, qubit):
    """RZ gate applied virtually."""
    return PulseSequence(
        [VirtualZ(phase=gate.parameters[0], channel=qubit.drive.name, qubit=qubit.name)]
    )


def gpi2_rule(gate, qubit):
    """Rule for GPI2."""
    theta = gate.parameters[0]
    pulse = replace(qubit.native_gates.RX90, relative_phase=theta)
    sequence = PulseSequence([pulse])
    return sequence


def gpi_rule(gate, qubit):
    """Rule for GPI."""
    theta = gate.parameters[0]
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
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
    sequence.append(VirtualZ(phase=lam, channel=qubit.drive.name, qubit=qubit.name))
    # Fetch pi/2 pulse from calibration and apply RX(pi/2)
    sequence.append(qubit.native_gates.RX90)
    # apply RZ(theta)
    sequence.append(VirtualZ(phase=theta, channel=qubit.drive.name, qubit=qubit.name))
    # Fetch pi/2 pulse from calibration and apply RX(-pi/2)
    sequence.append(replace(qubit.native_gates.RX90, relative_phase=-math.pi))
    # apply RZ(phi)
    sequence.append(VirtualZ(phase=phi, channel=qubit.drive.name, qubit=qubit.name))
    return sequence


def cz_rule(gate, pair):
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    return pair.native_gates.CZ


def cnot_rule(gate, pair):
    """CNOT applied as defined in the platform runcard."""
    return pair.native_gates.CNOT


def measurement_rule(gate, qubits):
    """Measurement gate applied using the platform readout pulse."""
    return PulseSequence([qubit.native_gates.MZ for qubit in qubits])
