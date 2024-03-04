"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math
from dataclasses import replace

from qibolab.pulses import PulseSequence, VirtualZ


def identity_rule(gate, platform):
    """Identity gate skipped."""
    return PulseSequence()


def z_rule(gate, platform):
    """Z gate applied virtually."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    return PulseSequence(
        [VirtualZ(phase=math.pi, channel=qubit.drive.name, qubit=qubit.name)]
    )


def rz_rule(gate, platform):
    """RZ gate applied virtually."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    return PulseSequence(
        [VirtualZ(phase=gate.parameters[0], channel=qubit.drive.name, qubit=qubit.name)]
    )


def gpi2_rule(gate, platform):
    """Rule for GPI2."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    theta = gate.parameters[0]
    sequence = PulseSequence()
    pulse = qubit.native_gates.RX90
    pulse.relative_phase = theta
    sequence.append(pulse)
    return sequence


def gpi_rule(gate, platform):
    """Rule for GPI."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    theta = gate.parameters[0]
    sequence = PulseSequence()
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
    pulse = qubit.native_gates.RX
    pulse.relative_phase = theta
    sequence.append(pulse)
    return sequence


def u3_rule(gate, platform):
    """U3 applied as RZ-RX90-RZ-RX90-RZ."""
    qubit = platform.get_qubit(gate.target_qubits[0])
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


def cz_rule(gate, platform):
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    pair = platform.pairs[tuple(platform.get_qubit(q).name for q in gate.qubits)]
    return pair.native_gates.CZ


def cnot_rule(gate, platform):
    """CNOT applied as defined in the platform runcard."""
    pair = platform.pairs[tuple(platform.get_qubit(q).name for q in gate.qubits)]
    return pair.native_gates.CNOT


def measurement_rule(gate, platform):
    """Measurement gate applied using the platform readout pulse."""
    sequence = PulseSequence(
        [platform.get_qubit(q).native_gates.MZ for q in gate.qubits]
    )
    return sequence
