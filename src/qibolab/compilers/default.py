"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

from qibolab.pulses import PulseSequence


def identity_rule(gate, platform):
    """Identity gate skipped."""
    return PulseSequence(), {}


def z_rule(gate, platform):
    """Z gate applied virtually."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    return PulseSequence(), {qubit.name: math.pi}


def rz_rule(gate, platform):
    """RZ gate applied virtually."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    return PulseSequence(), {qubit.name: gate.parameters[0]}


def gpi2_rule(gate, platform):
    """Rule for GPI2."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    theta = gate.parameters[0]
    sequence = PulseSequence()
    pulse = qubit.native_gates.RX90
    pulse.relative_phase = theta
    sequence.append(pulse)
    return sequence, {}


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
    return sequence, {}


def u3_rule(gate, platform):
    """U3 applied as RZ-RX90-RZ-RX90-RZ."""
    qubit = platform.get_qubit(gate.target_qubits[0])
    # Transform gate to U3 and add pi/2-pulses
    theta, phi, lam = gate.parameters
    # apply RZ(lam)
    virtual_z_phases = {qubit.name: lam}
    sequence = PulseSequence()
    # Fetch pi/2 pulse from calibration
    rx90_pulse1 = qubit.native_gates.RX90
    rx90_pulse1.relative_phase = virtual_z_phases[qubit.name]
    # apply RX(pi/2)
    sequence.append(rx90_pulse1)
    # apply RZ(theta)
    virtual_z_phases[qubit.name] += theta
    # Fetch pi/2 pulse from calibration
    rx90_pulse2 = qubit.native_gates.RX90
    rx90_pulse2.relative_phase = (virtual_z_phases[qubit.name] - math.pi,)
    # apply RX(-pi/2)
    sequence.append(rx90_pulse2)
    # apply RZ(phi)
    virtual_z_phases[qubit.name] += phi

    return sequence, virtual_z_phases


def cz_rule(gate, platform):
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    pair = platform.pairs[tuple(platform.get_qubit(q) for q in gate.qubits)]
    return pair.native_gates.CZ


def cnot_rule(gate, platform):
    """CNOT applied as defined in the platform runcard."""
    pair = platform.pairs[tuple(platform.get_qubit(q) for q in gate.qubits)]
    return pair.native_gates.CNOT


def measurement_rule(gate, platform):
    """Measurement gate applied using the platform readout pulse."""
    sequence = PulseSequence(
        [platform.get_qubit(q).native_gates.MZ for q in gate.qubits]
    )
    return sequence, {}
