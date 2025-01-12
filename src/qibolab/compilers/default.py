"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

from qibolab.pulses import PulseSequence


def identity_rule(qubits_ids, platform, parameters=None):
    """Identity gate skipped."""
    return PulseSequence(), {}


def z_rule(qubits_ids, platform, parameters=None):
    """Z gate applied virtually."""
    return PulseSequence(), {qubits_ids[1][0]: math.pi}


def rz_rule(qubits_ids, platform, parameters=None):
    """RZ gate applied virtually."""
    return PulseSequence(), {qubits_ids[1][0]: -parameters[0]}


def gpi2_rule(qubits_ids, platform, parameters=None):
    """Rule for GPI2."""
    theta = parameters[0]
    sequence = PulseSequence()
    pulse = platform.create_RX90_pulse(qubits_ids[1][0], start=0, relative_phase=theta)
    sequence.add(pulse)
    return sequence, {}


def gpi_rule(qubits_ids, platform, parameters=None):
    """Rule for GPI."""
    theta = parameters[0]
    sequence = PulseSequence()
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
    pulse = platform.create_RX_pulse(qubits_ids[1][0], start=0, relative_phase=theta)
    sequence.add(pulse)
    return sequence, {}


def u3_rule(qubits_ids, platform, parameters=None):
    """U3 applied as RZ-RX90-RZ-RX90-RZ."""
    qubit = qubits_ids[1][0]
    # Transform gate to U3 and add pi/2-pulses
    theta, phi, lam = parameters
    # apply RZ(lam)
    virtual_z_phases = {qubit: lam}
    sequence = PulseSequence()
    # Fetch pi/2 pulse from calibration
    RX90_pulse_1 = platform.create_RX90_pulse(
        qubit, start=0, relative_phase=virtual_z_phases[qubit]
    )
    # apply RX(pi/2)
    sequence.add(RX90_pulse_1)
    # apply RZ(theta)
    virtual_z_phases[qubit] += theta
    # Fetch pi/2 pulse from calibration
    RX90_pulse_2 = platform.create_RX90_pulse(
        qubit,
        start=RX90_pulse_1.finish,
        relative_phase=virtual_z_phases[qubit] - math.pi,
    )
    # apply RX(-pi/2)
    sequence.add(RX90_pulse_2)
    # apply RZ(phi)
    virtual_z_phases[qubit] += phi

    return sequence, virtual_z_phases


def cz_rule(qubits_ids, platform, parameters=None):
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    qubits = qubits_ids[0] + qubits_ids[1]
    return platform.create_CZ_pulse_sequence(qubits)


def cnot_rule(qubits_ids, platform, parameters=None):
    """CNOT applied as defined in the platform runcard."""
    qubits = qubits_ids[0] + qubits_ids[1]
    return platform.create_CNOT_pulse_sequence(qubits)


def measurement_rule(qubits_ids, platform, parameters=None):
    """Measurement gate applied using the platform readout pulse."""
    sequence = PulseSequence()
    for qubit in qubits_ids[1]:
        MZ_pulse = platform.create_MZ_pulse(qubit, start=0)
        sequence.add(MZ_pulse)
    return sequence, {}
