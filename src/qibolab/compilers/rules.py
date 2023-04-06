import math

from qibo import gates

from qibolab.compilers.compiler import Compiler

compiler = Compiler()


@compiler.register(gates.I)
def identity_rule(sequence, virtual_z_phases, moment_start, gate, platform):
    return sequence, virtual_z_phases


@compiler.register(gates.Z)
def z_rule(sequence, virtual_z_phases, moment_start, gate, platform):
    qubit = gate.target_qubits[0]
    virtual_z_phases[qubit] += math.pi
    return sequence, virtual_z_phases


@compiler.register(gates.RZ)
def rz_rule(sequence, virtual_z_phases, moment_start, gate, platform):
    qubit = gate.target_qubits[0]
    virtual_z_phases[qubit] += gate.parameters[0]
    return sequence, virtual_z_phases


@compiler.register(gates.U3)
def u3_rule(sequence, virtual_z_phases, moment_start, gate, platform):
    qubit = gate.target_qubits[0]
    # Transform gate to U3 and add pi/2-pulses
    theta, phi, lam = gate.parameters
    # apply RZ(lam)
    virtual_z_phases[qubit] += lam
    # Fetch pi/2 pulse from calibration
    RX90_pulse_1 = platform.create_RX90_pulse(
        qubit,
        start=max(sequence.get_qubit_pulses(qubit).finish, moment_start),
        relative_phase=virtual_z_phases[qubit],
    )
    # apply RX(pi/2)
    sequence.add(RX90_pulse_1)
    # apply RZ(theta)
    virtual_z_phases[qubit] += theta
    # Fetch pi/2 pulse from calibration
    RX90_pulse_2 = platform.create_RX90_pulse(
        qubit, start=RX90_pulse_1.finish, relative_phase=virtual_z_phases[qubit] - math.pi
    )
    # apply RX(-pi/2)
    sequence.add(RX90_pulse_2)
    # apply RZ(phi)
    virtual_z_phases[qubit] += phi

    return sequence, virtual_z_phases


@compiler.register(gates.M)
def measurement_rule(sequence, virtual_z_phases, moment_start, gate, platform):
    measurement_start = max(sequence.get_qubit_pulses(*gate.target_qubits).finish, moment_start)
    gate.pulses = ()
    for qubit in gate.target_qubits:
        MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
        sequence.add(MZ_pulse)
        gate.pulses = (*gate.pulses, MZ_pulse.serial)

    return sequence, virtual_z_phases


@compiler.register(gates.CZ)
def cz_rule(sequence, virtual_z_phases, moment_start, gate, platform):
    # create CZ pulse sequence with start time = 0
    cz_sequence, cz_virtual_z_phases = platform.create_CZ_pulse_sequence(gate.qubits)

    # determine the right start time based on the availability of the qubits involved
    cz_qubits = {*cz_sequence.qubits, *gate.qubits}
    cz_start = max(sequence.get_qubit_pulses(*cz_qubits).finish, moment_start)

    # shift the pulses
    for pulse in cz_sequence.pulses:
        pulse.start += cz_start

    # add pulses to the sequence
    sequence.add(cz_sequence)

    # update z_phases registers
    for qubit in cz_virtual_z_phases:
        virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

    return sequence, virtual_z_phases
