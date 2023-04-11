import math

from qibo import gates

from qibolab.compilers.compiler import Compiler
from qibolab.pulses import PulseSequence

compiler = Compiler()


@compiler.register(gates.I)
def identity_rule(gate, platform):
    return PulseSequence(), {}


@compiler.register(gates.Z)
def z_rule(gate, platform):
    qubit = gate.target_qubits[0]
    return PulseSequence(), {qubit: math.pi}


@compiler.register(gates.RZ)
def rz_rule(gate, platform):
    qubit = gate.target_qubits[0]
    return PulseSequence(), {qubit: gate.parameters[0]}


@compiler.register(gates.U3)
def u3_rule(gate, platform):
    qubit = gate.target_qubits[0]
    # Transform gate to U3 and add pi/2-pulses
    theta, phi, lam = gate.parameters
    # apply RZ(lam)
    virtual_z_phases = {qubit: lam}
    sequence = PulseSequence()
    # Fetch pi/2 pulse from calibration
    RX90_pulse_1 = platform.create_RX90_pulse(qubit, start=0, relative_phase=virtual_z_phases[qubit])
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
def measurement_rule(gate, platform):
    gate.pulses = ()
    sequence = PulseSequence()
    for qubit in gate.target_qubits:
        MZ_pulse = platform.create_MZ_pulse(qubit, start=0)
        sequence.add(MZ_pulse)
        gate.pulses = (*gate.pulses, MZ_pulse.serial)
    return sequence, {}


@compiler.register(gates.CZ)
def cz_rule(gate, platform):
    # create CZ pulse sequence with start time = 0
    return platform.create_CZ_pulse_sequence(gate.qubits)
