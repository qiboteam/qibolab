# -*- coding: utf-8 -*-
from qibo import gates
from qibo.transpilers.connectivity import fix_connecivity
from qibo.transpilers.native import NativeGates


def transpile(self, circuit, fuse_one_qubit=True):
    """Implements full transpilation pipeline."""
    native_gates = NativeGates()
    circuit1, hardware_qubits = fix_connecivity(circuit)

    # two-qubit gates to native
    circuit2 = circuit.__class__(circuit.nqubits)
    for gate in circuit1.queue:
        if len(gate.qubits) > 1:
            circuit2.add(native_gates(gate))
        else:
            circuit2.add(gate)

    # fuse one-qubit gates
    if fuse_one_qubit:
        circuit2 = circuit2.fuse(max_qubits=1)

    # one-qubit gates to native
    circuit3 = circuit.__class__(circuit.nqubits)
    for gate in circuit2.queue:
        if isinstance(gate, gates.FusedGate):
            matrix = gate.asmatrix(self)
            circuit3.add(native_gates(gates.Unitary(matrix, *gate.qubits)))
        else:
            circuit3.add(native_gates(gate))

    return circuit3, hardware_qubits


def can_execute(circuit):
    """Checks if a circuit can be executed on tii5q.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.

    Returns ``True`` if the following conditions are satisfied:
        - Circuit does not contain more than two-qubit gates.
        - All one-qubit gates are I, Z, RZ or U3.
        - All two-qubit gates are CZ.
        - All two-qubit gates have qubit 0 as target or control.
    otherwise returns ``False``.
    """
    for gate in circuit.queue:
        if len(gate.qubits) == 1:
            if not isinstance(gate, (gates.I, gates.Z, gates.RZ, gates.U3)):
                log.info(f"{gate.name} is not native gate.")
                return False

        elif len(gate.qubits) == 2:
            if not isinstance(gate, gates.CZ):
                log.info(f"{gate.name} is not native gate.")
                return False
            if 0 not in gate.qubits:
                log.info(
                    "Circuit does not respect connectivity. "
                    f"{gate.name} acts on {gate.qubits}."
                )
                return False

        else:
            log.info(f"{gate.name} acts on more than two qubits.")
            return False

    log.info("Circuit can be executed.")
    return True
