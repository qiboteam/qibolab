# -*- coding: utf-8 -*-
from qibo import gates
from qibo.config import log

from qibolab.transpilers.connectivity import fix_connecivity
from qibolab.transpilers.native import NativeGates


def transpile(circuit, fuse_one_qubit=True):
    """Implements full transpilation pipeline."""
    native_gates = NativeGates()
    new, hardware_qubits = fix_connecivity(circuit)
    # two-qubit gates to native
    new = native_gates.translate_circuit(new, translate_single_qubit=False)
    # fuse one-qubit gates
    if fuse_one_qubit:
        new = new.fuse(max_qubits=1)
    # one-qubit gates to native
    new = native_gates.translate_circuit(new, translate_single_qubit=True)
    return new, hardware_qubits


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
