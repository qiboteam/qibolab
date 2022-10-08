# -*- coding: utf-8 -*-
from qibo import gates
from qibo.config import log

from qibolab.transpilers.connectivity import fix_connectivity
from qibolab.transpilers.native import NativeGates


def transpile(circuit, fuse_one_qubit=False):
    """Implements full transpilation pipeline.

    Args:
        circuit (qibo.models.Circuit): Circuit model to transpile.

    Returns:
        new (qibo.models.Circuit): New circuit that can be executed on tii5q platform.
        hardware_qubits (dict): Mapping between logical and physical qubits.
    """
    # Re-arrange gates using qibo's fusion algorithm
    # this may reduce number of SWAPs when fixing for connectivity
    fcircuit = circuit.fuse(max_qubits=2)
    new = circuit.__class__(circuit.nqubits)
    for fgate in fcircuit.queue:
        if isinstance(fgate, gates.FusedGate):
            new.add(fgate.gates)
        else:
            new.add(fgate)

    # Add SWAPs to satisfy connectivity constraints
    new, hardware_qubits = fix_connectivity(circuit)

    # two-qubit gates to native
    native_gates = NativeGates()
    new = native_gates.translate_circuit(new, translate_single_qubit=False)

    # Optional: fuse one-qubit gates to reduce circuit depth
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
        if isinstance(gate, gates.M):
            continue

        if len(gate.qubits) == 1:
            if not isinstance(gate, (gates.I, gates.Z, gates.RZ, gates.U3)):
                log.info(f"{gate.name} is not native gate.")
                return False

        elif len(gate.qubits) == 2:
            if not isinstance(gate, gates.CZ):
                log.info(f"{gate.name} is not native gate.")
                return False
            if 0 not in gate.qubits:
                log.info("Circuit does not respect connectivity. " f"{gate.name} acts on {gate.qubits}.")
                return False

        else:
            log.info(f"{gate.name} acts on more than two qubits.")
            return False

    log.info("Circuit can be executed.")
    return True
