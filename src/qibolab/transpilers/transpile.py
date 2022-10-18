# -*- coding: utf-8 -*-
from qibo import gates
from qibo.config import log

from qibolab.transpilers.connectivity import fix_connectivity
from qibolab.transpilers.native import NativeGates


def transpile(circuit, fuse_one_qubit=False):
    """Implements full transpilation pipeline.

    Args:
        circuit (qibo.models.Circuit): Circuit model to transpile.
        fuse_one_qubit (bool): If ``True`` it fuses adjacent one-qubit gates to reduce
            circuit depth.

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


def can_execute(circuit, verbose=True):
    """Checks if a circuit can be executed on tii5q.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.
        verbose (bool): If ``True`` it prints debugging log messages.

    Returns ``True`` if the following conditions are satisfied:
        - Circuit does not contain more than two-qubit gates.
        - All one-qubit gates are I, Z, RZ or U3.
        - All two-qubit gates are CZ.
        - All two-qubit gates have qubit 0 as target or control.
    otherwise returns ``False``.
    """
    # pring messages only if ``verbose == True``
    vlog = lambda msg: log.info(msg) if verbose else lambda msg: None
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue

        if len(gate.qubits) == 1:
            if not isinstance(gate, (gates.I, gates.Z, gates.RZ, gates.U3)):
                if verbose:
                    vlog(f"{gate.name} is not native gate.")
                return False

        elif len(gate.qubits) == 2:
            if not isinstance(gate, gates.CZ):
                vlog(f"{gate.name} is not native gate.")
                return False
            if 0 not in gate.qubits:
                vlog("Circuit does not respect connectivity. " f"{gate.name} acts on {gate.qubits}.")
                return False

        else:
            vlog(f"{gate.name} acts on more than two qubits.")
            return False

    vlog("Circuit can be executed.")
    return True
