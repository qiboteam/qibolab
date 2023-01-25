from qibo import gates
from qibo.config import log

from qibolab.transpilers.connectivity import fix_connectivity
from qibolab.transpilers.gate_decompositions import translate_gate


def transpile(circuit, two_qubit_natives, fuse_one_qubit=False):
    """Implements full transpilation pipeline.

    Args:
        circuit (qibo.models.Circuit): Circuit model to transpile.
        two_qubit_natives (list): List of two qubit native gates
        supported by the quantum hardware ("CZ" and/or "iSWAP").

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
    new = translate_circuit(new, two_qubit_natives, translate_single_qubit=False)

    # Optional: fuse one-qubit gates to reduce circuit depth
    if fuse_one_qubit:
        new = new.fuse(max_qubits=1)

    # one-qubit gates to native
    new = translate_circuit(new, two_qubit_natives, translate_single_qubit=True)

    return new, hardware_qubits


def translate_circuit(circuit, two_qubit_natives, translate_single_qubit=False):
    """Translates a circuit to native gates.

    Args:
        circuit (qibo.models.Circuit): Circuit model to translate into native gates.
        two_qubit_natives (list): List of two qubit native gates
        supported by the quantum hardware ("CZ" and/or "iSWAP").

    Returns:
        new (qibo.models.Circuit): Equivalent circuit with native gates.
    """
    new = circuit.__class__(circuit.nqubits)
    for gate in circuit.queue:
        if len(gate.qubits) > 1 or translate_single_qubit:
            new.add(translate_gate(gate, two_qubit_natives))
        else:
            new.add(gate)
    return new


def can_execute(circuit, two_qubit_natives):
    """Checks if a circuit can be executed on tii5q.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.
        two_qubit_natives (list): List of two qubit native gates
        supported by the quantum hardware ("CZ" and/or "iSWAP").

    Returns ``True`` if the following conditions are satisfied:
        - Circuit does not contain more than two-qubit gates.
        - All one-qubit gates are I, Z, RZ or U3.
        - All two-qubit gates are CZ or iSWAP based on two_qubit_native.
        - All two-qubit gates have qubit 0 as target or control.
    otherwise returns ``False``.
    """
    CZ_is_native = "CZ" in two_qubit_natives
    iSWAP_is_native = "iSWAP" in two_qubit_natives
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue

        if len(gate.qubits) == 1:
            if not isinstance(gate, (gates.I, gates.Z, gates.RZ, gates.U3)):
                log.info(f"{gate.name} is not a single qubit native gate.")
                return False

        elif len(gate.qubits) == 2:
            if CZ_is_native and iSWAP_is_native:
                if not isinstance(gate, (gates.CZ, gates.iSWAP)):
                    log.info(f"{gate.name} is not a two qubit native gate.")
                    return False
            elif CZ_is_native:
                if not isinstance(gate, (gates.CZ)):
                    log.info(f"{gate.name} is not a two qubit native gate.")
                    return False
            elif iSWAP_is_native:
                if not isinstance(gate, (gates.iSWAP)):
                    log.info(f"{gate.name} is not a two qubit native gate.")
                    return False
            else:
                log.info("Use only CZ and/or iSWAP as native gates")
            if 0 not in gate.qubits:
                vlog("Circuit does not respect connectivity. " f"{gate.name} acts on {gate.qubits}.")
                return False

        else:
            vlog(f"{gate.name} acts on more than two qubits.")
            return False

    vlog("Circuit can be executed.")
    return True
