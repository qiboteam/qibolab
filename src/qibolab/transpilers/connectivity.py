# -*- coding: utf-8 -*-
from qibo import gates
from qibo.config import log, raise_error


def find_connected_qubit(qubits, queue, hardware_qubits):
    """Helper method for :meth:`qibolab.transpilers.fix_connecivity`.

    Finds which qubit should be mapped to hardware qubit 0 (middle)
    by looking at the two-qubit gates that follow.
    """
    possible_qubits = set(qubits)
    for next_gate in queue:
        if len(next_gate.qubits) == 2:
            possible_qubits &= {hardware_qubits.index(q) for q in next_gate.qubits}
            if not possible_qubits:
                # freedom of choice
                return qubits[0]
            elif len(possible_qubits) == 1:
                return possible_qubits.pop()
    # freedom of choice
    return qubits[0]


def fix_connectivity(circuit):
    """Transforms an arbitrary circuit to one that can be executed on hardware.

    This method produces a circuit that respects the following connectivity:
          1
          |
     2 -- 0 -- 3
          |
          4
    by adding SWAP gates when needed.
    It does not translate gates to native.

    Args:
        circuit (qibo.models.Circuit): The original Qibo circuit to transform.
            This circuit must contain up to two-qubit gates.

    Returns:
        new (qibo.models.Circuit): Qibo circuit that performs the same operation
            as the original but respects the hardware connectivity.
        hardware_qubits (list): List that maps logical to hardware qubits.
            This is required for transforming final measurements.
    """
    # TODO: Change this to a more lightweight form that takes a list of pairs
    # instead of the whole circuit.

    # new circuit object that will be compatible to hardware connectivity
    new = circuit.__class__(circuit.nqubits)
    # list to maps logical to hardware qubits
    hardware_qubits = list(range(circuit.nqubits))

    # find initial qubit mapping
    for i, gate in enumerate(circuit.queue):
        if len(gate.qubits) == 2:
            if 0 not in gate.qubits:
                new_zero = find_connected_qubit(gate.qubits, circuit.queue[i + 1 :], hardware_qubits)
                hardware_qubits[0], hardware_qubits[new_zero] = (
                    hardware_qubits[new_zero],
                    hardware_qubits[0],
                )
            break

    # the first SWAP is not needed as it can be applied via virtual mapping
    add_swap = False
    for i, gate in enumerate(circuit.queue):
        # map gate qubits to hardware
        qubits = tuple(hardware_qubits.index(q) for q in gate.qubits)
        if len(qubits) > 2 and not isinstance(gate, gates.M):
            raise_error(
                NotImplementedError,
                "Transpiler does not support gates targeting more than two-qubits.",
            )

        elif len(qubits) == 2 and 0 not in qubits:
            # find which qubit should be moved to 0
            new_zero = find_connected_qubit(qubits, circuit.queue[i + 1 :], hardware_qubits)
            # update hardware qubits according to the swap
            hardware_qubits[0], hardware_qubits[new_zero] = (
                hardware_qubits[new_zero],
                hardware_qubits[0],
            )
            if add_swap:
                new.add(gates.SWAP(0, new_zero))
            # update gate qubits according to the new swap
            qubits = tuple(hardware_qubits.index(q) for q in gate.qubits)

        # add gate to the hardware circuit
        new.add(gate.__class__(*qubits, **gate.init_kwargs))
        if len(qubits) == 2:
            add_swap = True

    return new, hardware_qubits


def respects_connectivity(circuit):
    """Checks if a circuit respects connectivity constraints.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.

    Returns ``True`` if the following conditions are satisfied:
        - Circuit does not contain more than two-qubit gates.
        - All two-qubit gates have qubit 0 as target or control.
    otherwise returns ``False``.
    """
    for gate in circuit.queue:
        if len(gate.qubits) > 2 and not isinstance(gate, gates.M):
            log.info(f"{gate.name} acts on more than two qubits.")
            return False
        elif len(gate.qubits) == 2:
            if 0 not in gate.qubits:
                log.info("Circuit does not respect connectivity. " f"{gate.name} acts on {gate.qubits}.")
                return False

    log.info("Circuit can be executed.")
    return True
