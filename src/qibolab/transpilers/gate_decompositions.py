import copy

import numpy as np
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import raise_error

from qibolab.transpilers.unitary_decompositions import (
    two_qubit_decomposition,
    u3_decomposition,
)

backend = NumpyBackend()


class GateDecompositions:
    """Abstract data structure that holds decompositions of gates."""

    def __init__(self):
        self.decompositions = {}

    def add(self, gate, decomposition):
        """Register a decomposition for a gate."""
        self.decompositions[gate] = decomposition

    def count_2q(self, gate):
        """Count the number of two-qubit gates in the decomposition of the given gate."""
        if gate.parameters:
            decomposition = self.decompositions[gate.__class__](gate)
        else:
            decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) > 1))

    def count_1q(self, gate):
        """Count the number of single qubit gates in the decomposition of the given gate."""
        if gate.parameters:
            decomposition = self.decompositions[gate.__class__](gate)
        else:
            decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) == 1))

    def __call__(self, gate):
        """Decompose a gate."""
        if gate.parameters or (gate.__class__ is gates.FusedGate):
            decomposition = self.decompositions[gate.__class__](gate)
        else:
            decomposition = self.decompositions[gate.__class__]
        return [g.on_qubits({i: q for i, q in enumerate(gate.qubits)}) for g in decomposition]


def translate_gate(gate, native_gates):
    """Maps Qibo gates to a hardware native implementation.

    Args:
        gate (qibo.gates.abstract.Gate): gate to be decomposed.
        native_gates (list): List of two qubit native gates
            supported by the quantum hardware ("CZ" and/or "iSWAP").

    Returns:
        Shortest list of native gates
    """
    if isinstance(gate, gates.M):
        return gate

    if len(gate.qubits) == 1:
        return onequbit_dec(gate)

    if "CZ" in native_gates and "iSWAP" in native_gates:
        # Check for a special optimized decomposition.
        if gate.__class__ in opt_dec.decompositions:
            return opt_dec(gate)
        # Check if the gate has a CZ decomposition
        if not gate.__class__ in iswap_dec.decompositions:
            return cz_dec(gate)
        # Check the decomposition with less 2 qubit gates.
        else:
            if cz_dec.count_2q(gate) < iswap_dec.count_2q(gate):
                return cz_dec(gate)
            elif cz_dec.count_2q(gate) > iswap_dec.count_2q(gate):
                return iswap_dec(gate)
            # If equal check the decomposition with less 1 qubit gates.
            # This is never used for now but may be useful for future generalization
            elif cz_dec.count_1q(gate) < iswap_dec.count_1q(gate):  # pragma: no cover
                return cz_dec(gate)
            else:  # pragma: no cover
                return iswap_dec(gate)
    elif "CZ" in native_gates:
        return cz_dec(gate)
    elif "iSWAP" in native_gates:
        if gate.__class__ in iswap_dec.decompositions:
            return iswap_dec(gate)
        else:
            # First decompose into CZ
            cz_decomposed = cz_dec(gate)
            # Then CZ are decomposed into iSWAP
            iswap_decomposed = []
            for g in cz_decomposed:
                # Need recursive function as gates.Unitary is not in iswap_dec
                for g_translated in translate_gate(g, ["iSWAP"]):
                    iswap_decomposed.append(g_translated)
            return iswap_decomposed
    else:  # pragma: no cover
        raise_error(NotImplementedError, "Use only CZ and/or iSWAP as native gates")


onequbit_dec = GateDecompositions()
onequbit_dec.add(gates.H, [gates.U3(0, 7 * np.pi / 2, np.pi, 0)])
onequbit_dec.add(gates.X, [gates.U3(0, np.pi, 0, np.pi)])
onequbit_dec.add(gates.Y, [gates.U3(0, np.pi, 0, 0)])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.Z, [gates.Z(0)])
onequbit_dec.add(gates.S, [gates.RZ(0, np.pi / 2)])
onequbit_dec.add(gates.SDG, [gates.RZ(0, -np.pi / 2)])
onequbit_dec.add(gates.T, [gates.RZ(0, np.pi / 4)])
onequbit_dec.add(gates.TDG, [gates.RZ(0, -np.pi / 4)])
onequbit_dec.add(gates.I, [gates.I(0)])
onequbit_dec.add(gates.Align, [gates.Align(0)])
onequbit_dec.add(gates.RX, lambda gate: [gates.U3(0, gate.parameters[0], -np.pi / 2, np.pi / 2)])
onequbit_dec.add(gates.RY, lambda gate: [gates.U3(0, gate.parameters[0], 0, 0)])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.U1, lambda gate: [gates.RZ(0, gate.parameters[0])])
onequbit_dec.add(gates.U2, lambda gate: [gates.U3(0, np.pi / 2, gate.parameters[0], gate.parameters[1])])
onequbit_dec.add(gates.U3, lambda gate: [gates.U3(0, gate.parameters[0], gate.parameters[1], gate.parameters[2])])
onequbit_dec.add(
    gates.Unitary,
    lambda gate: [gates.U3(0, *u3_decomposition(gate.parameters[0]))],
)
onequbit_dec.add(
    gates.FusedGate,
    lambda gate: [gates.U3(0, *u3_decomposition(gate.asmatrix(backend)))],
)

# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.U3(0, 7 * np.pi / 2, np.pi, 0),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
    ],
)
iswap_dec.add(
    gates.CZ,
    [
        gates.U3(0, 7 * np.pi / 2, np.pi, 0),
        gates.U3(1, 7 * np.pi / 2, np.pi, 0),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
        gates.U3(1, 7 * np.pi / 2, np.pi, 0),
    ],
)
iswap_dec.add(
    gates.SWAP,
    [
        gates.iSWAP(0, 1),
        gates.U3(1, np.pi / 2, -np.pi / 2, np.pi / 2),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, -np.pi / 2, np.pi / 2),
        gates.iSWAP(0, 1),
        gates.U3(1, np.pi / 2, -np.pi / 2, np.pi / 2),
    ],
)
iswap_dec.add(gates.iSWAP, [gates.iSWAP(0, 1)])

# register CZ decompositions
cz_dec = GateDecompositions()
cz_dec.add(gates.CNOT, [gates.H(1), gates.CZ(0, 1), gates.H(1)])
cz_dec.add(gates.CZ, [gates.CZ(0, 1)])
cz_dec.add(
    gates.SWAP,
    [
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.H(0),
        gates.CZ(1, 0),
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.iSWAP,
    [
        gates.U3(0, np.pi / 2.0, 0, -np.pi / 2.0),
        gates.U3(1, np.pi / 2.0, 0, -np.pi / 2.0),
        gates.CZ(0, 1),
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(0),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.CRX,
    lambda gate: [
        gates.RX(1, gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.RX(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
    ],
)
cz_dec.add(
    gates.CRY,
    lambda gate: [
        gates.RY(1, gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.RY(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
    ],
)
cz_dec.add(
    gates.CRZ,
    lambda gate: [
        gates.RZ(1, gate.parameters[0] / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.CU1,
    lambda gate: [
        gates.RZ(0, gate.parameters[0] / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.H(1),
        gates.RZ(1, gate.parameters[0] / 2.0),
    ],
)
cz_dec.add(
    gates.CU2,
    lambda gate: [
        gates.RZ(1, (gate.parameters[1] - gate.parameters[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, -np.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, np.pi / 4, gate.parameters[0], 0),
    ],
)
cz_dec.add(
    gates.CU3,
    lambda gate: [
        gates.RZ(1, (gate.parameters[2] - gate.parameters[1]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, -gate.parameters[0] / 2.0, 0, -(gate.parameters[2] + gate.parameters[1]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, gate.parameters[0] / 2.0, gate.parameters[1], 0),
    ],
)
cz_dec.add(
    gates.FSWAP,
    [
        gates.U3(0, np.pi / 2, -np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, np.pi / 2, np.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, np.pi / 2, 0, -np.pi / 2),
        gates.U3(1, np.pi / 2, 0, np.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, 0, -np.pi),
    ],
)
cz_dec.add(
    gates.RXX,
    lambda gate: [
        gates.H(0),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.H(0),
    ],
)
cz_dec.add(
    gates.RYY,
    lambda gate: [
        gates.RX(0, np.pi / 2),
        gates.U3(1, np.pi / 2, np.pi / 2, -np.pi),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.RX(0, -np.pi / 2),
        gates.U3(1, np.pi / 2, 0, np.pi / 2),
    ],
)
cz_dec.add(
    gates.RZZ,
    lambda gate: [
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.TOFFOLI,
    [
        gates.CZ(1, 2),
        gates.RX(2, -np.pi / 4),
        gates.CZ(0, 2),
        gates.RX(2, np.pi / 4),
        gates.CZ(1, 2),
        gates.RX(2, -np.pi / 4),
        gates.CZ(0, 2),
        gates.RX(2, np.pi / 4),
        gates.RZ(1, np.pi / 4),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RZ(0, np.pi / 4),
        gates.RX(1, -np.pi / 4),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(gates.Unitary, lambda gate: two_qubit_decomposition(0, 1, gate.parameters[0]))
cz_dec.add(gates.fSim, lambda gate: two_qubit_decomposition(0, 1, gate.asmatrix(backend)))
cz_dec.add(gates.GeneralizedfSim, lambda gate: two_qubit_decomposition(0, 1, gate.asmatrix(backend)))


# register other optimized gate decompositions
opt_dec = GateDecompositions()
opt_dec.add(
    gates.SWAP,
    [
        gates.H(0),
        gates.SDG(0),
        gates.SDG(1),
        gates.iSWAP(0, 1),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
