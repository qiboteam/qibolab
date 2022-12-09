import numpy as np
from qibo import gates
from qibo.config import raise_error

from qibolab.transpilers.unitary_decompositions import (
    two_qubit_decomposition,
    u3_decomposition,
)


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
            decomposition = self.decompositions[gate.__class__](gate.parameters)
        else:
            decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) > 1))

    def count_1q(self, gate):
        """Count the number of single qubit gates in the decomposition of the given gate."""
        if gate.parameters:
            decomposition = self.decompositions[gate.__class__](gate.parameters)
        else:
            decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) == 1))

    def __call__(self, gate):
        """Decompose a gate."""
        if gate.parameters:
            decomposition = self.decompositions[gate.__class__](gate.parameters)
        else:
            decomposition = self.decompositions[gate.__class__]
        return [g.on_qubits({i: q for i, q in enumerate(gate.qubits)}) for g in decomposition]


def translate_gate(gate, native_gates):
    """Maps Qibo gates to a hardware native implementation.

    Args:
        gate (qibo.gates.abstract.Gate): gate to be decomposed.
        two_qubit_natives (list): List of two qubit native gates
        supported by the quantum hardware ("CZ" and/or "iSWAP").

    Returns:
        Shortest list of native gates
    """

    if len(gate.qubits) == 1:
        return onequbit_dec(gate)

    if "CZ" in native_gates and "iSWAP" in native_gates:
        # Check for a special optimized decomposition.
        if gate in opt_dec.decompositions:
            return opt_dec(gate)
        # Check the decomposition with less 2 qubit gates.
        else:
            if cz_dec.count_2q(gate) < iswap_dec.count_2q(gate):
                return cz_dec(gate)
            elif cz_dec.count_2q(gate) > iswap_dec.count_2q(gate):
                return iswap_dec(gate)
            # If equal check the decomposition with less 1 qubit gates.
            elif cz_dec.count_1q(gate) < iswap_dec.count_1q(gate):
                return cz_dec(gate)
            else:
                return iswap_dec(gate)
    elif "CZ" in native_gates:
        return cz_dec(gate)
    elif "iSWAP" in native_gates:
        if gate in iswap_dec.decompositions:
            return iswap_dec(gate)
        else:
            # First decompose
            return cz_dec(gate)
            # cz_decomposed = cz_dec(gate)
            # Now everithing will be decomposed into iSWAP
            # return [translate_gate(g, native_gates) for g in cz_decomposed]
    else:
        raise_error("Use only CZ and/or iSWAP as native gates")


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
# onequbit_dec.add(gates.M, [gates.M(0)])
onequbit_dec.add(gates.RX, lambda params: [gates.U3(0, params[0], -np.pi / 2, np.pi / 2)])
onequbit_dec.add(gates.RY, lambda params: [gates.U3(0, params[0], 0, 0)])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.RZ, lambda params: [gates.RZ(0, params[0])])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.U1, lambda params: [gates.RZ(0, params[0])])
onequbit_dec.add(gates.U2, lambda params: [gates.U3(0, np.pi / 2, params[0], params[1])])
onequbit_dec.add(gates.U3, lambda params: [gates.U3(0, params[0], params[1], params[2])])
onequbit_dec.add(
    gates.Unitary,
    lambda params: [
        gates.U3(0, u3_decomposition(params[0])[0], u3_decomposition(params[0])[1], u3_decomposition(params[0])[2])
    ],
)

# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.H(0),
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
        gates.H(0),
        gates.H(1),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
        gates.H(1),
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
    lambda params: [
        gates.RX(1, params[0] / 2.0),
        gates.CZ(0, 1),
        gates.RX(1, -params[0] / 2.0),
        gates.CZ(0, 1),
    ],
)
cz_dec.add(
    gates.CRY,
    lambda params: [
        gates.RY(1, params[0] / 2.0),
        gates.CZ(0, 1),
        gates.RY(1, -params[0] / 2.0),
        gates.CZ(0, 1),
    ],
)
cz_dec.add(
    gates.CRZ,
    lambda params: [
        gates.RZ(1, params[0] / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, -params[0] / 2.0),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.CU1,
    lambda params: [
        gates.RZ(0, params[0] / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, -params[0] / 2.0),
        gates.CZ(0, 1),
        gates.H(1),
        gates.RZ(1, params[0] / 2.0),
    ],
)
cz_dec.add(
    gates.CU2,
    lambda params: [
        gates.RZ(1, (params[1] - params[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, -np.pi / 4, 0, -(params[1] + params[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, np.pi / 4, params[0], 0),
    ],
)
cz_dec.add(
    gates.CU3,
    lambda params: [
        gates.RZ(1, (params[2] - params[1]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, -params[0] / 2.0, 0, -(params[2] + params[1]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, params[0] / 2.0, params[1], 0),
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
    lambda params: [
        gates.H(0),
        gates.CZ(0, 1),
        gates.RX(1, params[0]),
        gates.CZ(0, 1),
        gates.H(0),
    ],
)
cz_dec.add(
    gates.RYY,
    lambda params: [
        gates.RX(0, np.pi / 2),
        gates.U3(1, np.pi / 2, np.pi / 2, -np.pi),
        gates.CZ(0, 1),
        gates.RX(1, params[0]),
        gates.CZ(0, 1),
        gates.RX(0, -np.pi / 2),
        gates.U3(1, np.pi / 2, 0, np.pi / 2),
    ],
)
cz_dec.add(
    gates.RZZ,
    lambda params: [
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, params[0]),
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
cz_dec.add(gates.Unitary, lambda params: two_qubit_decomposition(0, 1, params[0]))


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
