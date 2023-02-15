import numpy as np
from general_connectivity import Transpiler
from qibo import gates
from qibo.models import Circuit


def generate_random_circuit(nqubits, ngates):
    """Generate a random circuit with RX and CZ gates."""
    one_qubit_gates = [gates.RX]
    two_qubit_gates = [gates.CZ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta, trainable=False))
        else:
            circuit.add(gate(*q))
    return circuit


t = Transpiler(connectivity="21_qubits", init_method="greedy", init_samples=10)
# t.draw_connectivity()
circ = generate_random_circuit(4, 15)
print(circ.draw())
t.transpile(circ)
