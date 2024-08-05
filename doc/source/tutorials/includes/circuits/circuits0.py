import numpy as np
import qibo
from qibo import Circuit, gates

np.random.seed(0)

# create a single qubit circuit
circuit = Circuit(1)

# attach Hadamard gate and a measurement
circuit.add(gates.GPI2(0, phi=np.pi / 2))
circuit.add(gates.M(0))

# execute on quantum hardware
qibo.set_backend("qibolab", platform="dummy")
hardware_result = circuit(nshots=5000)

# retrieve measured probabilities
freq = hardware_result.frequencies()
p0 = freq["0"] / 5000 if "0" in freq else 0
p1 = freq["1"] / 5000 if "1" in freq else 0
hardware = [p0, p1]

# execute with classical quantum simulation
qibo.set_backend("numpy")
simulation_result = circuit(nshots=5000)

simulation = simulation_result.probabilities(qubits=(0,))

# print results
print(f"Qibolab: P(0) = {hardware[0]:.2f}\tP(1) = {hardware[1]:.2f}")
print(f"Numpy:   P(0) = {simulation[0]:.2f}\tP(1) = {simulation[1]:.2f}")
