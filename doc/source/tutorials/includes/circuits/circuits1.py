import matplotlib.pyplot as plt
import numpy as np
import qibo
from qibo import Circuit, gates


def execute_rotation():
    # create single qubit circuit
    circuit = Circuit(1)

    # attach Rotation on X-Pauli with angle = 0
    circuit.add(gates.GPI2(0, phi=0))
    circuit.add(gates.M(0))

    # define range of angles from [0, 2pi]
    exp_angles = np.arange(0, 2 * np.pi, np.pi / 16)

    res = []
    for angle in exp_angles:
        # update circuit's rotation angle
        circuit.set_parameters([angle])

        # execute circuit
        result = circuit.execute(nshots=4000)
    freq = result.frequencies()
    p0 = freq["0"] / 4000 if "0" in freq else 0
    p1 = freq["1"] / 4000 if "1" in freq else 0

    # store probability in state |1>
    res.append(p1)

    return res


# execute on quantum hardware
qibo.set_backend("qibolab", platform="dummy")
hardware = execute_rotation()

# execute with classical quantum simulation
qibo.set_backend("numpy")
simulation = execute_rotation()

# plot results
exp_angles = np.arange(0, 2 * np.pi, np.pi / 16)
plt.plot(exp_angles, hardware, label="qibolab hardware")
plt.plot(exp_angles, simulation, label="numpy")

plt.legend()
plt.ylabel("P(1)")
plt.xlabel("Rotation [rad]")
plt.show()
