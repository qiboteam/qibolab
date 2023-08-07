.. _tutorials_circuits:

Circuit execution
=================

Qibolab can be used as a ``qibo`` backend for executing executions. The purpose
of this section is to show how to do it, without entering into the details of
circuits definition that we leave to the `Qibo
<https://qibo.science/qibo/stable/>`_ documentation.

.. code-block:: python

    import qibo
    from qibo import Circuit, gates

    # create a single qubit circuit
    circuit = Circuit(1)

    # attach Hadamard gate and a measurement
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    # execute on quantum hardware
    qibo.set_backend("qibolab", "tii1q_b1")
    hardware_result = circuit(nshots=5000)

    # execute with classical quantum simulation
    qibo.set_backend("numpy")
    simulation_result = circuit(nshots=5000)

    # retrieve measured probabilities
    hardware = hardware_result.probabilities(qubits=(0,))
    simulation = simulation_result.probabilities(qubits=(0,))


In this snippet, we first define a single-qubit circuit containing a single Hadamard gate and a measurement.
We then proceed to define the qibo backend as ``qibolab`` using the ``tii1q_b1`` platform.
Finally, we change the backend to ``numpy``, a simulation one, to compare the results with ideality.
After executing the script we can print our results that will appear more or less as:

.. code-block:: python

    print(f"Qibolab: P(0) = {hardware[0]:.2f}\tP(1) = {hardware[1]:.2f}")
    print(f"Numpy:   P(0) = {simulation[0]:.2f}\tP(1) = {simulation[1]:.2f}")

Returns:

.. code-block:: text

    > Qibolab: P(0) = 0.54 P(1) = 0.46
    > Numpy:   P(0) = 0.50 P(1) = 0.40

Clearly, we do not expect the results to be exactly equal due to the non
ideality of current NISQ devices.

.. note::
   Qibo circuits and gates are backend agnostic. The same circuit can be executed on multiple backends, including simulation and quantum platforms.

A slightly more complex circuit, a variable rotation, will produce similar
results:

.. code-block:: python

    import numpy as np
    from qibo import Circuit, gates


    def execute_rotation():
        # create single qubit circuit
        circuit = Circuit(1)

        # attach Rotation on X-Pauli with angle = 0
        circuit.add(gates.RX(0, theta=0))
        circuit.add(gates.M(0))

        # define range of angles from [0, 2pi]
        exp_angles = np.arange(0, 2 * np.pi, np.pi / 16)

        res = []
        for angle in exp_angles:
            # update circuit's rotation angle
            circuit.set_parameters([angle])

            # execute circuit
            state = circuit.execute(nshots=4000)
            p0, p1 = state.probabilities(qubits=(0,))

            # store probability in state |1>
            res.append(p1)

        return res


    # execute on quantum hardware
    qibo.set_backend("qibolab", "tii1q_b1")
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

Returns the following plot:

.. image:: rotation.svg
