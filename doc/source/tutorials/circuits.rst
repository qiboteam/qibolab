.. _tutorials_circuits:

Executing a circuit
===================

This tutorial follows a circuit from construction to acquired measurement
samples. We use the built-in ``dummy`` platform so the workflow can be checked
without hardware access. The purpose is to validate gate compatibility,
measurement bookkeeping and result handling, not to reproduce an ideal quantum
distribution. For a pulse-level physical simulation, continue with
:ref:`tutorials_emulator`.

Install ``qibolab[backend]`` before running these examples:

.. code-block:: console

    pip install "qibolab[backend]"

Select a platform and write a native circuit
--------------------------------------------

Construct an explicit Qibo backend first, so its platform and supported
operations are available while preparing the circuit. The dummy platform has
physical qubits 0 through 4 and includes a CZ native sequence on pair ``(0, 2)``.
We keep the default integer placement here: three logical wires correspond to
physical qubits 0, 1 and 2.

.. testcode:: circuit-execution

    import numpy as np
    from qibo import Circuit, construct_backend, gates

    backend = construct_backend("qibolab", platform="dummy")
    platform = backend.platform
    assert (0, 2) in backend.connectivity
    assert platform.natives.two_qubit[(0, 2)].CZ is not None

    circuit = Circuit(3, wire_names=[0, 1, 2])
    circuit.add(gates.GPI2(0, phi=np.pi / 2))
    circuit.add(gates.CZ(0, 2))
    measurement = circuit.add(gates.M(0, 2))

``GPI2`` is a native equatorial rotation by :math:`\pi/2`; its ``phi`` parameter
chooses the axis, not the rotation angle. CZ then acts on the connected physical
pair. The measurement requests only logical wires 0 and 2, so the returned sample
array will have two columns, not three. We intentionally use compiler-supported
gates; an arbitrary Qibo circuit would first need transpilation as explained in
:ref:`main_doc_compiler`.

Gate arguments remain logical indices even when ``wire_names`` contains
different physical identifiers. Check the mapping, pair calibration and
compiled timing before using a nontrivial placement on hardware. Changing
wire names does not route otherwise disconnected interactions.

Execute and check the acquired data
-----------------------------------

The explicit backend compiles the circuit and manages the connection for this
execution. A ``finally`` block is useful even here: it demonstrates cleanup for
applications that later replace the dummy platform with hardware, where an
execution failure may leave a connection open.

.. testcode:: circuit-execution

    nshots = 128
    try:
        result = backend.execute_circuit(circuit, nshots=nshots)
    finally:
        platform.disconnect()

    samples = result.samples()
    frequencies = result.frequencies()
    assert set(frequencies).issubset({"00", "01", "10", "11"})
    np.testing.assert_array_equal(measurement.samples(), samples)
    print(samples.shape)
    print(sum(frequencies.values()))
    print(platform.is_connected)

.. testoutput:: circuit-execution

    (128, 2)
    128
    False

These are deterministic checks of the interface. We deliberately do not print
exact frequencies: the dummy platform generates random binary acquisitions
regardless of the preceding gates. A seemingly balanced histogram would not
demonstrate a successful rotation, entanglement or hardware noise.

On a calibrated hardware or emulator platform, a frequency divided by the shot
count estimates the probability of that measured bit string. Dictionary entries
for unobserved strings may be absent, so use ``get`` when extracting a particular
outcome:

.. testcode:: circuit-execution

    observed_11_fraction = frequencies.get("11", 0) / nshots
    assert 0 <= observed_11_fraction <= 1

This result concerns only the measured wires, in their measurement order. It is
not a full state vector, and it says nothing about the unmeasured wire 1.

Prepare and submit several circuits
-----------------------------------

State preparation is itself a circuit. It is prepended to the experiment rather
than passed as a numerical state array. For a collection of experiments,
``initial_states`` accepts one common preparation circuit.

.. testcode:: circuit-execution

    preparation = Circuit(1, wire_names=[0])
    preparation.add(gates.GPI(0, phi=0.0))

    experiments = []
    for phi in (0.0, np.pi / 2):
        experiment = Circuit(1, wire_names=[0])
        experiment.add(gates.GPI2(0, phi=phi))
        experiment.add(gates.M(0))
        experiments.append(experiment)

    try:
        results = backend.execute_circuits(
            experiments, initial_states=preparation, nshots=32
        )
    finally:
        platform.disconnect()

    print(len(results))
    print([outcome.samples().shape for outcome in results])
    assert all(sum(outcome.frequencies().values()) == 32 for outcome in results)

.. testoutput:: circuit-execution

    2
    [(32, 1), (32, 1)]

The backend submits both compiled sequences in one connection lifetime and
returns results in input order. The sweep above changes the axis of a fixed-angle
GPI2 rotation; it is not a sweep of rotation angle. With dummy acquisitions,
the two experiments cannot be used to infer a physical phase response.

Using Qibo's global backend
---------------------------

If the rest of an application executes circuits through ``circuit(...)``, select
Qibolab as the global Qibo backend. This is an alternative to the explicit object
used above, not a required extra step. This example selects NumPy again when
finished; in an application, reselect whichever backend you intend to use next.

.. testcode:: circuit-global-backend

    import qibo
    from qibo import Circuit, gates

    qibo.set_backend("qibolab", platform="dummy")
    selected_backend = qibo.get_backend()
    try:
        circuit = Circuit(1, wire_names=[0])
        circuit.add(gates.M(0))
        outcome = circuit(nshots=16)
        print(outcome.samples().shape)
    finally:
        selected_backend.platform.disconnect()
        qibo.set_backend("numpy")

.. testoutput:: circuit-global-backend

    (16, 1)

An ideal Qibo simulation may be useful as a separate reference experiment, but
its state or probabilities should not be compared to dummy data as a measure of
device fidelity. For meaningful comparisons, use a calibrated platform and
respect its connectivity, native-gate availability and measurement limitations.
