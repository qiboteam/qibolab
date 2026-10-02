.. _main_doc_backend:

Backends and Circuit Execution
===============================

Qibolab provides a Qibo-compatible backend for executing quantum circuits on platforms.

Overview
--------

Qibolab integrates with `Qibo <https://qibo.science/>`_ to execute circuits:

1. **Qibo** provides the circuit language and high-level interface
2. **Qibolab** provides the hardware backend for execution
3. **Platform** executes pulse sequences on actual or simulated hardware

The :class:`.QibolabBackend` bridges these components.

Using QibolabBackend
--------------------

Create a backend and execute circuits:

.. code-block:: python

    from qibolab.backend import QibolabBackend
    from qibo.models import Circuit
    import qibo.gates as gates

    # Create backend for your platform
    backend = QibolabBackend(platform="my_platform")

    # Define circuit
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 1))

    # Execute
    result = backend.execute(circuit, nshots=1000)

    # Access results
    print(result.frequencies())
    print(result.counts())

Supported Operations
--------------------

QibolabBackend supports:

- Single-qubit gates: H, X, Y, Z, RX, RY, RZ, S, T, etc.
- Two-qubit gates: CNOT, CZ, etc. (topology-dependent)
- Measurements: M (classical measurement)
- Custom native gates defined in your platform

Execution Options
-----------------

Control execution parameters:

.. code-block:: python

    from qibolab import AcquisitionType, AveragingMode

    result = backend.execute(
        circuit,
        nshots=1000,
        relaxation_time=100,  # Wait between shots (ns)
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

Options are passed to :meth:`.Platform.execute`. See
:ref:`main_doc_experiment` for details.

Circuit Compilation
--------------------

Circuits are automatically compiled through these steps:

1. **Transpilation** (Qibo): Reorder qubits to respect topology
2. **Compilation** (Qibolab): Convert gates to pulses
3. **Execution** (Platform): Run pulses on hardware

This is handled transparently by :class:`.QibolabBackend`.

For advanced compilation control, see :ref:`main_doc_compiler`.

Example: Qubit Spectroscopy
----------------------------

Execute spectroscopy via circuit interface:

.. code-block:: python

    from qibolab.backend import QibolabBackend
    from qibo.models import Circuit
    import qibo.gates as gates
    import numpy as np

    backend = QibolabBackend(platform="my_platform")

    # Run spectroscopy at multiple frequencies
    frequencies = []
    populations = []

    for freq in np.linspace(4.8e9, 5.2e9, 50):
        # Modify platform frequency temporarily
        # (This is an advanced feature; see Platform API)

        circuit = Circuit(1)
        circuit.add(gates.RX(0, np.pi))
        circuit.add(gates.M(0))

        result = backend.execute(circuit, nshots=1000)
        pop = np.mean(result.frequencies()[1])
        frequencies.append(freq)
        populations.append(pop)

    print(f"Resonance at {frequencies[np.argmax(populations)]}")

Integration with Qibo Ecosystem
--------------------------------

You can use all Qibo features:

- **Transpilation**: Use Qibo's transpilers for optimization
- **Plugins**: Combine with Qibo plugin ecosystem
- **Analysis**: Use Qibo's result processing tools

See `Qibo documentation <https://qibo.science/>`_ for details.

Relating Circuits and Pulses
-----------------------------

Understanding the circuit-to-pulse mapping:

+--------------------+-------------------------------+
| Circuit Level      | Pulse Level (Qibolab)         |
+====================+===============================+
| Gate (e.g., RX)    | Pulse sequence                |
| Circuit            | List of pulse sequences       |
| Backend execution  | Platform.execute()            |
+--------------------+-------------------------------+

Example mapping:

.. code-block:: python

    # Circuit level
    circuit = Circuit(1)
    circuit.add(gates.RX(0, np.pi / 2))
    circuit.add(gates.M(0))

    # Pulse level (automatic via backend)
    qubit = platform.qubits[0]
    sequence = platform.natives.single_qubit[0].RX() | platform.natives.single_qubit[0].MZ()
    results = platform.execute([sequence])

    # Results are equivalent
    circuit_result = backend.execute(circuit)
    print(circuit_result.frequencies())

Limitations
-----------

- Only gates compatible with platform's native gates work
- Topology constraints (qubits must be connected for 2-qubit gates)
- Some Qibo features may not be supported by all platforms

See Also
--------

- :ref:`main_doc_experiment` for pulse-level control
- :ref:`main_doc_compiler` for compilation details
- :ref:`main_doc_platform` for platform definition
- `Qibo documentation <https://qibo.science/>`_
