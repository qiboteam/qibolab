.. _main_doc_compiler:

Compiler
========

The Qibolab compiler translates Qibo circuits to pulse sequences.

Overview
--------

The compilation process converts high-level quantum circuits to low-level pulse
sequences that run on hardware:

1. **Transpilation** (Qibo): Reorder qubits to respect chip topology
2. **Compilation** (Qibolab): Convert native gates to pulses
3. **Execution** (Platform): Run pulse sequences on hardware

Qibolab handles step 2 via the :class:`qibolab._core.compilers.compiler.Compiler`.

Public API: QibolabBackend
---------------------------

For most users, use :class:`.QibolabBackend` as a Qibo backend:

.. code-block:: python

    from qibolab.backend import QibolabBackend
    from qibo.models import Circuit

    # Create a Qibo circuit
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    # Execute on Qibolab platform
    backend = QibolabBackend(platform="my_platform")
    result = backend.execute(circuit)

This handles compilation automatically. See :ref:`main_doc_backend` for details.

Compiler Details
----------------

If you need direct compiler access (advanced):

.. code-block:: python

    from qibolab._core.compilers.compiler import Compiler
    from qibo.models import Circuit
    import qibo.gates as gates

    # Create circuit
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 0.5))  # Native gate
    circuit.add(gates.M(0))

    # Compile
    compiler = Compiler.from_platform(platform)
    sequences = compiler(circuit)

    # Execute
    results = platform.execute(sequences)

Compilation Rules
~~~~~~~~~~~~~~~~~

The compiler uses a rule system. Each native gate has a rule defining how to
translate it to pulses:

.. code-block:: python

    from qibolab._core.compilers.default import default_rules

    # Default rules provided by Qibolab
    # Each rule: gate -> PulseSequence

Rules can be custom for each platform. See the compiler module for details.

Integration with Qibo
---------------------

Qibolab is a Qibo backend. The integration:

1. **Qibo transpiler** prepares circuits (layout, optimization)
2. **Qibolab compiler** translates native gates to pulses
3. **Qibolab platform** executes pulse sequences

From Circuits to Hardware
--------------------------

Complete workflow:

.. code-block:: python

    from qibolab.backend import QibolabBackend
    from qibo.models import Circuit
    import qibo.gates as gates

    # 1. Define circuit (Qibo)
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 1))

    # 2. Execute on platform (automatic transpilation + compilation)
    backend = QibolabBackend(platform="my_platform")
    result = backend.execute(circuit, nshots=1000)

    # 3. Get results
    print(result.frequencies())

Advanced: Custom Compiler Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For platforms with non-standard native gates:

.. code-block:: python

    from qibolab._core.compilers.compiler import Compiler


    # Define custom rules
    def custom_rx_rule(gate, platform):
        """Compile RX gate to pulse sequence."""
        # Return PulseSequence implementing RX
        pass


    # Create compiler with custom rules
    rules = {
        "rx": custom_rx_rule,
        # ... other rules
    }
    compiler = Compiler(rules=rules, platform=platform)

Limitations
-----------

- Compiler is internal API; interface may change
- Only native gates can be compiled; others raise errors
- Transpilation must respect chip topology
- Virtual-Z gates require runtime phase tracking

Future
------

The compiler API will be stabilized and exposed as public in future releases.

See Also
--------

- :ref:`main_doc_backend` for circuit execution details
- :ref:`main_doc_experiment` for pulse sequences
- Qibo documentation for circuit definition and transpilation
