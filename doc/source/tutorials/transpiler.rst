.. _tutorials_transpiler:

How to modify the transpiler?
=============================

A Qibolab platform can execute pulse sequences.
As shown in :ref:`tutorials_circuits`, Qibo circuits can be executed by invoking the :class:`qibolab.backends.QibolabBackend`, which is the object integrating Qibolab to Qibo.
When a Qibo circuit is executed, the backend will automatically transpile and compile it to a pulse sequence, which will be sent to the platform for execution.
The default transpilers and compiler outlined in the :ref:`main_doc_transpiler` section will be used in this process.
In this tutorial we will demonstrate how the user can modify this process for custom applications.

The ``transpiler`` and ``compiler`` objects used when executing a circuit are attributes of :class:`qibolab.backends.QibolabBackend`.
Creating an instance of the backend provides access to these objects:

.. testcode:: python

    from qibolab.backends import QibolabBackend

    backend = QibolabBackend(platform="dummy")

    print(type(backend.transpiler))
    print(type(backend.compiler))

.. testoutput:: python
    :hide:

    <class 'qibo.transpiler.pipeline.Passes'>
    <class 'qibolab.compilers.compiler.Compiler'>

The transpiler is responsible for transforming the circuit to respect the chip connectivity and native gates,
while the compiler transforms the circuit to the equivalent pulse sequence.
The user can modify these attributes before executing a circuit.
For example:

.. testcode:: python

    from qibo import gates
    from qibo.models import Circuit
    from qibolab.backends import QibolabBackend

    # define circuit
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.M(0))

    backend = QibolabBackend(platform="dummy")
    # disable the transpiler
    backend.transpiler = None

    # execute circuit
    result = backend.execute_circuit(circuit, nshots=1000)

completely disables the transpilation steps. In this case the circuit will be sent directly to the compiler, to be compiled to a pulse sequence.
Instead of completely disabling, custom transpilation steps can be given:

.. testcode:: python

    from qibolab.backends import QibolabBackend

    from qibo.transpiler.unroller import NativeGates, Unroller

    backend = QibolabBackend(platform="dummy")
    backend.transpiler = Unroller(native_gates=NativeGates.CZ)


Now circuits will only be transpiled to native gates, without any connectivity matching steps.
The :class:`qibolab.transpilers.gate_decompositions.NativeGates` transpiler used in this example assumes Z, RZ, GPI2 or U3 as the single-qubit native gates, and supports CZ and iSWAP as two-qubit natives.
In this case we restricted the two-qubit gate set to CZ only.
If the circuit to be executed contains gates that are not included in this gate set, they will be transformed to multiple gates from the gate set.
Arbitrary single-qubit gates are typically transformed to U3.
Arbitrary two-qubit gates are transformed to two or three CZ gates following their `universal CNOT decomposition <https://arxiv.org/abs/quant-ph/0307177>`_.
The decomposition of some common gates such as the SWAP and CNOT is hard-coded for efficiency.

Multiple transpilation steps can be implemented using the :class:`qibolab.transpilers.pipeline.Pipeline`:

.. testcode:: python

    from qibo.transpiler import Passes
    from qibo.transpiler.unroller import NativeGates, Unroller
    from qibo.transpiler.star_connectivity import StarConnectivity

    backend = QibolabBackend(platform="dummy")
    backend.transpiler = Passes(
        [
            StarConnectivity(middle_qubit=2),
            Unroller(native_gates=NativeGates.CZ),
        ]
    )

In this case circuits will first be transpiled to respect the 5-qubit star connectivity, with qubit 2 as the middle qubit. This will potentially add some SWAP gates. Then all gates will be converted to native.

The compiler can be modified similarly, by adding new compilation rules or modifying existing ones.
As explained in :ref:`main_doc_transpiler` section, a rule is a function that accepts a Qibo gate and a Qibolab platform and returns the corresponding pulse sequence implementing this gate.

The following example shows how to modify the transpiler and compiler in order to execute a circuit containing a Pauli X gate using a single pi-pulse:

.. testcode:: python

    from qibo import gates
    from qibo.models import Circuit
    from qibolab.backends import QibolabBackend
    from qibolab.pulses import PulseSequence

    # define the circuit
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))


    # define a compiler rule that translates X to the pi-pulse
    def x_rule(gate, platform):
        """X gate applied with a single pi-pulse."""
        qubit = gate.target_qubits[0]
        sequence = PulseSequence()
        sequence.add(platform.create_RX_pulse(qubit, start=0))
        return sequence, {}


    # the empty dictionary is needed because the X gate does not require any virtual Z-phases

    backend = QibolabBackend(platform="dummy")
    # disable the transpiler (the default transpiler will attempt to convert X to U3)
    backend.transpiler = None
    # register the new X rule in the compiler
    backend.compiler[gates.X] = x_rule

    # execute the circuit
    result = backend.execute_circuit(circuit, nshots=1000)

Here we completely disabled the transpiler to avoid transforming the X gate to a different gate and we added a rule that instructs the compiler how to transform the X gate.

The default set of compiler rules is defined in :py:mod:`qibolab.compilers.default`.

.. note::
   If the compiler receives a circuit that contains a gate for which it has no rule, an error will be raised.
   This means that the native gate set that the transpiler uses, should be compatible with the available compiler rules.
   If the transpiler is disabled, a rule should be available for all gates in the original circuit.

In the above examples we executed circuits using the backend ``backend.execute_circuit`` method,
unlike the previous example (:ref:`tutorials_circuits`) where circuits were executed directly using ``circuit(nshots=1000)``.
It is possible to perform transpiler and compiler manipulation in both approaches.
When using ``circuit(nshots=1000)``, Qibo is automatically initializing a ``GlobalBackend()`` singleton that is used to execute the circuit.
Therefore the previous manipulations can be done as follows:

.. testcode:: python

    import qibo
    from qibo import gates
    from qibo.models import Circuit
    from qibo.backends import GlobalBackend

    # define circuit
    circuit = Circuit(1)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.M(0))

    # set backend to qibolab
    qibo.set_backend("qibolab", platform="dummy")
    # disable the transpiler
    GlobalBackend().transpiler = None

    # execute circuit
    result = circuit(nshots=1000)
