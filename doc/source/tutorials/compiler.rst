.. _tutorials_compiler:

How to modify the default circuit compilation
=============================================

A Qibolab platform can execute pulse sequences.
As shown in :ref:`tutorials_circuits`, Qibo circuits can be executed by invoking the :class:`qibolab.backends.QibolabBackend`, which is the object integrating Qibolab to Qibo.
When a Qibo circuit is executed, the backend will automatically compile it to a pulse sequence, which will be sent to the platform for execution.
The default compiler outlined in the :ref:`main_doc_compiler` section will be used in this process.
In this tutorial we will demonstrate how the user can modify this process for custom applications.

.. note::
    The compiler is only responsible for converting circuits to pulse sequences.
    These circuits should already contain only native gates and respect the chip connectiviy.
    If the circuit does not respect these constraints, it should be converted to an equivalent
    circuit that respects them using a Qibo transpiler.


The ``compiler`` object used when executing a circuit is an attribute of :class:`qibolab.backends.QibolabBackend`.
Creating an instance of the backend provides access to these objects:

.. testcode:: python

    from qibolab.backends import QibolabBackend

    backend = QibolabBackend(platform="dummy")

    print(type(backend.compiler))

.. testoutput:: python
    :hide:

    <class 'qibolab.compilers.compiler.Compiler'>


The default compiler provides rules for the following single-qubit gates:
``Z``, ``RZ``, ``I``, ``GPI``, ``GPI2``, ``M``
and the following two-qubit gates: ``CZ``, ``CNOT`` subject to platform availability.
The user can modify the compilation process by defining custom compilation rules and registering
them to the  ``compiler`` attribute of the ``QibolabBackend``.
The default rules defined in :py:mod:`qibolab.compilers.default` can serve as an example.

The following example shows how to modify the compiler in order to execute a circuit containing an ``X`` gate
directly using a single pi-pulse, instead of converting it to ``GPI2``:

.. testcode:: python

    from qibo import gates
    from qibo.models import Circuit
    from qibolab.backends import QibolabBackend
    from qibolab.sequence import PulseSequence

    # define the circuit
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))


    # define a compiler rule that translates X to the pi-pulse
    def x_rule(gate, natives):
        """X gate applied with a single pi-pulse."""
        return natives.ensure("RX").create_sequence()


    backend = QibolabBackend(platform="dummy")
    # register the new X rule in the compiler
    backend.compiler.rules[gates.X] = x_rule

    # execute the circuit
    result = backend.execute_circuit(circuit, nshots=1000)


.. note::
   If the compiler receives a circuit that contains a gate for which it has no rule, an error will be raised.
   This means that the native gate set that the Qibo transpiler uses, should be compatible with the available compiler rules.
