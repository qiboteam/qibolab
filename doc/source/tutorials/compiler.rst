.. _tutorials_compiler:

How to modify the default circuit compilation
=============================================

A Qibolab platform can execute pulse sequences.

As shown in :ref:`tutorials_circuits`, Qibo circuits can be executed by invoking the :class:`qibolab.backends.QibolabBackend`, which is the object integrating Qibolab to Qibo.
When a Qibo circuit is executed, the backend will automatically compile it to a pulse sequence, which will be sent to the platform for execution.
The default compiler outlined in the :ref:`main_doc_compiler` section will be used in this process.
In this tutorial we will demonstrate how the user can modify this process for custom applications.

The ``compiler`` object used when executing a circuit are attributes of :class:`qibolab.backends.QibolabBackend`.
Creating an instance of the backend provides access to these objects:

.. testcode:: python

    from qibolab.backends import QibolabBackend

    backend = QibolabBackend(platform="dummy")

    print(type(backend.compiler))

.. testoutput:: python
    :hide:

    <class 'qibolab.compilers.compiler.Compiler'>

The transpiler is responsible for transforming any circuit to one that respects
the chip connectivity and native gates. The compiler then transforms this circuit
to the equivalent pulse sequence. Note that there is no transpiler in Qibolab, therefore
the backend can only execute circuits that contain native gates by default.
The user can modify the compilation process by changing the  ``compiler`` attributes of
the ``QibolabBackend``.

In this example, we executed circuits using the backend ``backend.execute_circuit`` method,
unlike the previous example (:ref:`tutorials_circuits`) where circuits were executed directly using ``circuit(nshots=1000)``.
It is possible to perform compiler manipulation in both approaches.
When using ``circuit(nshots=1000)``, Qibo is automatically initializing a ``GlobalBackend()`` singleton that is used to execute the circuit.
Therefore the previous manipulations can be done as follows:

.. literalinclude:: ./includes/compiler/compiler0.py

Defining custom compiler rules
==============================

The compiler can be modified by adding new compilation rules or changing existing ones.
As explained in :ref:`main_doc_compiler` section, a rule is a function that accepts a Qibo gate and a Qibolab platform
and returns the pulse sequence implementing this gate.

The following example shows how to modify the compiler in order to execute a circuit containing a Pauli X gate using a single pi-pulse:

.. literalinclude:: ./includes/compiler/compiler1.py

The default set of compiler rules is defined in :py:mod:`qibolab.compilers.default`.

.. note::
   If the compiler receives a circuit that contains a gate for which it has no rule, an error will be raised.
   This means that the native gate set that the transpiler uses, should be compatible with the available compiler rules.
