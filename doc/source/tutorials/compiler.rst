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

    # execute circuit
    result = circuit(nshots=1000)


Defining custom compiler rules
==============================

The compiler can be modified by adding new compilation rules or changing existing ones.
As explained in :ref:`main_doc_compiler` section, a rule is a function that accepts a Qibo gate and a Qibolab platform
and returns the pulse sequence implementing this gate.

The following example shows how to modify the compiler in order to execute a circuit containing a Pauli X gate using a single pi-pulse:

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
    # register the new X rule in the compiler
    backend.compiler[gates.X] = x_rule

    # execute the circuit
    result = backend.execute_circuit(circuit, nshots=1000)

The default set of compiler rules is defined in :py:mod:`qibolab.compilers.default`.

.. note::
   If the compiler receives a circuit that contains a gate for which it has no rule, an error will be raised.
   This means that the native gate set that the transpiler uses, should be compatible with the available compiler rules.
