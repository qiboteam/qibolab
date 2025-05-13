.. _main_doc_compiler:

Compiler
========

While pulse sequences can be directly deployed using a platform, circuits need to first be transpiled and compiled to the equivalent pulse sequence.
This procedure typically involves the following steps:

1. The circuit needs to respect the chip topology, that is, two-qubit gates can only target qubits that share a physical connection. To satisfy this constraint SWAP gates may need to be added to rearrange the logical qubits.
2. All gates are transpiled to native gates, which represent the universal set of gates that can be implemented (via pulses) in the chip.
3. Native gates are compiled to a pulse sequence.

The transpiler is responsible for steps 1 and 2, while the compiler for step 3 of the list above.
To be executed in Qibolab, a circuit should be already transpiled. It possible to use the transpilers provided by Qibo to do it. For more information, please refer the `examples in the Qibo documentation <https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-modify-the-transpiler>`_.
On the other hand, the compilation process is taken care of automatically by the :class:`qibolab.QibolabBackend`.

Once a circuit has been compiled, it is converted to a :class:`qibolab.PulseSequence` by the :class:`qibolab._core.compilers.compiler.Compiler`.
This is a container of rules which define how each native gate can be translated to pulses.
A rule is a Python function that accepts a Qibo gate and a platform object and returns the :class:`qibolab.PulseSequence` implementing this gate and a dictionary with potential virtual-Z phases that need to be applied in later pulses.
Examples of rules can be found on :py:mod:`qibolab._core.compilers.default`, which defines the default rules used by Qibolab.

.. note::
   Rules return a :class:`qibolab.PulseSequence` for each gate, instead of a single pulse, because some gates such as the U3 or two-qubit gates, require more than one pulses to be implemented.
