.. _main_doc_compiler:

Compiler
========

Circuit execution crosses two different abstraction boundaries. Transpilation
chooses a placement of logical qubits, routes interactions onto available
connections and rewrites gates into an executable native set. Compilation then
translates that native circuit into timed pulse sequences using the chosen
platform's calibrated operations. Qibolab performs the latter step automatically
through the :ref:`Qibo backend <main_doc_backend>`; calling the backend directly
does not perform the former.

For a general circuit, prepare an appropriate Qibo transpilation pipeline before
handing the circuit to Qibolab. The
`Qibo transpiler examples
<https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-modify-the-transpiler>`_
describe how to configure that stage. A circuit already written in supported
gates and placed on suitable physical pairs can go straight to compilation.

.. attention::

    The compiler is currently an internal part of ``QibolabBackend``.
    Its rule registration and scheduling interfaces are not a stable public
    customization API. Prefer the backend for circuit execution and the public
    pulse-sequence interface for experiments that need direct pulse control.

From native gates to pulses
---------------------------

The backend creates a default compiler with rules for ``I``, ``Z``, ``RZ``,
``GPI``, ``GPI2``, ``CZ``, ``CNOT``, ``iSWAP``, ``M`` and ``Align``. Gates such
as ``H``, ``RX`` and ``U3`` are not default compilation rules, even though
they are available in Qibo's circuit language. Unsupported gate classes are
not automatically decomposed.

A rule returns a :class:`qibolab.PulseSequence`, not a single pulse and not a
separate dictionary of pending phases. ``I`` contributes no pulses. ``Z`` and
``RZ`` contribute virtual-Z events on the drive channel. ``GPI`` and ``GPI2``
use the platform's single-qubit rotation construction, with angles of
:math:`\pi` and :math:`\pi/2` respectively; their ``phi`` parameter selects the
rotation axis in the equatorial plane. This construction needs an ``RX90`` or
``RX`` native sequence.

Two-qubit rules copy the calibrated sequence for the requested ``CZ``, ``CNOT``
or ``iSWAP`` on the mapped physical pair. Such a sequence may operate on
additional channels or a coupler, not just the gate's two logical wires.
Measurement rules combine the selected qubits' ``MZ`` sequences. ``Align``
contributes delays on the involved qubits' channels when its delay is nonzero.
The existence of a rule is thus separate from the existence of the native data
needed to apply it on a particular platform.

Scheduling and measurement bookkeeping
--------------------------------------

The compiler visits the circuit's moments and translates each gate using its
physical wire names. It tracks channel durations and adds delays so that channels
participating in an operation start together and subsequent operations respect
the occupied qubits and couplers. Independent operations can overlap. The
result is a channel-based pulse sequence, rather than a global list of gates with
one common clock. Unneeded trailing delays are trimmed.

Compilation also returns a measurement map, associating each Qibo measurement
gate with the sequence implementing its readout. After execution, the backend
uses the acquisition identifiers in this map to register the shot arrays with
the correct measurement gates. This is what makes both the circuit-wide result
and individual measurement-gate results accessible through Qibo.

Inspecting compilation without execution
----------------------------------------

The following diagnostic example requires ``qibolab[backend]``. It accesses
the backend's internal compiler deliberately: it illustrates the compilation
boundary, but should not be taken as a stability guarantee for that interface.
No connection or pulse execution is needed to inspect the result.

.. testcode:: compiler-inspection

    from qibo import Circuit, construct_backend, gates
    from qibolab import PulseSequence

    backend = construct_backend("qibolab", platform="dummy")
    circuit = Circuit(1, wire_names=[0])
    circuit.add(gates.GPI2(0, phi=0.0))
    measurement = gates.M(0)
    circuit.add(measurement)

    sequence, measurement_map = backend.compiler.compile(circuit, backend.platform)
    assert isinstance(sequence, PulseSequence)
    assert measurement in measurement_map
    print(len(sequence.acquisitions))
    print(backend.platform.is_connected)

.. testoutput:: compiler-inspection

    1
    False

Successful compilation checks neither the fidelity of a calibration nor the
physical validity of a model. It also cannot repair missing native operations,
an invalid placement or unsupported gates. In particular, the current internal
scheduler retains positional assumptions for some timing bookkeeping under
nontrivial wire remappings, as discussed in :ref:`main_doc_backend`. Inspect the
sequence before using such mappings on hardware. For a first execution using the
default mapping, continue with :ref:`tutorials_circuits`.
