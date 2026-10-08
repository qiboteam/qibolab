.. _main_doc_backend:

Circuits
========

Qibolab connects Qibo's circuit language to pulse execution on a platform. A
circuit describes operations on logical wires; the platform supplies the physical
qubits, their connectivity and the native pulse sequences.
:class:`qibolab.backend.QibolabBackend` compiles a compatible circuit, executes
its pulses and packages the acquisitions as Qibo measurement outcomes. It is
not a state-vector simulator.

The circuit interface is optional. Install ``qibolab[backend]`` to include Qibo;
the base Qibolab installation remains sufficient for working directly with pulse
sequences. An emulator platform additionally needs ``qibolab[emulator]``. Installing
the emulator extra alone does not install Qibo.

Choosing a backend
------------------

Use Qibo's ``construct_backend`` when you want an explicit backend object, or
``qibo.set_backend("qibolab", platform=...)`` to select the global backend used by
``circuit(nshots=...)``. Both accept a platform name or an existing
:class:`qibolab.Platform`. An explicit object is useful when an application also
uses a Qibo simulation backend, because it does not change Qibo's global selection.

With the backend extra installed, the following example inspects the built-in
dummy platform without executing a circuit:

.. testcode:: backend-overview

    from qibo import construct_backend

    backend = construct_backend("qibolab", platform="dummy")
    print(backend.qubits)
    print(sorted(backend.connectivity))
    print(backend.platform.is_connected)

.. testoutput:: backend-overview

    [0, 1, 2, 3, 4]
    [(0, 2), (1, 2), (2, 3), (2, 4)]
    False

``dummy`` is a built-in platform for testing the execution interface. Its
acquisitions contain synthetic random values, not the evolution of the circuit.
Other platform names are resolved from directories listed in
``QIBOLAB_PLATFORMS``. Neither installing an extra nor selecting the Qibolab backend
provides a calibrated hardware platform automatically. For physical simulation
rather than interface testing, see :ref:`main_doc_emulator`.

Preparing an executable circuit
--------------------------------

The default compiler accepts ``I``, ``Z``, ``RZ``, ``GPI``, ``GPI2``, ``CZ``,
``CNOT``, ``iSWAP``, ``M`` and ``Align``. This is a set of compilation rules, not
a promise that every platform can implement every gate. Rotations require a
single-qubit ``RX90`` or ``RX`` native sequence, measurements require ``MZ``, and
each two-qubit operation needs its corresponding native sequence on the selected
pair. A connected pair may have a CZ calibration but no CNOT calibration, for
example.

``backend.natives`` reports the default rule names, excluding two-qubit gate
types that are absent on every connected pair. It is therefore a useful overview,
but not a per-qubit or per-pair availability check. Inspect the selected
platform's native gates before choosing operations for an experiment.

The backend's compilation step does not route a circuit or decompose arbitrary
gates. If a circuit contains a Hadamard, an arbitrary rotation or interactions
outside the platform connectivity, first use an appropriate Qibo transpilation
pipeline to obtain supported gates and a valid placement. The
:ref:`compiler discussion <main_doc_compiler>` explains this boundary.

Qibo gate arguments always refer to logical wire indices
``0, ..., circuit.nqubits - 1``. ``Circuit(..., wire_names=[...])`` associates
those indices with physical platform identifiers, which may be integers or
strings. For example, ``wire_names=[1, 2]`` associates logical wire 0 with
physical qubit 1; ``gates.CZ(0, 1)`` then selects the native operation on the
physical pair ``(1, 2)``. Connectivity is expressed in physical identifiers,
whereas measurement sample columns follow the circuit's measured logical wires.
Without explicit wire names, Qibo uses consecutive integer indices; Qibolab also
has a positional fallback when those integers are not platform keys.

.. caution::

    Wire names select the native sequences, but the current internal compiler
    retains positional assumptions in parts of its timing bookkeeping. Validate
    the compiled sequence before deploying nontrivial remappings on hardware;
    wire names are not a routing algorithm. The circuit tutorial uses the
    dummy platform's default integer mapping.

Execution and connection lifetime
---------------------------------

Creating or selecting a backend loads the platform but does not connect it.
``backend.execute_circuit(circuit, nshots=...)`` first compiles the circuit, then
calls ``platform.connect()``, executes its pulse sequence and calls
``platform.disconnect()`` before returning. ``execute_circuits`` compiles a
nonempty collection and submits the sequences together within one connection
lifetime, returning one result per input circuit. Whether that submission is
unrolled or otherwise batched depends on the platform.

There is normally no need to connect a platform manually for circuit execution.
A successful backend call also disconnects a platform that was already connected,
so do not rely on it preserving an external connection. The current backend does
not wrap execution in a ``try/finally``: if execution raises after connecting,
the normal disconnect call is skipped. Applications using hardware should arrange
cleanup in their own ``finally`` block.

An optional ``initial_state`` must be a Qibo circuit: its gates are prepended to
the circuit being executed, and must satisfy the same native-gate and mapping
requirements. State vectors and density matrices are rejected. For
``execute_circuits``, ``initial_states`` accepts one preparation circuit applied
to every input circuit, not a list of preparation states.

Understanding the result
------------------------

Execution returns a Qibo ``MeasurementOutcomes`` object, not a quantum state.
``result.samples()`` returns the measured bits, with one row per shot and one
column per measured wire. ``result.frequencies()`` counts the observed bit
strings; an unobserved string need not appear in that dictionary. Measurement
gates also expose their own samples through their registered results.

Include explicit measurement gates to acquire data. The backend cannot provide
an unmeasured state vector, apply gates directly to a supplied state, or infer
ideal probabilities from the pulse program. Shot frequencies estimate the
distribution produced by the selected platform; with ``dummy`` they describe
only random test data. Follow :ref:`tutorials_circuits` for a complete, reproducible
interface check without treating that data as a physical experiment.
