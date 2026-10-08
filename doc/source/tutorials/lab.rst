.. _tutorial_platform:

Constructing and customizing a platform
=======================================

A laboratory integration supplies instrument objects and declares the channels
that they control. Qibolab combines this hardware description with operating
parameters to form a :class:`qibolab.Platform`. This tutorial starts at that
boundary: it does not require a particular instrument implementation or explain
how to configure one. The runnable examples use ``create_platform("dummy")``
and never communicate with a physical device.

For a real laboratory, obtain the instrument objects and channel declarations
from the integration maintained for your setup. Before connecting, verify that
the qubit-to-channel mapping agrees with the wiring and that the operating
parameters are appropriate for the device. A model that validates successfully
is not necessarily calibrated or safe to execute.

Composing hardware and parameters
---------------------------------

:class:`qibolab.Hardware` groups instrument, qubit, and optional coupler
mappings. It is deliberately independent of the calibration parameters.
Here an existing dummy platform supplies those mappings; in a laboratory the
same step can use a ``Hardware`` object supplied by an external integration.

.. doctest:: lab-composition

    >>> from qibolab import Parameters, Platform, create_platform
    >>> from qibolab.platform import Hardware
    >>> existing = create_platform("dummy")
    >>> hardware = Hardware(
    ...     instruments=existing.instruments,
    ...     qubits=existing.qubits,
    ...     couplers=existing.couplers,
    ... )
    >>> parameters = Parameters.model_validate_json(existing.parameters.model_dump_json())
    >>> platform = Platform(
    ...     name="my_experiment",
    ...     parameters=parameters,
    ...     instruments=hardware.instruments,
    ...     qubits=hardware.qubits,
    ...     couplers=hardware.couplers,
    ... )
    >>> platform.name
    'my_experiment'
    >>> platform.is_connected
    False

The JSON round trip makes an independent parameter model. The instrument objects
are still shared with ``existing``: composing hardware this way does not clone
physical devices or create independent connection ownership. Use only one
platform to manage these shared instruments at a time.

With an integration-provided hardware object, the constructor has exactly the
same form. Replace the copied dummy parameters with a ``Parameters`` model
containing the integration's calibrated configurations and pulse sequences, or
load that model from your own storage. No directory layout or environment
variable is required for direct construction. If the integration instead
supplies separate instrument and qubit mappings, pass them as
``Hardware(instruments=provided_instruments, qubits=provided_qubits)`` and include
``couplers=provided_couplers`` when applicable. These are already constructed
objects from the integration; this composition step does not instantiate
instrument classes.

Checking the mapping
--------------------

The qubit map is the starting point for finding the channels needed by an
experiment. Native operations are keyed by the same physical qubit identifiers.
Inspect both, rather than assuming that the qubit's number is a physical port:

.. doctest:: lab-composition

    >>> qubit_id = 0
    >>> qubit = platform.qubits[qubit_id]
    >>> qubit.drive, qubit.acquisition
    ('0/drive', '0/acquisition')
    >>> all(channel in platform.channels for channel in qubit.channels)
    True
    >>> platform.config(qubit.drive).kind
    'iq'
    >>> platform.natives.single_qubit[qubit_id].MZ is not None
    True

The test above establishes that the identifiers are declared, not that their
physical paths are correctly wired. ``Qubit.default("q0")`` is useful when an
integration adopts names such as ``"q0/drive"``, but it only creates the
identifier container. It neither allocates an output nor adds that output to an
instrument's channels. Optional qubit fields can remain ``None`` when the
corresponding role is not needed.

Component configurations should agree with the channel types and all shared
references. For instance, an IQ channel referring to an oscillator requires an
appropriate configuration under that oscillator's identifier. A component that
is also a separately configurable instrument uses the same identifier in
``configs`` and ``instruments``. The conceptual distinctions and units are
described in :ref:`main_doc_platform`.

Customizing an existing definition
----------------------------------

Start with known parameters when adapting an existing platform.
``Platform.update`` accepts dotted paths and replaces the in-memory parameter
model; it does not alter the hardware mapping or automatically save a file.
For this dummy-only example, choose explicit deterministic values:

.. doctest:: lab-composition

    >>> platform.update(
    ...     {
    ...         "settings.nshots": 4,
    ...         "settings.relaxation_time": 1000,
    ...         f"configs.{qubit.drive}.frequency": 4.1e9,
    ...         "native_gates.single_qubit.0.RX.0.1.duration": 40,
    ...         "native_gates.single_qubit.0.RX.0.1.amplitude": 0.2,
    ...     }
    ... )
    >>> platform.settings.nshots, platform.settings.relaxation_time
    (4, 1000)
    >>> platform.natives.single_qubit[0].RX[0][1].duration
    40.0
    >>> platform.config(qubit.drive).frequency
    4100000000.0

The timing values are in ns, the frequency is in Hz, and the pulse amplitude is
normalized and dimensionless. These are demonstration values, not calibration
recommendations. Updating a native operation changes its stored template;
construct new sequences after the update.

Now make a measurement sequence, connect, execute, and release the connection:

.. doctest:: lab-composition

    >>> sequence = platform.natives.single_qubit[qubit_id].MZ.create_sequence()
    >>> acquisition = sequence.acquisitions[0][1]
    >>> platform.connect()
    >>> try:
    ...     results = platform.execute(
    ...         [sequence],
    ...         updates=[{qubit.drive: {"frequency": 4.2e9}}],
    ...     )
    ... finally:
    ...     platform.disconnect()
    ...
    >>> results[acquisition.id].shape
    (4,)
    >>> platform.config(qubit.drive).frequency
    4100000000.0
    >>> platform.is_connected
    False

The result values are generated by the dummy platform, so only their shape is
shown. The dictionary is keyed by ``acquisition.id``, a UUID, rather than by
qubit number or the sequence's position in the execution batch. A stored JSON
qubit key such as ``"0"`` is also distinct from the integer key ``0`` used in
this example's in-memory qubit and native mappings.

``updates`` here is a **list of configuration overrides**, not the dotted
path dictionary accepted by ``Platform.update``. The override affects this call
without changing the saved parameter model. ``nshots`` and ``relaxation_time``
can also be supplied as execution keywords to override their stored defaults.
To retain a deliberate parameter change across runs, use ``update`` followed by
``dump`` as explained in :ref:`main_doc_storage`.

Starting a new parameter model
------------------------------

When no parameter model exists yet,
:func:`qibolab.platform.initialize_parameters` can generate a structural
starting point from a ``Hardware`` object:

.. doctest:: lab-initialization

    >>> from qibolab import create_platform
    >>> from qibolab.platform import Hardware, initialize_parameters
    >>> existing = create_platform("dummy")
    >>> hardware = Hardware(
    ...     instruments=existing.instruments,
    ...     qubits=existing.qubits,
    ...     couplers=existing.couplers,
    ... )
    >>> initial = initialize_parameters(
    ...     hardware,
    ...     natives={"RX", "MZ", "CZ"},
    ...     pairs=["0-1"],
    ... )
    >>> initial.settings.nshots, initial.settings.relaxation_time
    (1000, 100000)
    >>> initial.configs[hardware.qubits[0].drive].frequency
    0.0
    >>> pulse = initial.native_gates.single_qubit[0].RX[0][1]
    >>> pulse.duration, pulse.amplitude
    (0.0, 0.0)
    >>> list(initial.native_gates.two_qubit)
    [(0, 1)]
    >>> initial.native_gates.two_qubit[(0, 1)].CZ[0][0]
    '0/flux'

This helper does not measure the device, copy the existing platform's
calibration, or discover valid two-qubit interactions. It inspects the hardware's
channel mappings and generates zero-valued DC biases, IQ frequencies,
acquisition timing, and oscillator frequency/power where those channel types
and references are present. Mixer offsets start at zero and IQ corrections
retain their model defaults. Other components or integration-specific
configuration fields may need to be supplied separately.

With ``natives`` omitted, single-qubit native fields remain undefined.
Requested gate names select the supported template fields; they do not
implement or calibrate the operations. Generated pulses have zero duration and
amplitude. ``MZ`` receives a zero-duration readout template, and couplers receive
a ``CP`` template on their flux line. Required roles must already exist: for
example, ``RX12`` needs a ``drive_extra[(1, 2)]`` entry.

Pairs are explicitly supplied as strings such as ``"0-1"``. Their generated
two-qubit templates initially act on the first qubit's drive or flux channel,
depending on the operation. This placeholder is not a determination of the
correct participating qubit or coupler. Replace it with the full calibrated
sequence appropriate to the interaction.

Before using the model on hardware, fill and calibrate its configurations and
native sequences, set meaningful execution defaults, and verify the routing.
``initialize_parameters`` is a convenient schema bootstrap, not a shortcut
around that process.

For an on-disk definition whose ``create()`` returns ``Hardware``,
:func:`qibolab.platform.reset_parameters` performs the corresponding bootstrap
and writes ``parameters.json``. It **overwrites existing parameters**, requires
the current platform-discovery environment, and is not intended as a routine
load operation. See :ref:`main_doc_storage` for that environment and the
difference between loading hardware and loading a full platform.

.. _parameters_json:

Serializing parameters independently
------------------------------------

Parameters are ordinary validated models and can be serialized without a
platform directory. This is useful when an application stores calibrations in
a database or manages versions independently of the hardware factory:

.. doctest:: lab-composition

    >>> payload = platform.parameters.model_dump_json(indent=2)
    >>> restored = Parameters.model_validate_json(payload)
    >>> restored.settings.nshots
    4
    >>> restored.configs[qubit.drive].frequency
    4100000000.0
    >>> restored.native_gates.single_qubit[0].RX[0][1].amplitude
    0.2

The JSON has three top-level sections, ``settings``, ``configs``, and
``native_gates``. Typed configurations and instructions carry a ``kind`` field
used for deserialization. Pair identifiers become hyphen-separated string keys;
sequence entries contain a channel identifier and a serialized instruction.
Prefer generating this representation with ``model_dump_json`` instead of
hand-writing large dictionaries.

If an external integration supplies additional configuration kinds, it must
register them with ``qibolab.ConfigKinds.extend`` before deserializing them.
This registry is process-wide, so changing it during a session affects later
loads. Built-in configuration kinds need no registration.

Loading ``Parameters`` validates the data but does not create hardware objects,
open connections, or verify a physical calibration. Reuse the explicit
``Platform`` constructor from the first example to combine a restored model
with the integration's hardware. Alternatively, adopt the conventional
``platform.py`` and ``parameters.json`` layout in :ref:`main_doc_storage`.
