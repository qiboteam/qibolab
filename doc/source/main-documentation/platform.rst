.. _main_doc_platform:

Platform API
============

The :class:`.Platform` is Qibolab's central interface for controlling quantum hardware.
It combines hardware configuration, instrument management, and execution into a unified API.

Overview
--------

A :class:`.Platform` represents your QPU system and consists of:

- **Instruments**: Physical devices (controllers, LOs, etc.) connected to the QPU
- **Qubits**: Logical qubit definitions grouping control channels
- **Parameters**: Hardware configurations and native gate definitions
- **Execution**: Single entry point for running experiments

Basic Usage
-----------

Typical workflow:

1. Load or create a platform
2. Connect to hardware
3. Define experiments using the :ref:`Experiment API <main_doc_experiment>`
4. Execute experiments
5. Disconnect

.. code-block:: python

    from qibolab import create_platform

    platform = create_platform("my_platform")
    platform.connect()

    # Define experiment (see Experiment API)
    sequence = platform.natives.single_qubit[0].RX()

    # Execute
    results = platform.execute([sequence])
    platform.disconnect()

.. hint::

    While Qibolab primarily supports pulse-based experiments, it also integrates with
    Qibo for circuit execution via :class:`.QibolabBackend`. See :ref:`main_doc_backend`.

Hardware Components
-------------------

**Instruments**

Each instrument is a physical device with:

- A unique identifier (name)
- A network address for communication
- Optionally, channels it controls

Two types of instruments exist:

- :class:`._core.instruments.abstract.Controller`: Produces pulses and acquires results
- :class:`._core.instruments.abstract.Instrument`: Passive role (configuration only)

Access instruments via :attr:`.Platform.instruments`.

**Qubits**

Qubits are containers for qubit control channels. A qubit can have:

- :attr:`.Qubit.drive`: XY control
- :attr:`.Qubit.flux`: Z control (frequency tuning)
- :attr:`.Qubit.probe`: Measurement probe (device to QPU)
- :attr:`.Qubit.acquisition`: Measurement acquisition (QPU to device)
- :attr:`.Qubit.drive_extra`: Additional drive lines (higher levels, cross-resonance)

All elements are optional; define only what your platform uses.

Access qubits via :attr:`.Platform.qubits` and couplers via :attr:`.Platform.couplers`.

**Channels**

Channels are physical communication lines. Each channel:

- Belongs to exactly one instrument
- Has a path for routing within the instrument
- References related channels or instruments (e.g., mixer, LO)

Channels are typed: :class:`.IqChannel` for modulated pulses, :class:`.DcChannel` for DC,
:class:`.AcquisitionChannel` for measurements.

.. note::

    Qibolab validates channels during execution. If a pulse sequence references an
    undeclared channel, execution fails. This is intentional: undeclared channels are
    likely mistakes in platform setup, not features.

    If you need a placeholder channel, add it via a dummy instrument.

Parameters
----------

The :class:`.Parameters` object stores two types of information:

**Configurations**

Channel and oscillator configurations hold runtime parameters:

- Frequency, amplitude offset, phase
- Mixer, LO references
- Instrument-specific parameters

Configurations are persistent (saved via :meth:`.Platform.dump`) but can be
temporarily overridden during execution.

Access via :meth:`.Platform.config(channel)`.

**Native Gates**

Native gate definitions are pulse sequences compiled from platform capabilities.
These include pre-defined operations like RX, RY, MZ (measurement).

Access via :attr:`.Platform.natives`.

Architecture
------------

A platform is built from two components:

1. **Hardware** (:class:`.Hardware`): Instrument and channel definitions
2. **Parameters** (:class:`.Parameters`): Configurations and native gates

The full :class:`.Platform` combines both:

.. code-block:: python

    from qibolab._core.platform import Platform, Hardware, Parameters

    hardware = Hardware(instruments=..., qubits=..., couplers=...)
    parameters = Parameters(configurations=..., gates=...)
    platform = Platform(name="my_platform", hardware=hardware, parameters=parameters)

Serialization
--------------

Platforms can be saved and loaded from disk via:

- :meth:`.Platform.dump`: Save parameters to JSON
- :meth:`.Platform.load`: Load parameters from JSON

The default pattern stores platform parameters in ``parameters.json``. This is
optional; see :ref:`main_doc_storage` for details.

.. important::

    Currently, Qibolab assumes a single :class:`._core.instruments.abstract.Controller`
    producing pulses. Other instruments are passive (configuration only).
    This limitation will be lifted in future releases.

Creating Custom Platforms
--------------------------

To create a new platform:

1. Define your hardware (instruments, qubits, channels)
2. Define parameters (configurations, native gates)
3. Create a platform by combining them
4. Optionally, save to disk for reuse

See the :ref:`tutorial_platform` tutorial for a detailed walkthrough.

Driver Implementation
---------------------

Instruments require a driver (subclass of :class:`._core.instruments.abstract.Instrument`)
that handles compilation and hardware communication.

See the :ref:`main_doc_driver_api` section for driver details.
