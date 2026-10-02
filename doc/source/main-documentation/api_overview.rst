.. _main_doc_api_overview:

Qibolab APIs Overview
=====================

Qibolab provides three main public APIs for quantum computing tasks:

**Platform API** - Define and manage quantum hardware
**Experiment API** - Define and execute quantum experiments
**Driver API** - Implement custom instrument drivers

Quick Navigation
----------------

+---------------------------+-------------------------------------+-------------------------------------+
| Task                      | API                                 | Learn More                          |
+===========================+=====================================+=====================================+
| Build a quantum platform  | :ref:`Platform API <main_doc_platform>` | :ref:`Platform tutorial <tutorial_platform>` |
| Run an experiment         | :ref:`Experiment API <main_doc_experiment>` | :ref:`Experiment tutorial <tutorial_experiment>` |
| Add new hardware          | :ref:`Driver API <main_doc_driver_api>` | :ref:`Driver tutorial <tutorial_driver>` |
| Execute circuits          | :ref:`Backend <main_doc_backend>`   | Use :class:`.QibolabBackend`        |
| Simulate without hardware | :ref:`Emulator <main_doc_emulator>` | :ref:`Emulator tutorial <tutorial_emulator_intro>` |
+---------------------------+-------------------------------------+-------------------------------------+

Platform API
------------

**Purpose**: Define hardware components, qubits, and configurations.

**When to use**:
- Setting up a new quantum platform
- Managing instrument connectivity
- Defining qubits and their control channels
- Persisting configurations

**Key classes**:
- :class:`.Platform` - Central platform object
- :class:`.Hardware` - Instrument topology
- :class:`.Qubit` - Qubit definition with channels
- :class:`.Parameters` - Configurations and native gates

**Start here**: :ref:`main_doc_platform` → :ref:`tutorial_platform`

Experiment API
--------------

**Purpose**: Define quantum experiments using pulse sequences and sweepers.

**When to use**:
- Creating pulse sequences
- Sweeping experimental parameters
- Running measurements
- Analyzing results

**Key classes**:
- :class:`.PulseSequence` - Synchronized pulses across channels
- :class:`.Sweeper` - Parameter sweeping
- :class:`.ExecutionParameters` - Execution options (nshots, averaging, etc.)
- :class:`.Pulse` - Individual pulse with envelope

**Start here**: :ref:`main_doc_experiment` → :ref:`tutorial_experiment`

Driver API
----------

**Purpose**: Implement hardware drivers for new instruments.

**When to use**:
- Adding support for new control electronics
- Customizing instrument behavior
- Implementing real-time sweepers
- Optimizing compilation for specific hardware

**Key classes**:
- :class:`._core.instruments.abstract.Instrument` - Base instrument class
- :class:`._core.instruments.abstract.Controller` - Pulse-generating controller
- :class:`.Channel` - Channel definition
- :class:`.Config` - Configuration class

**Start here**: :ref:`main_doc_driver_api` → :ref:`tutorial_driver`

Integration Points
------------------

The three APIs work together:

1. **Platform defines** instruments, channels, qubits, and parameters
2. **Experiment API uses** platform to access channels and native gates
3. **Driver API implements** how instruments handle experiments

.. code-block:: python

    # 1. Create platform (Platform API)
    platform = create_platform("my_platform")
    platform.connect()

    # 2. Define experiment (Experiment API)
    sequence = platform.natives.single_qubit[0].RX()
    sweeper = Sweeper(parameter=Parameter.frequency, ...)

    # 3. Platform delegates to drivers (Driver API)
    results = platform.execute([sequence], [[sweeper]])

    platform.disconnect()

Learning Path
-------------

**For New Users**:
1. Read :ref:`getting_started_first_experiment`
2. Try :ref:`tutorial_emulator_intro` with the emulator
3. Follow :ref:`tutorial_experiment` for experiment examples
4. Explore :ref:`tutorial_platform` for platform setup

**For Platform Engineers**:
1. Study :ref:`main_doc_platform`
2. Complete :ref:`tutorial_platform`
3. Review existing platforms in the repository
4. Implement custom platforms and drivers

**For Hardware Engineers**:
1. Understand :ref:`main_doc_driver_api`
2. Follow :ref:`tutorial_driver`
3. Review existing drivers in ``src/qibolab/_core/instruments/``
4. Implement drivers for your specific hardware

**For Circuit Execution**:
1. Learn :ref:`main_doc_backend` for circuit execution
2. Use :class:`.QibolabBackend` with Qibo
3. Understand compilation via :ref:`main_doc_compiler`

Key Concepts
------------

**Channels**: Physical communication lines between instruments and QPU.
- Belong to instruments
- Have types (IQ, DC, Acquisition)
- Carry metadata (mixer, LO references)

**Configurations**: Runtime parameters for channels and oscillators.
- Stored in :attr:`.Parameters.configurations`
- Frequency, amplitude, delays, etc.
- Can be temporarily overridden during execution

**Native Gates**: Calibrated pulse sequences for quantum gates.
- Stored in :attr:`.Parameters.native_gates`
- Accessed via :attr:`.Platform.natives`
- Examples: RX, RY, MZ (measurement)

**Pulse Sequences**: Synchronized operations across multiple channels.
- List of (channel, pulse) pairs
- Execute in order, pulses on different channels in parallel
- Can include sweepers for parameter variation

**Sweepers**: Parameter variation during experiment execution.
- More efficient than host-loop iterations
- Can sweep pulse (amplitude, duration, phase) or channel (frequency, offset) parameters
- Support nested and parallel execution

Terminology
-----------

- **Platform**: Complete quantum computing system (instruments, qubits, configs)
- **Instrument**: Physical device (controller, LO, mixer, etc.)
- **Controller**: Instrument that generates pulses and acquires results
- **Channel**: Physical line for signals to/from qubits
- **Qubit**: Logical grouping of channels controlling one physical qubit
- **Pulse**: Single control operation with specific duration, amplitude, phase, envelope
- **Sequence**: Ordered list of pulses across channels
- **Gate**: Native quantum operation (RX, CNOT, MZ) represented as pulse sequence
- **Sweeper**: Runtime parameter variation across multiple values
- **Configuration**: Runtime parameter values (frequency, amplitude, etc.)

Common Patterns
---------------

**Running a Simple Experiment**:

.. code-block:: python

    from qibolab import create_platform

    platform = create_platform("dummy")
    platform.connect()
    results = platform.execute([platform.natives.single_qubit[0].RX()])
    platform.disconnect()

**Sweeping a Parameter**:

.. code-block:: python

    from qibolab import Parameter, Sweeper

    sweeper = Sweeper(
        parameter=Parameter.frequency,
        range=(4.5e9, 4.6e9, 1e6),
        channels=[qubit.drive],
    )
    results = platform.execute([sequence], [[sweeper]])

**Executing a Circuit**:

.. code-block:: python

    from qibolab.backend import QibolabBackend
    from qibo.models import Circuit
    import qibo.gates as gates

    backend = QibolabBackend(platform="my_platform")
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    result = backend.execute(circuit)

Troubleshooting
---------------

**"Channel not found"**: Ensure channel is defined in platform qubits
**"Gate not implemented"**: Check platform has native gate definition
**"Sweeper not supported"**: Hardware may not support this sweeper; check driver
**"Connection failed"**: Verify instrument address and network connectivity

See Also
--------

- :ref:`main_doc_platform` - Platform API reference
- :ref:`main_doc_experiment` - Experiment API reference
- :ref:`main_doc_driver_api` - Driver API reference
- :ref:`main_doc_emulator` - Emulator documentation
- :ref:`main_doc_backend` - Backend and circuit execution
- :ref:`main_doc_compiler` - Circuit compilation details
