.. _main_doc_experiment:

Experiment API
==============

The Experiment API enables definition and execution of quantum experiments.
It consists of pulse sequences, sweepers, and execution parameters—everything that
enters :meth:`.Platform.execute`.

Overview
--------

An experiment in Qibolab is defined by:

1. **Pulse Sequences** (:class:`.PulseSequence`): Synchronized pulses across channels
2. **Sweepers** (:class:`.Sweeper`): Parameter sweeps (optional)
3. **Execution Parameters** (:class:`.ExecutionParameters`): Options like nshots, averaging

Together, these form a complete experiment specification.

Pulse Sequences
---------------

A :class:`.PulseSequence` is a list of (channel, pulse) pairs representing operations
to execute in order. Pulses on different channels execute in parallel.

**Creating Sequences**

.. code-block:: python

    from qibolab import Pulse, PulseSequence, Rectangular

    # Create individual pulses
    pulse1 = Pulse(
        duration=40,  # nanoseconds
        amplitude=0.5,  # normalized [-1, 1]
        relative_phase=0,  # radians
        envelope=Rectangular(),
    )

    # Create sequence
    sequence = PulseSequence(
        [
            (qubit.drive, pulse1),
            (qubit.probe, measurement_pulse),
        ]
    )

**Using Native Gates**

Platforms provide pre-defined gates (RX, RY, MZ) accessible via :attr:`.Platform.natives`:

.. code-block:: python

    natives = platform.natives.single_qubit[0]
    rx_pulse = natives.RX()
    sequence = rx_pulse | natives.MZ()  # Concatenate with |

**Pulse Shapes**

Qibolab supports several pulse envelope shapes:

- :class:`.Rectangular`: Constant amplitude
- :class:`.Gaussian`: Gaussian envelope
- :class:`.Drag`: DRAG (derivative removal by adiabatic gate)
- :class:`.Exponential`: Exponential decay
- :class:`.Snz`: Second-order nested SWAP gate
- :class:`.Custom`: User-defined waveform

See :py:mod:`qibolab._core.pulses.envelope` for details.

**Sequence Operations**

Sequences support several operations:

.. code-block:: python

    seq1 = ...
    seq2 = ...

    # Concatenate: execute seq1 then seq2
    combined = seq1 | seq2

    # Duration of sequence
    duration = seq1.duration

    # Channel duration
    drive_duration = seq1.channel_duration(qubit.drive)

Sweepers
--------

:class:`.Sweeper` objects enable efficient parameter sweeps on hardware without
requiring multiple round-trips to the host.

**Sweepable Parameters**

Sweepers can modify:

- **Pulse parameters**: amplitude, duration, relative_phase
- **Channel parameters**: frequency, offset

**Creating Sweepers**

Specify sweeper using either ``values`` (explicit array) or ``range`` (start, stop, step):

.. code-block:: python

    from qibolab import Parameter, Sweeper
    import numpy as np

    # Sweep frequency on a channel
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([4e9, 4.01e9, 4.02e9]),  # or
        # range=(4e9, 4.1e9, 0.01e9),
        channels=[qubit.drive],
    )

    # Sweep amplitude on a pulse
    pulse = sequence[0][1]
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(0, 1, 0.1),
        pulses=[pulse],
    )

**Multiple Sweepers**

Multiple sweepers can be combined:

- Sweepers in the **same list** execute in parallel (zip style)
- Sweepers in **separate lists** form nested loops (outer to inner)

.. code-block:: python

    # Parallel sweeps
    results = platform.execute(
        [sequence], [[sweeper1, sweeper2]], **options  # sweeper1 and sweeper2 in parallel
    )

    # Nested sweeps
    results = platform.execute(
        [sequence],
        [[sweeper1], [sweeper2]],  # sweeper1 (outer), sweeper2 (inner)
        **options
    )

.. warning::

    Hardware support for sweepers varies. Unsupported sweepers either raise an error
    or are automatically unrolled as separate sequences.

Execution Parameters
--------------------

:class:`.ExecutionParameters` specifies how to run experiments via keyword arguments
to :meth:`.Platform.execute`:

.. code-block:: python

    results = platform.execute(
        sequences,
        sweepers,
        nshots=1000,
        relaxation_time=100,  # nanoseconds between shots
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

**Key Parameters**

- ``nshots`` (int): Number of repetitions
- ``relaxation_time`` (int): Wait between shots (ns)
- ``fast_reset`` (bool): Enable fast reset if supported
- ``acquisition_type`` (:class:`.AcquisitionType`): How to acquire data
- ``averaging_mode`` (:class:`.AveragingMode`): How to average results

**Acquisition Types**

- :attr:`.AcquisitionType.DISCRIMINATION`: Demodulate, integrate, and discriminate states
- :attr:`.AcquisitionType.INTEGRATION`: Demodulate and integrate
- :attr:`.AcquisitionType.RAW`: Acquire raw waveform (unintegrated)

**Averaging Modes**

- :attr:`.AveragingMode.CYCLIC`: Best noise resistance; sweeper is outer loop
- :attr:`.AveragingMode.SINGLESHOT`: No averaging
- :attr:`.AveragingMode.SEQUENTIAL`: Worse noise; sweeper is inner loop (avoid)

Results
-------

:meth:`.Platform.execute` returns a dictionary mapping acquisition pulse IDs to results.
Result shape depends on nshots, acquisition type, averaging mode, and sweepers.

**Result Shapes**

Without sweepers:

- Discrimination, Cyclic: Shape ``()`` (single value)
- Discrimination, Singleshot: Shape ``(nshots,)`` with values 0 or 1
- Integration, Cyclic: Shape ``(2,)`` (I and Q averaged)
- Integration, Singleshot: Shape ``(nshots, 2)`` (I and Q per shot)
- Raw, Singleshot: Shape ``(nshots, samples, 2)``

With sweepers:

- Result shape includes sweep dimensions
- Sweeper order follows nesting/parallelization

Example with two sweepers:

.. code-block:: python

    # sweeper1: 100 values, sweeper2: 50 values
    # INTEGRATION + SINGLESHOT
    result_shape = (nshots, len(sweeper1), len(sweeper2), 2)  # I and Q

Full Example
------------

Complete experiment combining all components:

.. code-block:: python

    from qibolab import create_platform, Parameter, Sweeper, AcquisitionType, AveragingMode

    platform = create_platform("my_platform")
    platform.connect()

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]

    # Define experiment
    rx_sequence = natives.RX()
    sequence = rx_sequence | natives.MZ()

    # Define sweepers
    f0 = platform.config(qubit.drive).frequency
    freq_sweeper = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 100e6, f0 + 100e6, 1e6),
        channels=[qubit.drive],
    )

    # Execute
    results = platform.execute(
        [sequence],
        [[freq_sweeper]],
        nshots=1000,
        relaxation_time=100,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    platform.disconnect()

    # Process results
    print(results[ro_pulse_id].shape)  # (frequency_points,)
