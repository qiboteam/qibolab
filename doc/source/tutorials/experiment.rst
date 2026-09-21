.. _tutorial_experiment:

Defining and Running Experiments
================================

This tutorial covers the Experiment API: pulse sequences, sweepers, and execution parameters.

Basic Experiment
----------------

The simplest experiment is a single pulse sequence:

.. code-block:: python

    from qibolab import create_platform

    platform = create_platform("dummy")
    platform.connect()

    # Get native single-qubit gate
    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]

    # Create experiment: single RX pulse
    sequence = natives.RX()

    # Execute
    results = platform.execute([sequence])
    platform.disconnect()

Adding Measurements
-------------------

Measure your qubit after the pulse:

.. code-block:: python

    # Sequence: RX, then measure
    sequence = natives.RX() | natives.MZ()

    # Execute
    results = platform.execute([sequence])

    # Results are indexed by measurement pulse ID
    for pulse_id, data in results.items():
        print(f"Pulse {pulse_id}: {data}")

Custom Pulses
--------------

Create your own pulses:

.. code-block:: python

    from qibolab import Pulse, PulseSequence, Gaussian, Rectangular

    # Create pulses with different envelopes
    rx_pulse = Pulse(
        duration=40,
        amplitude=0.5,
        relative_phase=0,
        envelope=Rectangular(),
    )

    qubit_drive = qubit.drive
    probe = qubit.probe

    # Build sequence manually
    sequence = PulseSequence(
        [
            (qubit_drive, rx_pulse),
            (probe, measurement_pulse),
        ]
    )

    results = platform.execute([sequence])

Pulse Shapes
~~~~~~~~~~~~

Different envelope shapes:

.. code-block:: python

    from qibolab import Pulse
    from qibolab._core.pulses.envelope import Rectangular, Gaussian, Drag, Exponential, Snz

    # Rectangular: constant amplitude
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())

    # Gaussian: smooth turn-on/off
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Gaussian())

    # DRAG: derivative removal by adiabatic gate
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Drag(beta=1.0))  # DRAG parameter

Sweeping Parameters
-------------------

Sweep a single parameter efficiently:

.. code-block:: python

    from qibolab import Parameter, Sweeper
    import numpy as np

    # Define experiment
    sequence = natives.RX() | natives.MZ()

    # Sweep drive frequency around default
    f0 = platform.config(qubit.drive).frequency
    freq_sweep = Sweeper(
        parameter=Parameter.frequency,
        values=np.linspace(f0 - 100e6, f0 + 100e6, 21),
        channels=[qubit.drive],
    )

    # Execute with sweep
    results = platform.execute(
        [sequence],
        [[freq_sweep]],  # Single sweep
    )

    # Results shape includes sweep dimension
    print(results[pulse_id].shape)  # (21,) for DISCRIMINATION

Sweeping Pulse Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweep pulse parameters (amplitude, duration, phase):

.. code-block:: python

    # Get the RX pulse to sweep
    rx_sequence = natives.RX()
    rx_pulse = rx_sequence[0][1]

    # Sweep amplitude
    amp_sweep = Sweeper(
        parameter=Parameter.amplitude,
        range=(0, 1, 0.1),  # start, stop, step
        pulses=[rx_pulse],
    )

    sequence = rx_sequence | natives.MZ()
    results = platform.execute([sequence], [[amp_sweep]])

Nested Sweepers
~~~~~~~~~~~~~~~

Combine multiple independent sweepers as nested loops:

.. code-block:: python

    # Two sweepers: frequency (outer loop) and amplitude (inner loop)
    freq_sweep = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 100e6, f0 + 100e6, 10e6),
        channels=[qubit.drive],
    )

    rx_pulse = sequence[0][1]
    amp_sweep = Sweeper(
        parameter=Parameter.amplitude,
        range=(0, 1, 0.1),
        pulses=[rx_pulse],
    )

    # Nested sweeps: frequency outer, amplitude inner
    results = platform.execute(
        [sequence],
        [[freq_sweep], [amp_sweep]],
    )

    # Result shape: (nfreq, namp, 2) for INTEGRATION
    print(results[pulse_id].shape)

Parallel Sweepers
~~~~~~~~~~~~~~~~~

Execute sweepers in parallel (zip):

.. code-block:: python

    # Two qubits, sweep both drives in parallel
    q0 = platform.qubits[0]
    q1 = platform.qubits[1]

    q0_sweep = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 50e6, f0 + 50e6, 5e6),
        channels=[q0.drive],
    )

    q1_sweep = Sweeper(
        parameter=Parameter.frequency,
        range=(f1 - 50e6, f1 + 50e6, 5e6),
        channels=[q1.drive],
    )

    # Same list = parallel
    results = platform.execute(
        [sequence],
        [[q0_sweep, q1_sweep]],
    )

Execution Options
-----------------

Control how experiments run:

.. code-block:: python

    from qibolab import AcquisitionType, AveragingMode

    results = platform.execute(
        [sequence],
        nshots=1000,  # Repetitions
        relaxation_time=100,  # Wait between shots (ns)
        fast_reset=False,  # Fast reset (if supported)
        acquisition_type=AcquisitionType.DISCRIMINATION,  # How to acquire
        averaging_mode=AveragingMode.CYCLIC,  # How to average
    )

Acquisition Types
~~~~~~~~~~~~~~~~~

Choose how to acquire data:

.. code-block:: python

    # DISCRIMINATION: Demodulate, integrate, discriminate into states
    # Result: 0 or 1 per shot
    results = platform.execute(
        [sequence],
        nshots=1000,
        acquisition_type=AcquisitionType.DISCRIMINATION,
    )

    # INTEGRATION: Demodulate and integrate
    # Result: I and Q values per shot
    results = platform.execute(
        [sequence],
        nshots=1000,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    # RAW: Unintegrated waveform
    # Result: Full waveform samples per shot
    results = platform.execute(
        [sequence],
        nshots=1000,
        acquisition_type=AcquisitionType.RAW,
    )

Averaging Modes
~~~~~~~~~~~~~~~

Choose how to average:

.. code-block:: python

    # CYCLIC: Best noise resistance (sweeper is outer loop)
    results = platform.execute(
        [sequence],
        [[sweep]],
        averaging_mode=AveragingMode.CYCLIC,
    )

    # SINGLESHOT: No averaging
    results = platform.execute(
        [sequence],
        averaging_mode=AveragingMode.SINGLESHOT,
    )

Multiple Sequences
-------------------

Execute multiple sequences in a single call:

.. code-block:: python

    # Different experiments
    seq1 = natives.RX() | natives.MZ()
    seq2 = natives.RY() | natives.MZ()

    # Execute all
    results = platform.execute([seq1, seq2])

Processing Results
------------------

Results are a dictionary of numpy arrays:

.. code-block:: python

    results = platform.execute([sequence])

    for pulse_id, data in results.items():
        print(f"Pulse {pulse_id}:")
        print(f"  Shape: {data.shape}")
        print(f"  dtype: {data.dtype}")
        print(f"  Data: {data}")

Analyzing Results
~~~~~~~~~~~~~~~~~

Common analysis patterns:

.. code-block:: python

    import numpy as np

    # Single pulse, discrimination, singleshot
    counts = results[pulse_id]  # shape (nshots,)
    prob_excited = np.mean(counts)
    print(f"P(|1>) = {prob_excited:.3f}")

    # With frequency sweep
    probs = results[pulse_id]  # shape (nfreq,)
    max_idx = np.argmax(probs)
    resonance_freq = f0 + sweep_range[max_idx]
    print(f"Resonance at {resonance_freq/1e9:.3f} GHz")

    # Integration with sweep
    iq = results[pulse_id]  # shape (npoints, 2)
    magnitude = np.linalg.norm(iq, axis=1)
    phase = np.arctan2(iq[:, 1], iq[:, 0])

Complete Resonator Spectroscopy Example
----------------------------------------

Full example combining all concepts:

.. code-block:: python

    from qibolab import create_platform, Parameter, Sweeper, AcquisitionType, AveragingMode
    import numpy as np

    platform = create_platform("dummy")
    platform.connect()

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]

    # Readout pulse
    sequence = natives.MZ()

    # Sweep probe frequency
    f_probe = platform.config(qubit.probe).frequency
    probe_sweep = Sweeper(
        parameter=Parameter.frequency,
        range=(f_probe - 200e6, f_probe + 200e6, 1e6),
        channels=[qubit.probe],
    )

    # Execute spectroscopy
    results = platform.execute(
        [sequence],
        [[probe_sweep]],
        nshots=1000,
        relaxation_time=100,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    # Analyze
    ro_pulse_id = list(results.keys())[0]
    signal = results[ro_pulse_id]  # shape (nfreq, 2)
    magnitude = np.linalg.norm(signal, axis=1)
    resonance_idx = np.argmax(magnitude)

    print(f"Resonator resonance at point {resonance_idx}")

    platform.disconnect()

Tips and Best Practices
-----------------------

1. **Use native gates when available**: They're calibrated
2. **Prefer sweepers to loops**: Hardware sweepers are faster
3. **Check result shapes**: Match your expectations
4. **Use CYCLIC averaging**: Better noise resistance
5. **Validate sequences**: Ensure channels are defined
6. **Monitor hardware time**: Use ``estimate_duration()`` to check runtime

Next Steps
----------

- See :ref:`main_doc_experiment` for complete API reference
- Explore :ref:`tutorial_calibration` for gate calibration
- Try :ref:`tutorial_emulator` for simulation without hardware
