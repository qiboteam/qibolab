.. _tutorial_emulator_intro:

Getting Started with the Emulator
==================================

The Qibolab emulator simulates quantum systems without real hardware. It's perfect
for testing code, learning, or running quick calibrations.

Why Use the Emulator?
---------------------

- **No hardware needed**: Run experiments on your computer
- **Realistic dynamics**: Simulates quantum system evolution
- **Fast iteration**: Test and refine experiments quickly
- **Same interface**: Use the exact same code as real hardware

Quick Start
-----------

Load the built-in emulated platform:

.. code-block:: python

    from qibolab import create_platform

    # Load emulated platform (uses QuTiP by default)
    platform = create_platform("dummy_emulator")
    platform.connect()

    # Run experiment just like real hardware
    gates = platform.natives.single_qubit[0]
    sequence = gates.RX() | gates.MZ()
    results = platform.execute([sequence], nshots=1000)

    platform.disconnect()

That's it! The emulator automatically handles the quantum simulation.

Understanding Results
---------------------

The emulator returns realistic-looking measurement results:

.. code-block:: python

    import numpy as np

    results = platform.execute([sequence], nshots=1000)
    ro_pulse = list(results.keys())[0]
    counts = results[ro_pulse]  # shape (1000,)

    # Compute excited state probability
    prob_excited = np.mean(counts)
    print(f"P(|1>) = {prob_excited:.3f}")

Results are similar to real hardware:

- **Discrimination mode**: 0 or 1 per shot (realistic quantum noise)
- **Integration mode**: I and Q quadrature values
- **Raw mode**: Full waveform samples

Advanced: Choosing Engines
---------------------------

Choose a simulation engine for speed/accuracy tradeoffs:

.. code-block:: python

    from qibolab._core.instruments.emulator.engine import QutipEngine, DynamiqsEngine

    # Default: QuTiP (CPU, accurate)
    platform = create_platform("dummy_emulator")

    # Alternative: Dynamiqs (GPU-enabled, JAX-based)
    from qibolab.instruments.emulator import EmulatorController

    emulator = EmulatorController(
        address="0.0.0.0",
        channels={...},
        engine=DynamiqsEngine(device="gpu", precision="single"),
    )

See :ref:`main_doc_emulator` for GPU setup details.

Advanced: Custom Emulated Platforms
------------------------------------

Create an emulated platform with your custom Hamiltonian:

.. code-block:: python

    from qibolab._core.instruments.emulator import EmulatorController
    from qibolab._core.platform import Hardware, Platform, Parameters

    # Define emulator controller with custom channels
    emulator = EmulatorController(
        address="0.0.0.0",
        channels={...},  # Your channel definitions
    )

    # Create platform with emulator
    hardware = Hardware(
        instruments={"emulator": emulator},
        qubits={...},
    )

    parameters = Parameters(
        configurations={...},
        # Define Hamiltonian parameters in configs
    )

    platform = Platform(
        name="my_emulator",
        parameters=parameters,
        **vars(hardware),
    )

See :ref:`tutorial_emulator` for complete platform setup.

Performance Tips
----------------

1. **Use small nshots for development**: Start with 100 shots, scale up later
2. **GPU acceleration**: For batched sweeps, GPU acceleration (Dynamiqs) is faster
3. **Cached compilation**: Emulator caches compiled programs automatically
4. **Adjust Nyquist frequency**: For fast experiments, reduce Nyquist to speed up simulation

Known Limitations
-----------------

- **No mid-circuit measurements**: State collapse not implemented
- **No classical feedback**: Can't condition later operations on measurement results
- **Measurement ordering**: All measurements should be at the end of sequences
- **Limited to 2-3 qubits**: Larger systems become computationally expensive

These limitations match typical QPU capabilities anyway.

Testing Your Code
------------------

Use the emulator to test before deploying to real hardware:

.. code-block:: python

    def run_calibration(platform):
        """Calibration routine (works on emulator or real hardware)."""
        qubit = platform.qubits[0]
        gates = platform.natives.single_qubit[0]

        # Rabi oscillation sweep
        sequence = gates.RX() | gates.MZ()
        # ... define sweepers ...
        results = platform.execute([sequence], [[sweepers]])

        return results


    # Test on emulator
    emulator = create_platform("dummy_emulator")
    emulator.connect()
    emulator_results = run_calibration(emulator)
    emulator.disconnect()

    # Same code on real hardware
    hardware = create_platform("my_real_platform")
    hardware.connect()
    hardware_results = run_calibration(hardware)
    hardware.disconnect()

Debugging
---------

The emulator outputs useful debugging info:

.. code-block:: python

    import logging

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    # Run experiment with debug output
    results = platform.execute([sequence])

Check logs for:

- Hamiltonian parameters being used
- Simulation time steps
- Number of shots sampled
- Noise levels applied

Next Steps
----------

- See :ref:`main_doc_emulator` for detailed emulator documentation
- Try :ref:`tutorial_emulator` to build a custom emulated platform
- Use :ref:`tutorial_calibration` with the emulator for fast calibration
- Explore :ref:`tutorial_experiment` for advanced experiment definitions

See Also
--------

- :ref:`main_doc_platform` for platform concepts
- :ref:`main_doc_experiment` for experiment definition
- :ref:`tutorial_platform` for building platforms
