.. admonition:: Work in progress

This documentation is currently in draft form and may be incomplete.

Emulator
========

Qibolab provides an internal simulation instrument that enables the emulation of a variety of quantum chip configurations, including systems with single or multiple qubits, fixed- or tunable-frequency architectures, and optional couplers. This emulator allows users to execute virtual Qibolab experiments in a manner fully consistent with their execution on real quantum processing units (QPUs).

For a more detailed description of how Qibolab handles different QPUs (here referred to as :class:`.Platforms`), the reader is referred to :ref:`Platform guidelines <main_doc_platform>`.

Usage
-----

The emulator leverages a third-party numerical engine that solves the Master Equation for a system governed by the sum of a time-independent Hamiltonian and a time-dependent (pulse) Hamiltonian. The solver produces the system density matrix at each time step. From this solution, the emulator extracts only those density matrices corresponding to acquisition pulses defined within the pulse sequence. The default engine is ``QutipEngine``; ``DynamiqsEngine`` is also available for users who want to run the same emulator workflow through Dynamiqs' JAX-based solvers.

The simulation engine can be selected when defining the emulator instrument in a platform:

.. code-block:: python

    from qibolab._core.instruments.emulator.engine import DynamiqsEngine
    from qibolab.instruments.emulator import EmulatorController

    emulator = EmulatorController(
        address="0.0.0.0",
        channels=channels,
        engine=DynamiqsEngine(device="gpu", precision="single"),
    )

GPU execution is available through the third-party engine APIs, and the engines fail early with an import or device-selection error instead of silently running on CPU when the optional accelerator packages are missing.

For Dynamiqs, set ``DynamiqsEngine(device="gpu")`` and install a CUDA-enabled JAX runtime. On CUDA 12 machines this is a single extra package staying within the JAX range supported by Dynamiqs:

.. code-block:: bash

    pip install qibolab[emulator] "jax[cuda12]"

``DynamiqsEngine`` also exposes ``precision`` (``"single"`` is strongly recommended on consumer GPUs, whose double-precision throughput is severely limited) and ``method``: the default ``"adaptive"`` Tsit5 solver with ``rtol``/``atol`` control, or ``"fixed"``, a Rouchon solver with a constant ``fixed_step_dt`` that guarantees the same finest time resolution targeted by the QuTiP engine.

For QuTiP, set ``QutipEngine(device="gpu")``. The default data layer is `qutip-jax <https://github.com/qutip/qutip-jax>`_ (``pip install qutip-jax``), in which case the evolution runs through the diffrax integrator entirely on the accelerator, preserving the engine's maximum-step bound. The experimental `qutip-cupy <https://github.com/qutip/qutip-cupy>`_ data layer (not released on PyPI) can be selected with ``QutipEngine(device="gpu", gpu_dtype="cupyd")``; there the ODE integration remains on the CPU and only the operator algebra is offloaded.

As a rule of thumb from the benchmarks in ``benchmarks/emulator_gpu.md``: single evolutions of the small bundled platforms are fastest on the QuTiP CPU engine; GPU execution pays off for batched parameter scans (the typical calibration workload) already at moderate system sizes, and for single evolutions of large Hilbert spaces, especially in single precision. The repository includes two benchmark helpers reproducing those tables:

.. code-block:: bash

    # end-to-end emulator benchmark on the bundled platforms
    python benchmarks/emulator_gpu.py --engine dynamiqs --device gpu --precision single
    # solver-level size sweep, batched amplitude scan, and accuracy validation
    python benchmarks/engine_sweep.py single --engine dynamiqs --device gpu --cases 3x3,3x4,3x5
    python benchmarks/engine_sweep.py batched --engine dynamiqs --device gpu --case 3x3
    python benchmarks/engine_sweep.py validate --case 3x2 --device gpu

Since the simulator operates at the level of density matrices, it does not reproduce in-phase and quadrature (I-Q) measurement signals as in real experimental setups. Instead, it computes the measurement probabilities :math:`p_m = \bra{m} \rho \ket{m}` for each computational basis state :math:`\ket{m}`. Consequently, while the emulator supports all experiment types, in signal-based experiments (i.e., when :paramref:`AcquisitionType.INTEGRATION` is selected), the signal magnitude corresponds directly to these probabilities, whereas the signal phase carries no physical meaning.

The emulator supports both :paramref:`AveragingMode.SINGLESHOT` and :paramref:`AveragingMode.CYCLIC`. In the former case, the simulator returns discrete measurement outcomes corresponding to a finite number of shots, whereas in the latter it returns expectation values derived from the density matrix, typically the probability of the :math:`\ket{1}` state. Although SINGLESHOT mode includes statistical sampling noise, it is often computationally advantageous to bypass sampling and directly return the diagonal elements of the density matrix. In CYCLIC mode, Gaussian noise is additionally introduced, with a fixed standard deviation.

To accurately resolve qubit dynamics and capture all contributions from the time-dependent Hamiltonian (including control pulses), a Nyquist frequency is defined. By default, this is set to :math:`f_N = 20 \text{GHz}`, which allows accurate resolution of oscillations up to approximately :math:`10–15 \text{GHz}`. The choice of Nyquist frequency is critical, as it determines the temporal resolution of the simulation and informs adaptive tuning of the ODE solver parameters. Further details on this tuning procedure are provided in the implementation of the numerical engines.

At present, state collapse is not implemented in the Qibolab emulator. As a result, this tool is not suitable for simulating mid-circuit measurements, and should be restricted to circuits in which all measurements occur simultaneously at the end of the computation. In physical systems, mid-circuit measurement induces wavefunction collapse; if the measured qubit is entangled with others, this process introduces correlations that condition the state of the remaining system. The current emulator does not account for such measurement-induced correlations, leading to intrinsically inaccurate results in these scenarios.

An additional limitation arises from the handling of measurement ordering. In certain experimental protocols, the temporal order of measurements may vary across parameter sweeps, while the :paramref:`PulseSequence` object remains fixed and does not reflect such reordering. This discrepancy may introduce inconsistencies during the execution of :func:`qibolab._core.instruments.emulator.results.results`, which assumes alignment between the time-ordering structure and the acquisition pulses defined in the pulse sequence. If this alignment is violated, acquisition events may be incorrectly matched to simulation time steps, ultimately resulting in invalid outputs.


Additionally, Qibolab emulator implements a confusion matrix for simulating mislabeling of the qubits states. At the time being we are assuming an uncorrelated model, in which the total confusion matrix is simply the Kroneker product between single qubits confusion matrix, which indeed does not take into account correlations between qubits due to quantum effects.

This confusion matrix is applied directly to the states probabilities computed from the solution density matrix; then, if we call :math:`M_i` the matrix corresponding to the i-th qubit and :math:`P_{raw}` the output probability vector of the pde solver, we'll have that the 'corrected' probability vector :math:`P_{corr}` is:
.. math::

  P_{corr} = \left( \bigotimes_i M_i \right) P_{raw}