.. admonition:: Work in progress

This documentation is currently in draft form and may be incomplete.

Emulator
========

Qibolab provides an internal simulation instrument that enables the emulation of a variety of quantum chip configurations, including systems with single or multiple qubits, fixed- or tunable-frequency architectures, and optional couplers. This emulator allows users to execute virtual Qibolab experiments in a manner fully consistent with their execution on real quantum processing units (QPUs).

For a more detailed description of how Qibolab handles different QPUs (here referred to as :class:`.Platforms`), the reader is referred to :ref:Platform guidelines <main_doc_platform>.

Usage
-----

The emulator relies on a third-party numerical engine that solves the Master Equation for a system governed by the sum of a time-independent Hamiltonian and a time-dependent (pulse) Hamiltonian. The solver produces the system density matrix at each time step. From this solution, the emulator extracts only those density matrices corresponding to acquisition pulses defined within the pulse sequence.

Since the simulator operates at the level of density matrices, it does not reproduce in-phase and quadrature (I-Q) measurement signals as in real experimental setups. Instead, it computes the measurement probabilities :math:`p_m = \langle m | \rho | m \rangle` for each computational basis state :math:`|m\rangle`. Consequently, while the emulator supports all experiment types, in signal-based experiments (i.e., when :paramref:`AcquisitionType.INTEGRATION` is selected), the signal magnitude corresponds directly to these probabilities, whereas the signal phase carries no physical meaning.

The emulator supports both :paramref:`AveragingMode.SINGLESHOT` and :paramref:`AveragingMode.CYCLIC`. In the former case, the simulator returns discrete measurement outcomes corresponding to a finite number of shots, whereas in the latter it returns expectation values derived from the density matrix, typically the probability of the :math:`|1\rangle` state. Although SINGLESHOT mode includes statistical sampling noise, it is often computationally advantageous to bypass sampling and directly return the diagonal elements of the density matrix. In CYCLIC mode, Gaussian noise is additionally introduced, with a fixed standard deviation.

To accurately resolve qubit dynamics and capture all contributions from the time-dependent Hamiltonian (including control pulses), a Nyquist frequency is defined. By default, this is set to :math:`f_N = 20 \text{GHz}`, which allows accurate resolution of oscillations up to approximately :math:`10–15 \text{GHz}`. The choice of Nyquist frequency is critical, as it determines the temporal resolution of the simulation and informs adaptive tuning of the ODE solver parameters. Further details on this tuning procedure are provided in the implementation of the numerical engines.

At present, state collapse is not implemented in the Qibolab emulator. As a result, this tool is not suitable for simulating mid-circuit measurements, and should be restricted to circuits in which all measurements occur simultaneously at the end of the computation. In physical systems, mid-circuit measurement induces wavefunction collapse; if the measured qubit is entangled with others, this process introduces correlations that condition the state of the remaining system. The current emulator does not account for such measurement-induced correlations, leading to intrinsically inaccurate results in these scenarios.

An additional limitation arises from the handling of measurement ordering. In certain experimental protocols, the temporal order of measurements may vary across parameter sweeps, while the :paramref:`PulseSequence` object remains fixed and does not reflect such reordering. This discrepancy may introduce inconsistencies during the execution of :func:`qibolab._core.instruments.emulator.results.results`, which assumes alignment between the time-ordering structure and the acquisition pulses defined in the pulse sequence. If this alignment is violated, acquisition events may be incorrectly matched to simulation time steps, ultimately resulting in invalid outputs.
