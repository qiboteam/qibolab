.. admonition:: Work in progress

Emulator
=========

Qibolab contains its own simulation instrument, using which it is possible to simulate different chips configurations (one or multiple qubits, fixed or tunable frequency, with or without couplers) and run virtual Qibolab or Qibocal experiment in the same fashion as for real QPUs.

For a more detailed discussion on how Qibolab process different QPUs (here called :class:`.Platforms`), we recommend this page :ref:`Platform guidelines <main_doc_platform>`.


Usage
-----

The emulator exploits a third party engine which solves the Master Equation for the sum of a constant Hamiltonian plus the time-dependent Hamiltonian, which is the Pulse Hamiltonian. From the solver, the emulator takes the solution's density matrix of the system for each timestep and selects the only ones corresponding to the acquisition pulses in the pulse sequence. Since the initial data are density matrices, the emulator does not simulate I-Q measurement such as for a real system, but simply determines the probabilities :math:`p_m=|m><m|` for each state :math:`|m>`.

That's why, even though the emulator can simulate all kind of experiment, when simulate signal experiment (i.e. with :paramref:`AcquisitionType.INTEGRATION` selected) the signal magnitude is simply the computed probabilities while the signal phase is simply meaningless.

The emulator simulates both :paramref:`AveragingMode.SINGLESHOT` (i.e. the simulator returns a finite number of shots corresponding to the measured state) and :paramref:`AveragingMode.CYCLIC` (i.e. the simulator returns the probability of the :math:`|1>` state, from the density matrix), but even though `SINGLESHOT` experiments simulate finite-shots noise it is more convenient to directly return the diagonal entries of the solution density matrix without sampling. :paramref:`AveragingMode.CYCLIC` experiments moreover simulate gaussian noise with an hard-coded sigma value.

In order to resolve every qubit oscillations and not miss any contribution (i.e. pulse) in the total Hamiltonian, we define the NYQUIST frequency to use, which by default is set to :math:`f_N = 20 GHz`, which will allow us to correctly resolve oscillation at most of :math:`10-15 GHz`. Setting the NYQUIST frequency is a crucial part of a correct integration since from that we can adaptively tune specific options of the ODE solver and correctly solve the state evolution. For better understanding of this tuning process please see engines' implementations.

At the time being state collapse is not implemented in QiboLab Emulator, hence this version should not be used for simulating mid-circuit measurement, but only for circuits with synchronous measurement for all qubits at the end of the circuit. After a mid-circuit measurement, the measured qubit collapses to an eigenstate; if it is entangled with other qubits, this collapse induces correlations that condition the state of the remaining system. In the present implementation, the emulator does not account for these measurement-induced correlations and therefore produces intrinsically inaccurate results in such cases. Another limitation that prevents the use of this emulator for mid-circuit measurements is that, in certain experiments, the time ordering of measurements may vary across parameter sweeps, while the :paramref:`PulseSequence`` object (i.e., the pulse sequence being simulated) remains unchanged and therefore does not capture such reordering. This discrepancy can lead to inconsistencies when executing :func:`qibolab._core.instruments.emulator.results.results`, which iterates simultaneously over both the time-ordering array and the acquisition pulses defined in the :paramref:`PulseSequence`. Correct behavior relies on these two iterables remaining aligned; if either is reordered during the sweep, acquisition pulses may be associated with incorrect simulation timesteps, ultimately producing invalid results.
