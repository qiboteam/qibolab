.. admonition:: Work in progress

Emulator
=========

Qibolab contains its own simulation instrument, using which it is possible to simulate different chips configurations
(one or multiple qubits, fixed or tunable frequency, with or without couplers) and run virtual Qibolab or Qibocal 
experiment in the same fashion as for real QPUs.

For a more detailed discussion on how Qibolab process different QPUs (here called :class:`.Platforms`),
we recommend this page :ref:`Platform guidelines <main_doc_platform>`.


Usage
-----

The emulator exploits a third party engine which solves the Master Equation for the sum of a 
constant Hamiltonian plus the time-dependent Hamiltonian, which is the Pulse Hamiltonian.
From the solver the emulator takes the solution's density matrix of the system for each timestep and
selects the only ones corresponding to the acquisition pulses in the pulse sequence.
Since the initial data are density matrices, the emulator does not simulate I-Q measurement such as 
for a real system, but simply determines the probabilities :math:`p_m=|m><m|` for each state :math:`|m>`.

That's why, even though the emulator can simulate all kind of experiment, when simulate signal experiment
(i.e. with :paramref:`AcquisitionType.INTEGRATION` selected) the signal magnitude is simply the computed probabilities
while the signal phase is simply meaningless.

The emulator simulates both :paramref:`AveragingModeSINGLESHOOT` (i.e. the simulator returns a finite number of shots
corresponding to the measured state) and :paramref:`AveragingMode.CYCLIC` (i.e. the simulator returns the probability of the 
:math:`|1>` state, from the density matrix), but even though `SINGLESHOOT` experiments simulate finite-shots noise
it is more convienient to directly return the diagonal entries of the solution density matrix without sampling.
:paramref:`AveragingMode.CYCLIC` experiments moreover simulate gaussian noise with an hard-coded sigma value.


Additionally, Qibolab emulator implements a confusion matrix for simulating mislabeling of the qubits states.
At the time being we are assuming an uncorrelated model, in which the total confusion matrix is simply the 
Kroneker product between single qubits confusion matrix, which indeed does not take into account correlations
between qubits due to quantum effects.
This confusion matrix is applied directly to the states probabilities computed from the solution density matrix;
then, if we call :math:`M_i` the matrix corresponding to the i-th qubit and :math:`P_{raw}` the output probability vector
of the pde solver, we'll have that the 'corrected' probability vector :math:`P_{corr}` is:
.. math::

  P_{corr} = \left( \bigotimes_i M_i \right) P_{raw}