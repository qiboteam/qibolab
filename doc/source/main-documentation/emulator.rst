.. _main_doc_emulator:

Emulation
=========

An emulator platform runs Qibolab pulse experiments against a physical model
instead of a laboratory QPU. The experiment still uses a
:class:`qibolab.Platform`, native gates, pulse sequences, execution options and
acquisition identifiers. What changes is the source of the acquisitions:
they are obtained from simulated quantum dynamics. This makes emulation useful
for studying how pulse choices and model parameters affect an experiment before
deploying it on hardware.

Emulation is different from both the built-in ``dummy`` platform and an ideal
Qibo circuit simulator. ``dummy`` produces random test data without evolving a
quantum system. A Qibo state-vector backend applies mathematical gate operators.
A Qibolab emulator instead evolves the platform's physical model under the
actual pulse program, including the dissipation represented in that model.
Agreement with ideal gates depends on the model and the native pulse
calibrations; it is not guaranteed by selecting an emulator.

Installation and platform selection
-----------------------------------

Install the optional numerical dependencies with:

.. code-block:: console

    pip install "qibolab[emulator]"

The extra installs QuTiP and Dynamiqs. It does not install Qibo, supply a
calibrated platform, or turn an existing dummy platform into a simulator. To
execute Qibo circuits as well, install both extras:

.. code-block:: console

    pip install "qibolab[backend,emulator]"

The platform-loading rules are unchanged. ``create_platform("dummy")`` is the
only built-in special name. ``"emulator"`` is not a built-in platform name:
it works only if a platform directory with that name exists in a search path
listed in ``QIBOLAB_PLATFORMS``. Load the name of an emulator platform provided
by your platform repository, or pass an existing platform object to the backend.

For example, if your platform repository contains a configured numerical
platform in a directory named ``my_emulated_platform``, add its parent directory
to ``QIBOLAB_PLATFORMS`` and load it with
``create_platform("my_emulated_platform")``. This name is illustrative, not
built in. The :ref:`tutorials_emulator` tutorial follows this workflow without
constructing or configuring a simulation driver.

Execution and interpretation
----------------------------

Pulse-level experiments use the same connection lifecycle as other platforms:
connect, execute and disconnect, with cleanup in a ``finally`` block.
The emulator does not require a connection to laboratory electronics.
For circuits, the Qibolab backend manages this lifecycle as described in
:ref:`main_doc_backend`, and the same native-gate and connectivity requirements
apply.

The normal return value is still a dictionary of arrays keyed by acquisition
identifiers, not a quantum state or the full simulated evolution.
``AcquisitionType.DISCRIMINATION`` with ``AveragingMode.SINGLESHOT`` requests
classified shots, so their frequencies can estimate the modeled measurement
distribution. Simultaneous final acquisitions allow the modeled correlations
to be represented in those shots if the numerical platform samples the joint
distribution. For multiqubit studies, confirm that the supplied model preserves
these correlations rather than independently sampling each qubit's marginal
distribution. Cyclic averaging instead requests an averaged result without a
shot axis. A numerical platform may provide populations directly rather than
estimate them by repeatedly sampling.

``AcquisitionType.INTEGRATION`` requests an I/Q-shaped representation, but that
interface alone does not imply a physical readout-chain simulation. Numerical
platforms may return population proxies or synthetic signals rather than
laboratory voltages. Establish the meaning of these values for the supplied
platform before interpreting their magnitude or phase. Similarly, binary
classification does not by itself resolve leakage into higher levels of a
multilevel model. The tutorial concentrates on classified shots, which have a
clear interpretation as measurement outcomes, rather than raw digitizer traces.

With no sweeps, a single-shot discrimination acquisition has shape
``(nshots,)``, while single-shot integration has shape ``(nshots, 2)``.
Cyclic discrimination returns a scalar array and cyclic integration returns
an array of shape ``(2,)``. Parameter sweeps add their sweep axes to these
results according to the usual execution-options convention. These are interface
conventions; check which acquisition and averaging modes your supplied numerical
platform supports.

What the interface does not guarantee
-------------------------------------

The common execution interface does not specify a numerical model's treatment of
measurement-induced state collapse, feedback, or reset. Before studying those
protocols, establish that the supplied numerical platform implements the required
measurement-conditioned dynamics. Merely returning a sampled acquisition does
not establish that the state was collapsed before subsequent pulses. The
tutorial avoids that assumption by using final measurements only.

Similarly, verify support for the timing and parameters of a proposed sweep.
A model that can produce correlated samples for simultaneous final measurements
need not support acquisitions at different times, or sweeps that change those
times. Begin with a fixed acquisition schedule and extend the experiment only
within the supplied platform's capabilities.

Simulating a density matrix becomes expensive as the number of modeled
subsystems and levels increases. Pulse durations and numerical resolution also
affect runtime. Start with a small platform and a short sequence, and interpret
results in the context of the supplied model rather than as predictions of
unspecified hardware. Sampled readouts contain statistical noise, and a numerical
platform may also add synthetic noise to its signals; exact counts are not
expected to be reproducible physical predictions.
