.. _main_doc_platform:

Platforms
=========

One of the core goals of Qibolab is to allow the execution of the experiments defined
with its own :ref:`Experiment API <main_doc_experiment>` on diverse platforms.

In order to do this, the main abstraction introduced is exactly the
:class:`qibolab.Platform` itself, which is intended to represent a collection of
instruments, suitably connected to the device operated as a QPU.

The handling of the instruments it will be mainly internal to the Qibolab itself, and it
is defined by the instruments' drivers.

Usage
-----

The whole workflow is supposed to be the following:

#. the experiment is defined by the user using the mentioned :ref:`Experiment API
   <main_doc_experiment>`, together with an optional set of temporary configuration
   updates
#. the execution is invoked through :meth:`qibolab.Platform.execute`, which acts as the
   single entry point
#. internally, the configurations and experiment definition are shared with all the
   registered instruments, iterating over those registered in the platform, and
   converting the instructions to the each instrument's representation (which is the main
   role of the driver), finally uploading them
#. the experiment is then triggerred
#. upon completion, results are downloaded from the relevant sources, and collected into
   a single collection, which is returned as the output of :meth:`qibolab.Platform.execute`

.. note::

    Because of internal Qibolab's limitations, currently it is assumed that there is
    just a single instrument capable of producing pulses (a
    :class:`qibolab._core.instruments.abstract.Controller` instance). While all the
    other instruments will play a passive role (e.g. LOs), which boils down in only
    supporting configurations, but do not execute any synchronized operation.

    This limitation will be lifted in future releases.

    However, it is mostly affecting the way instruments' drivers are written, since a
    single driver should span all the active components. Once this is done, it only
    limits the composability of existing instruments.

The only other active operations which are relevant for the experimental workflow
consist in instruments' initialization and close up, which are performed through
:meth:`qibolab.Platform.connect` and :meth:`qibolab.Platform.disconnect` methods.

.. hint::

    The platform just exposes the API for pulse-based experiments. However, it is always
    central for hardware execution.

    Indeed, Qibolab exposes a Qibo compatible backend for circuits execution,
    :class:`qibolab.QibolabBackend`. Which at its heart, it is powered by a platform
    itself.

    Cf. :ref:`main_doc_backend` and :ref:`main_doc_compiler`.

However, there is a further relevant role which is performed by the
:class:`qibolab.Platform`: parameters' persistance.

Parameters
^^^^^^^^^^

The role of the :class:`qibolab.Platform` is to store all the information required to
execution.
While what exactly means execution is possibly debatable, since it is strictly related
to the purpose the QPU is being used for, to have a clear target we intend as executing
circuits.
Most (but not necessarily all) of the information required to perform different
experiments will be anyhow contained in this.

Specifically, the major ingredient for circuits' execution is the definition of a set of
native gates as low-level operations that can be achieved by the instruments. In
practice, each gate is represented by a "pulse" sequence.

.. note::

    There are also operations which are not strictly mapping to a pulse sequence, e.g.
    the active reset of qubits, where, after a measurement of the qubit, a :math:`pi`
    rotation is conditionally applied to reset the qubit in its ground state.

    This kind of operations are temporarily not supported by Qibolab, and for this
    reason the set of native operations reduces to pulse sequences.

The details of the pulse sequences definiton are described in details in the mentioned
:ref:`Experiment API <main_doc_experiment>`.
However, the important part is that each experiment supports its serialization, and it
is stored as such among the platform's so-called *parameters*.

The other main element which constitutes the :class:`qibolab.Parameters` are the
common hardware configurations.
E.g., one possible configuration is the frequency of the local oscillator used for the
upconversion of a certain set of channels.

The main separation between the general hardware configurations and the experiments
definitions (gates' pulse sequences) is the time in which they play role in the overall
experiment execution:

- pulse sequences are intended to contain operations which are executed according to a
  precise schedule, which is often to happen in *real time*
- the only moment when the general configurations will play a role is in the experiment
  preparation, thus *ahead of time*

All this information is known by the platform object, and can be arbitrarily queried,
following the declared schema (which is part of Qibolab's public API).
Moreover, the parameters are serialized on disk with a single method call
(:meth:`qibolab.Platform.dump`), for persistence across different runs.

Definition
----------

Channels
^^^^^^^^

Qubits
^^^^^^

x

----


.. _main_doc_channels:

Channels
--------

Channels play a pivotal role in connecting the quantum system to the control infrastructure.
Various types of channels are typically present in a quantum laboratory setup, including:

- the probe line (from device to qubit)
- the acquire line (from qubit to device)
- the drive line
- the flux line
- the TWPA pump line

Qibolab provides a general :class:`qibolab.Channel` object, as well as specializations depending on the channel role.
A channel is typically associated with a specific port on a control instrument, with port-specific properties like "attenuation" and "gain" that can be managed using provided getter and setter methods.
Channels are uniquely identified within the platform through their id.

The idea of channels is to streamline the pulse execution process.
The :class:`qibolab.PulseSequence` is a list of ``(channel_id, pulse)`` tuples, so that the platform identifies the channel that every pulse plays
and directs it to the appropriate port on the control instrument.

In setups involving frequency-specific pulses, a local oscillator (LO) might be required for up-conversion.
Although logically distinct from the qubit, the LO's frequency must align with the pulse requirements.
Qibolab accommodates this by enabling the assignment of a :class:`qibolab._core.instruments.oscillator.LocalOscillator` object
to the relevant channel :class:`qibolab.IqChannel`.
The controller's driver ensures the correct pulse frequency is set based on the LO's configuration.

Each channel has a :class:`qibolab._core.components.configs.Config` associated to it, which is a container of parameters related to the channel.
Configs also have different specializations that correspond to different channel types.
The platform holds default config parameters for all its channels, however the user is able to alter them by passing a config updates dictionary
when calling :meth:`qibolab.Platform.execute`.
The final configs are then sent to the controller instrument, which matches them to channels via their ids and ensures they are uploaded to the proper electronics.


.. _main_doc_qubits:

Qubits
------

The :class:`qibolab.Qubit` class serves as a container for the channels that are used to control the corresponding physical qubit.
These channels encompass distinct types, each serving a specific purpose:

- probe (measurement probe from controller device to the qubits)
- acquisition (measurement acquisition from qubits to controller)
- drive
- flux
- drive_extra (additional drive channels at different frequencies used to probe higher-level transition)

Some channel types are optional because not all hardware platforms require them.
For example, flux channels are typically relevant only for flux tunable qubits.

The :class:`qibolab.Qubit` class can also be used to represent coupler qubits, when these are available.

.. _main_doc_native:

Native
------

Each quantum platform supports a specific set of native gates, which are the quantum operations that have been calibrated.
If this set is universal any circuit can be transpiled and compiled to a pulse sequence which can then be deployed in the given platform.

:py:mod:`qibolab._core.native` provides data containers for holding the pulse parameters required for implementing every native gate.
The :class:`qibolab.Platform` provides a natives property that returns the :class:`qibolab._core.native.SingleQubitNatives`
which holds the single qubit native gates for every qubit and :class:`qibolab._core.native.TwoQubitNatives` for the two-qubit native gates of every qubit pair.
Each native gate is represented by a :class:`qibolab.PulseSequence` which contains all the calibrated parameters.

Typical single-qubit native gates are the Pauli-X gate, implemented via a pi-pulse which is calibrated using Rabi oscillations and the measurement gate,
implemented via a pulse sent in the readout line followed by an acquisition.
For a universal set of single-qubit gates, the RX90 (pi/2-pulse) gate is required,
which is implemented by halving the amplitude of the calibrated pi-pulse.

Typical two-qubit native gates are the CZ and iSWAP, with their availability being platform dependent.
These are implemented with a sequence of flux pulses, potentially to multiple qubits, and virtual Z-phases.
Depending on the platform and the quantum chip architecture, two-qubit gates may require pulses acting on qubits that are not targeted by the gate.
