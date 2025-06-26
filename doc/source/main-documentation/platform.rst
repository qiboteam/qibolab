.. _main_doc_platform:

Platforms
=========

One of the core goals of Qibolab is to allow the execution of the experiments defined
with its own :ref:`Experiment API <main_doc_experiment>` on diverse platforms.

In order to do this, the main abstraction introduced is exactly the
:class:`.Platform` itself, which is intended to represent a collection of
instruments, suitably connected to the device operated as a QPU.

The handling of the instruments it will be mainly internal to the Qibolab itself, and it
is defined by the instruments' drivers.

Usage
-----

The whole workflow is supposed to be the following:

#. the experiment is defined by the user using the mentioned :ref:`Experiment API
   <main_doc_experiment>`, together with an optional set of temporary configuration
   updates
#. the execution is invoked through :meth:`.Platform.execute`, which acts as the
   single entry point
#. internally, the configurations and experiment definition are shared with all the
   registered instruments, iterating over those registered in the platform, and
   converting the instructions to the each instrument's representation (which is the main
   role of the driver), finally uploading them
#. the experiment is then triggerred
#. upon completion, results are downloaded from the relevant sources, and collected into
   a single collection, which is returned as the output of :meth:`.Platform.execute`

.. note::

    Because of internal Qibolab's limitations, currently it is assumed that there is
    just a single instrument capable of producing pulses (a
    :class:`._core.instruments.abstract.Controller` instance). While all the
    other instruments will play a passive role (e.g. LOs), which boils down in only
    supporting configurations, but do not execute any synchronized operation.

    This limitation will be lifted in future releases.

    However, it is mostly affecting the way instruments' drivers are written, since a
    single driver should span all the active components. Once this is done, it only
    limits the composability of existing instruments.

The only other active operations which are relevant for the experimental workflow
consist in instruments' initialization and close up, which are performed through
:meth:`.Platform.connect` and :meth:`.Platform.disconnect` methods.

.. hint::

    The platform just exposes the API for pulse-based experiments. However, it is always
    central for hardware execution.

    Indeed, Qibolab exposes a Qibo compatible backend for circuits execution,
    :class:`.QibolabBackend`. Which at its heart, it is powered by a platform
    itself.

    Cf. :ref:`main_doc_backend` and :ref:`main_doc_compiler`.

However, there is a further relevant role which is performed by the
:class:`.Platform`: parameters' persistance.

Parameters
^^^^^^^^^^

The role of the :class:`.Platform` is to store all the information required to
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

The other main element which constitutes the :class:`.Parameters` are the
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
(:meth:`.Platform.dump`), for persistence across different runs.

.. important::

   The serialization is so frequent, and such a relevant part of the platform's
   operations, which Qibolab supports as a pattern their loading from a file named
   ``parameters.json``, through the :meth:`.Platform.load` method.

   However, this pattern is fully optional, as it is described in depth in
   :ref:`main_doc_storage`.

Definition
----------

In the last section, the structure of platform's parameters has been described. These
are not the only constituent of the platform, since there is another important
information which needs to be defined: the hardware layer.

Indeed, to actually use a platform, a crucial information regards how to address each
involved instrument, and how to route the pulses to the correct channels.

This complementary information is represented by the :class:`.Hardware` class,
which can be promoted to full-fledged :class:`.Platform` by providing an instance
of :class:`.Parameters`.
The information contain by a :class:`.Hardware` is the following:

- :attr:`.Hardware.instruments`, an identifier to instrument instance mapping,
  which may require further parameters to be instantiated
- :attr:`.Hardware.qubits` and :attr:`.Hardware.couplers`, which are
  just collections of channels identifiers, to easily retrieve channels from their role
  (described in the section below)

These two objects are mainly used to manage and access channels, which are then
described in the next two sections.
Indeed, the information of instruments may vary according to specific instrument kind
(i.e. class), but the common minimal content are:

- the network address, use to communicate with the device
- the information regarding the controlled channels - cf. next section

Other than the data they hold, the :class:`.Instrument`, and especially the
:class:`.Controller` (those instruments generating pulses and acquiring signals),
are the computational units used by Qibolab to delegate the compilation of experiments
instructions and configurations over a diverse set of possible instruments.
More on this topic will be described in the :ref:`main_doc_instruments` section.

.. note::

   While this section intends to describe the concepts behind platforms' definition, a
   practical guide can be found in a :ref:`dedicated tutorial <tutorial_platform>`.

Channels
^^^^^^^^

The mentioned *channels identifiers* label are the central ingredient to pulse routing
in the instruments' drivers. Indeed, one of the few parameters common to all instruments
instances is exactly the channel mapping.
Indeed, the channels are intended to be "owned" by the instrument generating the pulses
for that channel. This is true both at a conceptual and practical level, since the
instrument instance will then contain the only :class:`.Channel` instance, which
store the information related to:

- the *path* specifier, which is required to direct instructions to the correct location
  within the instrument
- other related instrument and channels (e.g. the *probe* channel on the same
  transmission line of an *acquisition* channel, or the mixer and local oscillator
  related to a certain modulated channel)

Because of this second point, different kind of channels may be defined.
E.g. a :class:`.DcChannel` is distinguished from an :class:`.IqChannel`
because of modulation, which potentially requires to coordinate the operation of such a
channel with an external mixer (identified by :attr:`.IqChannel.mixer`).

Channels' configurations
~~~~~~~~~~~~~~~~~~~~~~~~

Notice that channel identifiers play even a further role: they identify the channels'
configurations in the overall configuration mapping, part of the platform's parameters
(as described above).


- only common attribute is path, used to route in the driver
- the rest only store configurations
- main distinction built-in ones: output DC and RF, and input
- they can be further specialized

Qubits
^^^^^^

The :class:`.Qubit` class serves as a container for the channels that are used to
control the corresponding physical qubit.

These channels encompass distinct types, each serving a specific purpose:

- :attr:`.Qubit.probe`, measurement probe from controller device to the qubits
- :attr:`.Qubit.acquisition`, measurement acquisition from qubits to controller
- :attr:`.Qubit.drive`, used to control the single qubit Hamiltonian
- :attr:`.Qubit.flux`, tuning the qubit frequency through magnetic flux
- :attr:`.Qubit.drive_extra`, additional drive channels at different frequencies

The container structure is specifically engineered to match the typical roles in the
superconducting qubits.
However, this is just a structured collection for ease of access. Notice how the
channels (described in the section above) only retain the information related to their
operations, but not directly to the role they play in any experiment.
In this sense, the names above are just established as a convention, but they introduce
no limitation to the way the :class:`.Qubit` is used (see the note below).

Indeed, all elements are optional, because not all hardware platforms and elements
require them.
E.g., flux channels are typically relevant only for flux-tunable qubits.

Moreover, the :class:`.Qubit` class is also be used to represent coupler qubits,
when these are part of the platform. This case is quite complementary to the fixed
frequency transmon: only the :attr:`.Qubit.flux` line is used.

.. note::

    While :attr:`.Qubit.drive_extra` is named after *drive* role, there is no
    restriction to the type of channels it can contain, playing essentially the role of
    unadministered free space.

    What is often expected for these channels would be to be used for additional drives
    to implement further type of gates involving the qubit, and especially the same
    physical line of the :attr:`.Qubit.drive` channel. Mainly, this will be used
    to implement gates supposed to act on higher levels (qudits), and cross-resonance
    interactions.

    At present time, these guidelines are not enforced anyhow in .

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

Qibolab provides a general :class:`.Channel` object, as well as specializations depending on the channel role.
A channel is typically associated with a specific port on a control instrument, with port-specific properties like "attenuation" and "gain" that can be managed using provided getter and setter methods.
Channels are uniquely identified within the platform through their id.

The idea of channels is to streamline the pulse execution process.
The :class:`.PulseSequence` is a list of ``(channel_id, pulse)`` tuples, so that the platform identifies the channel that every pulse plays
and directs it to the appropriate port on the control instrument.

In setups involving frequency-specific pulses, a local oscillator (LO) might be required for up-conversion.
Although logically distinct from the qubit, the LO's frequency must align with the pulse requirements.
Qibolab accommodates this by enabling the assignment of a :class:`._core.instruments.oscillator.LocalOscillator` object
to the relevant channel :class:`.IqChannel`.
The controller's driver ensures the correct pulse frequency is set based on the LO's configuration.

Each channel has a :class:`._core.components.configs.Config` associated to it, which is a container of parameters related to the channel.
Configs also have different specializations that correspond to different channel types.
The platform holds default config parameters for all its channels, however the user is able to alter them by passing a config updates dictionary
when calling :meth:`.Platform.execute`.
The final configs are then sent to the controller instrument, which matches them to channels via their ids and ensures they are uploaded to the proper electronics.
