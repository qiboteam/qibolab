.. _main_doc_platform:

Platforms
---------

Qibolab provides support to different quantum laboratories.

Each lab configuration is implemented using a :class:`qibolab.platform.Platform` object which orchestrates instruments,
qubits and channels and provides the basic features for executing pulses.
Therefore, the ``Platform`` enables the user to interface with all
the required lab instruments at the same time with minimum effort.

The API reference section provides a description of all the attributes and methods of the ``Platform``. Here, let's focus on the main elements.

In the platform, the main methods can be divided in different sections:

- functions to coordinate the instruments (``connect``, ``disconnect``)
- a unique interface to execute experiments (``execute``)
- functions save parameters (``dump``)

The idea of the ``Platform`` is to serve as the only object exposed to the user, so that we can deploy experiments,
without any need of going into the low-level instrument-specific code.

For example, let's first define a platform (that we consider to be a single qubit platform) using the ``create`` method presented in :doc:`/tutorials/lab`:

.. testcode::  python

    from qibolab import create_platform

    platform = create_platform("dummy")

Now we connect to the instruments (note that we, the user, do not need to know which instruments are connected).

.. testcode::  python

    platform.connect()

We can easily access the names of channels and other components, and based on the name retrieve the corresponding configuration. As an example let's print some things:

.. note::
   If requested component does not exist in a particular platform, its name will be `None`, so watch out for such names, and make sure what you need exists before requesting its configuration.

.. testcode::  python

    drive_channel_id = platform.qubits[0].drive
    drive_channel = platform.channels[drive_channel_id]
    print(f"Drive channel name: {drive_channel_id}")
    print(f"Drive frequency: {platform.config(drive_channel_id).frequency}")

    drive_lo = drive_channel.lo
    if drive_lo is None:
        print(f"Drive channel {drive_channel_id} does not use an LO.")
    else:
        print(f"Name of LO for channel {drive_channel_id} is {drive_lo}")
        print(f"LO frequency: {platform.config(drive_lo).frequency}")

.. testoutput:: python
    :hide:

    Drive channel name: 0/drive
    Drive frequency: 4000000000.0
    Drive channel 0/drive does not use an LO.

Now we can create a simple sequence without explicitly giving any qubit specific parameter,
as these are loaded automatically from the platform, as defined in the corresponding ``parameters.json``:

.. testcode::  python

   from qibolab.pulses import Delay
   from qibolab.sequence import PulseSequence
   import numpy as np

   ps = PulseSequence()
   qubit = platform.qubits[0]
   natives = platform.natives.single_qubit[0]
   ps.concatenate(natives.RX())
   ps.concatenate(natives.RX(phi=np.pi / 2))
   ps.append((qubit.probe, Delay(duration=200)))
   ps.concatenate(natives.MZ())

Now we can execute the sequence on hardware:

.. testcode::  python

    from qibolab.execution_parameters import (
        AcquisitionType,
        AveragingMode,
        ExecutionParameters,
    )

    options = ExecutionParameters(
        nshots=1000,
        relaxation_time=10,
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    results = platform.execute([ps], options=options)

Finally, we can stop instruments and close connections.

.. testcode::  python

    platform.disconnect()


.. _main_doc_dummy:

Dummy platform
^^^^^^^^^^^^^^

In addition to the real instruments presented in the :ref:`main_doc_instruments` section, Qibolab provides the :class:`qibolab.instruments.dummy.DummyInstrument`.
This instrument represents a controller that returns random numbers of the proper shape when executing any pulse sequence.
This instrument is also part of the dummy platform which is defined in :py:mod:`qibolab.dummy` and can be initialized as

.. testcode::  python

    from qibolab import create_platform

    platform = create_platform("dummy")

This platform is equivalent to real platforms in terms of attributes and functions, but returns just random numbers.
It is useful for testing parts of the code that do not necessarily require access to an actual quantum hardware platform.


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

Qibolab provides a general :class:`qibolab.components.channels.Channel` object, as well as specializations depending on the channel role.
A channel is typically associated with a specific port on a control instrument, with port-specific properties like "attenuation" and "gain" that can be managed using provided getter and setter methods.
Channels are uniquely identified within the platform through their id.

The idea of channels is to streamline the pulse execution process.
The :class:`qibolab.sequence.PulseSequence` is a list of ``(channel_id, pulse)`` tuples, so that the platform identifies the channel that every pulse plays
and directs it to the appropriate port on the control instrument.

In setups involving frequency-specific pulses, a local oscillator (LO) might be required for up-conversion.
Although logically distinct from the qubit, the LO's frequency must align with the pulse requirements.
Qibolab accommodates this by enabling the assignment of a :class:`qibolab.instruments.oscillator.LocalOscillator` object
to the relevant channel :class:`qibolab.components.channels.IqChannel`.
The controller's driver ensures the correct pulse frequency is set based on the LO's configuration.

Each channel has a :class:`qibolab.components.configs.Config` associated to it, which is a container of parameters related to the channel.
Configs also have different specializations that correspond to different channel types.
The platform holds default config parameters for all its channels, however the user is able to alter them by passing a config updates dictionary
when calling :meth:`qibolab.platform.Platform.execute`.
The final configs are then sent to the controller instrument, which matches them to channels via their ids and ensures they are uploaded to the proper electronics.


.. _main_doc_qubits:

Qubits
------

The :class:`qibolab.qubits.Qubit` class serves as a container for the channels that are used to control the corresponding physical qubit.
These channels encompass distinct types, each serving a specific purpose:

- probe (measurement probe from controller device to the qubits)
- acquisition (measurement acquisition from qubits to controller)
- drive
- flux
- drive_qudits (additional drive channels at different frequencies used to probe higher-level transition)

Some channel types are optional because not all hardware platforms require them.
For example, flux channels are typically relevant only for flux tunable qubits.

The :class:`qibolab.qubits.Qubit` class can also be used to represent coupler qubits, when these are available.


.. _main_doc_pulses:

Pulses
------

In Qibolab, an extensive API is available for working with pulses and pulse sequences, a fundamental aspect of quantum experiments.
At the heart of this API is the :class:`qibolab.pulses.pulse.Pulse` object, which empowers users to define and customize pulses with specific parameters.

Additionally, pulses are defined by an envelope shape, represented by a subclass of :class:`qibolab.pulses.envelope.BaseEnvelope`.
Qibolab offers a range of pre-defined pulse shapes which can be found in :py:mod:`qibolab.pulses.envelope`.

- Rectangular (:class:`qibolab.pulses.envelope.Rectangular`)
- Exponential (:class:`qibolab.pulses.envelope.Exponential`)
- Gaussian (:class:`qibolab.pulses.envelope.Gaussian`)
- Drag (:class:`qibolab.pulses.envelope.Drag`)
- IIR (:class:`qibolab.pulses.envelope.Iir`)
- SNZ (:class:`qibolab.pulses.envelope.Snz`)
- eCap (:class:`qibolab.pulses.envelope.ECap`)
- Custom (:class:`qibolab.pulses.envelope.Custom`)

To illustrate, here is an examples of how to instantiate a pulse using the Qibolab API:

.. testcode:: python

    from qibolab.pulses import Pulse, Rectangular

    pulse = Pulse(
        duration=40.0,  # Pulse duration in ns
        amplitude=0.5,  # Amplitude normalized to [-1, 1]
        relative_phase=0.0,  # Phase in radians
        envelope=Rectangular(),
    )

Here, we defined a rectangular drive pulse using the generic Pulse object.

Both the Pulses objects and the PulseShape object have useful plot functions and several different various helper methods.

To organize pulses into sequences, Qibolab provides the :class:`qibolab.sequence.PulseSequence` object. Here's an example of how you can create and manipulate a pulse sequence:

.. testcode:: python

    from qibolab.pulses import Pulse, Rectangular
    from qibolab.sequence import PulseSequence


    pulse1 = Pulse(
        duration=40,  # timing, in all qibolab, is expressed in ns
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        relative_phase=0,  # phases are in radians
        envelope=Rectangular(),
    )
    pulse2 = Pulse(
        duration=40,  # timing, in all qibolab, is expressed in ns
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        relative_phase=0,  # phases are in radians
        envelope=Rectangular(),
    )
    pulse3 = Pulse(
        duration=40,  # timing, in all qibolab, is expressed in ns
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        relative_phase=0,  # phases are in radians
        envelope=Rectangular(),
    )
    pulse4 = Pulse(
        duration=40,  # timing, in all qibolab, is expressed in ns
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        relative_phase=0,  # phases are in radians
        envelope=Rectangular(),
    )
    sequence = PulseSequence(
        [
            ("qubit/drive", pulse1),
            ("qubit/drive", pulse2),
            ("qubit/drive", pulse3),
            ("qubit/drive", pulse4),
        ],
    )

    print(f"Total duration: {sequence.duration}")


.. testoutput:: python
    :hide:

    Total duration: 160.0


When conducting experiments on quantum hardware, pulse sequences are vital. Assuming you have already initialized a platform, executing an experiment is as simple as:

.. testcode:: python

    result = platform.execute([sequence])

Lastly, when conducting an experiment, it is not always required to define a pulse from scratch.
Usual pulses, such as pi-pulses or measurements, are already defined in the platform runcard and can be easily initialized with platform methods.
These are relying on parameters held in the :ref:`main_doc_native` data structures.
Typical experiments may include both pre-defined pulses and new ones:

.. testcode:: python

    from qibolab.pulses import Rectangular
    from qibolab.identifier import ChannelId

    natives = platform.natives.single_qubit[0]
    sequence = natives.RX() | natives.MZ()

    results = platform.execute([sequence])


Sweepers
--------

Sweeper objects, represented by the :class:`qibolab.sweeper.Sweeper` class, stand as a crucial component in experiments and calibration tasks within the Qibolab framework.

Consider a scenario where a resonator spectroscopy experiment is performed. This process involves a sequence of steps:

1. Define a pulse sequence.
2. Define a readout pulse with frequency :math:`A`.
3. Execute the sequence.
4. Define a new readout pulse with frequency :math:`A + \epsilon`.
5. Execute the sequence again.
6. Repeat for increasing frequencies :math:`A + 2 \epsilon`, :math:`A + 3 \epsilon`, and so on.

This approach is suboptimal and time-consuming, mainly due to the frequent communication between the control device and the Qibolab user after each execution. Such communication overhead significantly extends experiment duration.

In supported control devices, an efficient technique involves defining a "sweeper" or a parameter scan directly on the device. This scan, applied to specific parameters, allows multiple variations to be executed in a single communication round, drastically reducing experiment time.

To address the inefficiency, Qibolab introduces the concept of Sweeper objects.

Sweeper objects in Qibolab are characterized by a :class:`qibolab.sweeper.Parameter`. This parameter, crucial to the sweeping process, can be one of several types:

- Amplitude
- Duration
- Relative_phase
- Start

--

- Frequency
- Offset

The first group includes parameters of the pulses, while the second group includes parameters of channels.

To designate the pulse(s) or channel(s) to which a sweeper is applied, you can utilize the ``pulses`` or ``channels`` parameter within the Sweeper object.

.. note::

   It is possible to simultaneously execute the same sweeper on different pulses or channels. The ``pulses`` or ``channels`` attribute is designed as a list, allowing for this flexibility.

To effectively specify the sweeping behavior, Qibolab provides the ``values`` attribute along with the ``type`` attribute.

The ``values`` attribute comprises an array of numerical values that define the sweeper's progression.

Let's see some examples.
Consider now a system with three qubits (qubit 0, qubit 1, qubit 2) with resonator frequency at 4 GHz, 5 GHz and 6 GHz.
A typical resonator spectroscopy experiment could be defined with:

.. testcode:: python

    import numpy as np

    from qibolab.sweeper import Parameter, Sweeper

    natives = platform.natives.single_qubit

    sequence = (
        natives[0].MZ()  # readout pulse for qubit 0 at 4 GHz
        | natives[1].MZ()  # readout pulse for qubit 1 at 5 GHz
        | natives[2].MZ()  # readout pulse for qubit 2 at 6 GHz
    )

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(qubit.probe).frequency
            + np.arange(-200_000, +200_000, 1),  # define an interval of swept values
            channels=[qubit.probe],
        )
        for qubit in platform.qubits.values()
    ]

    results = platform.execute([sequence], options, [sweepers])

.. note::

   options is an :class:`qibolab.execution_parameters.ExecutionParameters` object, detailed in a separate section.

In this way, we first define three parallel sweepers with an interval of 400 MHz (-200 MHz --- 200 MHz). The resulting probed frequency will then be:
    - for qubit 0: [3.8 GHz, 4.2 GHz]
    - for qubit 1: [4.8 GHz, 5.2 GHz]
    - for qubit 2: [5.8 GHz, 6.2 GHz]

It is possible to define and executes multiple sweepers at the same time, in a nested loop style.
For example:

.. testcode:: python

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]
    rx_sequence = natives.RX()
    sequence = rx_sequence | natives.MZ()

    f0 = platform.config(qubit.drive).frequency
    sweeper_freq = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 100_000, f0 + 100_000, 10_000),
        channels=[qubit.drive],
    )
    rx_pulse = rx_sequence[0][1]
    sweeper_amp = Sweeper(
        parameter=Parameter.amplitude,
        range=(0, 0.43, 0.3),
        pulses=[rx_pulse],
    )

    results = platform.execute([sequence], options, [[sweeper_freq], [sweeper_amp]])

Let's say that the RX pulse has, from the runcard, a frequency of 4.5 GHz and an amplitude of 0.3, the parameter space probed will be:

- amplitudes: [0, 0.03, 0.06, 0.09, 0.12, ..., 0.39, 0.42]
- frequencies: [4.4999, 4.49991, 4.49992, ...., 4.50008, 4.50009] (GHz)

Sweepers given in the same list will be applied in parallel, in a Python ``zip`` style,
while different lists define nested loops, with the first list corresponding to the outer loop.

.. warning::

   Different control devices may have different limitations on the sweepers.
   It is possible that the sweeper will raise an error, if not supported, or that it will be automatically converted as a list of pulse sequences to perform sequentially.

Execution Parameters
--------------------

In the course of several examples, you've encountered the ``options`` argument in function calls like:

.. testcode:: python

   res = platform.execute([sequence], options=options)

Let's now delve into the details of the ``options`` parameter and understand its components.

The ``options`` parameter, represented by the :class:`qibolab.execution_parameters.ExecutionParameters` class, is a vital element for every hardware execution. It encompasses essential information that tailors the execution to specific requirements:

- ``nshots``: Specifies the number of experiment repetitions.
- ``relaxation_time``: Introduces a wait time between repetitions, measured in nanoseconds (ns).
- ``fast_reset``: Enables or disables fast reset functionality, if supported; raises an error if not supported.
- ``acquisition_type``: Determines the acquisition mode for results.
- ``averaging_mode``: Defines the mode for result averaging.

The first three parameters are straightforward in their purpose. However, let's take a closer look at the last two parameters.

Supported acquisition types, accessible via the :class:`qibolab.execution_parameters.AcquisitionType` enumeration, include:

- Discrimination: Distinguishes states based on acquired voltages.
- Integration: Returns demodulated and integrated waveforms.
- Raw: Offers demodulated, yet unintegrated waveforms.

Supported averaging modes, available through the :class:`qibolab.execution_parameters.AveragingMode` enumeration, consist of:

- Cyclic: Provides averaged results, yielding a single IQ point per measurement.
- Singleshot: Supplies non-averaged results.

.. note::

    Two averaging modes actually exists: cyclic and sequential.
    In sequential mode, a sweeper is executed with the repetition loop nested inside, while cyclic mode places the sweeper as the outermost loop. Cyclic execution generally offers better noise resistance.
    Ideally, use the cyclic mode. However, some devices lack support for it and will automatically convert it to sequential execution.


Results
-------

``platform.execute`` returns a dictionary, mapping the acquisition pulse id to the results of the corresponding measurements.
The results of each measurement are a numpy array with dimension that depends on the number of shots, acquisition type,
averaging mode and the number of swept points, if sweepers were used.

For example in

.. testcode:: python

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]

    ro_sequence = natives.MZ()
    sequence = natives.RX() | ro_sequence

    options = ExecutionParameters(
        nshots=1000,
        relaxation_time=10,
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    ro_pulse = ro_sequence[0][1]
    result = platform.execute([sequence], options=options)


``result`` will be a dictionary with a single key ``ro_pulse.id`` and an array of
two elements, the averaged I and Q components of the integrated signal.
If instead, ``(AcquisitionType.INTEGRATION, AveragingMode.SINGLESHOT)`` was used, the array would have shape ``(options.nshots, 2)``,
while for ``(AcquisitionType.DISCRIMINATION, AveragingMode.SINGLESHOT)`` the shape would be ``(options.nshots,)`` with values 0 or 1.

The shape of the values of an integrated acquisition with two sweepers will be:

.. testcode:: python

    f0 = platform.config(qubit.drive).frequency
    sweeper1 = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 100_000, f0 + 100_000, 1),
        channels=[qubit.drive],
    )
    sweeper2 = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 200_000, f0 + 200_000, 1),
        channels=[qubit.probe],
    )
    shape = (options.nshots, len(sweeper1.values), len(sweeper2.values), 2)

.. _main_doc_compiler:

Transpiler and Compiler
-----------------------

While pulse sequences can be directly deployed using a platform, circuits need to first be transpiled and compiled to the equivalent pulse sequence.
This procedure typically involves the following steps:

1. The circuit needs to respect the chip topology, that is, two-qubit gates can only target qubits that share a physical connection. To satisfy this constraint SWAP gates may need to be added to rearrange the logical qubits.
2. All gates are transpiled to native gates, which represent the universal set of gates that can be implemented (via pulses) in the chip.
3. Native gates are compiled to a pulse sequence.

The transpiler is responsible for steps 1 and 2, while the compiler for step 3 of the list above.
To be executed in Qibolab, a circuit should be already transpiled. It possible to use the transpilers provided by Qibo to do it. For more information, please refer the `examples in the Qibo documentation <https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-modify-the-transpiler>`_.
On the other hand, the compilation process is taken care of automatically by the :class:`qibolab.backends.QibolabBackend`.

Once a circuit has been compiled, it is converted to a :class:`qibolab.sequence.PulseSequence` by the :class:`qibolab.compilers.compiler.Compiler`.
This is a container of rules which define how each native gate can be translated to pulses.
A rule is a Python function that accepts a Qibo gate and a platform object and returns the :class:`qibolab.pulses.PulseSequence` implementing this gate and a dictionary with potential virtual-Z phases that need to be applied in later pulses.
Examples of rules can be found on :py:mod:`qibolab.compilers.default`, which defines the default rules used by Qibolab.

.. note::
   Rules return a :class:`qibolab.sequence.PulseSequence` for each gate, instead of a single pulse, because some gates such as the U3 or two-qubit gates, require more than one pulses to be implemented.

.. _main_doc_native:

Native
------

Each quantum platform supports a specific set of native gates, which are the quantum operations that have been calibrated.
If this set is universal any circuit can be transpiled and compiled to a pulse sequence which can then be deployed in the given platform.

:py:mod:`qibolab.native` provides data containers for holding the pulse parameters required for implementing every native gate.
The :class:`qibolab.platform.Platform` provides a natives property that returns the :class:`qibolab.native.SingleQubitNatives`
which holds the single qubit native gates for every qubit and :class:`qibolab.native.TwoQubitNatives` for the two-qubit native gates of every qubit pair.
Each native gate is represented by a :class:`qibolab.sequence.PulseSequence` which contains all the calibrated parameters.

Typical single-qubit native gates are the Pauli-X gate, implemented via a pi-pulse which is calibrated using Rabi oscillations and the measurement gate,
implemented via a pulse sent in the readout line followed by an acquisition.
For a universal set of single-qubit gates, the RX90 (pi/2-pulse) gate is required,
which is implemented by halving the amplitude of the calibrated pi-pulse.

Typical two-qubit native gates are the CZ and iSWAP, with their availability being platform dependent.
These are implemented with a sequence of flux pulses, potentially to multiple qubits, and virtual Z-phases.
Depending on the platform and the quantum chip architecture, two-qubit gates may require pulses acting on qubits that are not targeted by the gate.

.. _main_doc_instruments:

Instruments
-----------

One the key features of Qibolab is its support for multiple different electronics.
A list of all the supported electronics follows:

Controllers (subclasses of :class:`qibolab.instruments.abstract.Controller`):
    - Dummy Instrument: :class:`qibolab.instruments.dummy.DummyInstrument`
    - Zurich Instruments: :class:`qibolab.instruments.zhinst.Zurich`
    - Quantum Machines: :class:`qibolab.instruments.qm.controller.QMController`

Other Instruments (subclasses of :class:`qibolab.instruments.abstract.Instrument`):
    - Erasynth++: :class:`qibolab.instruments.erasynth.ERA`
    - RohseSchwarz SGS100A: :class:`qibolab.instruments.rohde_schwarz.SGS100A`

All instruments inherit the :class:`qibolab.instruments.abstract.Instrument` and implement methods for connecting and disconnecting.
:class:`qibolab.instruments.abstract.Controller` is a special case of instruments that provides the :class:`qibolab.instruments.abstract.execute`
method that deploys sequences on hardware.

Some more detail on the interal functionalities of instruments is given in :doc:`/tutorials/instrument`

The following is a table of the currently supported or not supported features (dev stands for `under development`):

.. csv-table:: Supported features
    :header: "Feature", "RFSoC", "Qblox", "QM", "ZH"
    :widths: 25, 5, 5, 5, 5

    "Arbitrary pulse sequence",     "yes","yes","yes","yes"
    "Arbitrary waveforms",          "yes","yes","yes","yes"
    "Multiplexed readout",          "yes","yes","yes","yes"
    "Hardware classification",      "no","yes","yes","yes"
    "Fast reset",                   "dev","dev","dev","dev"
    "Device simulation",            "no","no","yes","dev"
    "RTS frequency",                "yes","yes","yes","yes"
    "RTS amplitude",                "yes","yes","yes","yes"
    "RTS duration",                 "yes","yes","yes","yes"
    "RTS relative phase",           "yes","yes","yes","yes"
    "RTS 2D any combination",       "yes","yes","yes","yes"
    "Sequence unrolling",           "dev","dev","dev","dev"
    "Hardware averaging",           "yes","yes","yes","yes"
    "Singleshot (no averaging)",    "yes","yes","yes","yes"
    "Integrated acquisition",       "yes","yes","yes","yes"
    "Classified acquisition",       "yes","yes","yes","yes"
    "Raw waveform acquisition",     "yes","yes","yes","yes"


Zurich Instruments
^^^^^^^^^^^^^^^^^^

Qibolab has been tested with the following `instrument cluster <https://www.zhinst.com/others/en/instruments/product-finder/type/quantum_computing_systems>`_:

- 1 `SHFQC` (Superconducting Hybrid Frequency Converter)
- 2 `HDAWGs` (High-Density Arbitrary Waveform Generators)
- 1 `PQSC` (Programmable Quantum System Controller)

The integration of Qibolab with the instrument cluster is facilitated through the `LabOneQ <https://github.com/zhinst/laboneq>`_ Python library that handles communication and coordination with the instruments.

Quantum Machines
^^^^^^^^^^^^^^^^

Tested with a cluster of nine `OPX+ <https://www.quantum-machines.co/products/opx/>`_ controllers, using QOP213 and QOP220.

Qibolab is communicating with the instruments using the `QUA <https://docs.quantum-machines.co/0.1/>`_ language, via the ``qm-qua`` and ``qualang-tools`` Python libraries.
