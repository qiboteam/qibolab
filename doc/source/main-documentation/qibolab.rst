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
   ps.concatenate(natives.RX.create_sequence())
   ps.concatenate(natives.RX.create_sequence(phi=np.pi / 2))
   ps.append((qubit.probe, Delay(duration=200)))
   ps.concatenate(natives.MZ.create_sequence())

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


.. _main_doc_qubits:

Qubits
------

The :class:`qibolab.qubits.Qubit` class serves as a comprehensive representation of a physical qubit within the Qibolab framework.
It encapsulates three fundamental elements crucial to qubit control and operation:

- :ref:`Channels <main_doc_channels>`: Physical Connections
- :class:`Parameters <qibolab.qubits.Qubit>`: Configurable Properties
- :ref:`Native Gates <main_doc_native>`: Quantum Operations

Channels play a pivotal role in connecting the quantum system to the control infrastructure.
They are optional and encompass distinct types, each serving a specific purpose:

- measure (from controller device to the qubits)
- acquisition (from qubits to controller)
- drive
- flux

The Qubit class allows you to set and manage several key parameters that influence qubit behavior.
These parameters are typically extracted from the runcard during platform initialization.

.. _main_doc_couplers:

Couplers
--------

Instead of using a dedicated class, a :class:`qibolab.qubits.Qubit` object can also
serve as a comprehensive representation of a physical coupler qubit within the Qibolab
framework.
Used like this, it would control couplers during 2q gate operation:

- :ref:`Channels <main_doc_channels>`: Physical Connection
- :class:`Parameters <qibolab.qubit.Qubit>`: Configurable Properties
- :ref:`Qubits <main_doc_qubits>`: Qubits the coupler acts on

We have a single required Channel for flux coupler control:

- flux

These instances allow us to handle 2q interactions in coupler based architectures
in a simple way. They are usually associated with :class:`qibolab.qubits.QubitPair`
and usually extracted from the runcard during platform initialization.

.. _main_doc_channels:

Channels
--------

Various types of channels are typically present in a quantum laboratory setup, including:

- the drive line
- the readout line (from device to qubit)
- the feedback line (from qubit to device)
- the flux line
- the TWPA pump line

A channel is typically associated with a specific port on a control instrument, with port-specific properties like "attenuation" and "gain" that can be managed using provided getter and setter methods.

The idea of channels is to streamline the pulse execution process.
When initiating a pulse, the platform identifies the corresponding channel for the pulse type and directs it to the appropriate port on the control instrument.
For instance, to deliver a drive pulse to a qubit, the platform references the qubit's associated channel and delivers the pulse to the designated port.

In setups involving frequency-specific pulses, a local oscillator (LO) might be required for up-conversion.
Although logically distinct from the qubit, the LO's frequency must align with the pulse requirements.
Qibolab accommodates this by enabling the assignment of a :class:`qibolab.instruments.oscillator.LocalOscillator` object to the relevant channel.
The controller's driver ensures the correct pulse frequency is set based on the LO's configuration.

.. _main_doc_pulses:

Pulses
------

In Qibolab, an extensive API is available for working with pulses and pulse sequences, a fundamental aspect of quantum experiments.
At the heart of this API is the :class:`qibolab.pulses.Pulse` object, which empowers users to define and customize pulses with specific parameters.

Each pulse is associated with a channel and a qubit.
Additionally, pulses are defined by a shape, represented by a subclass of :class:`qibolab.pulses.PulseShape`.
Qibolab offers a range of pre-defined pulse shapes:

- Rectangular (:class:`qibolab.pulses.Rectangular`)
- Exponential (:class:`qibolab.pulses.Exponential`)
- Gaussian (:class:`qibolab.pulses.Gaussian`)
- Drag (:class:`qibolab.pulses.Drag`)
- IIR (:class:`qibolab.pulses.IIR`)
- SNZ (:class:`qibolab.pulses.SNZ`)
- eCap (:class:`qibolab.pulses.eCap`)
- Custom (:class:`qibolab.pulses.Custom`)

To illustrate, here are some examples of single pulses using the Qibolab API:

.. testcode:: python

    from qibolab.pulses import Pulse, Rectangular

    pulse = Pulse(
        duration=40,  # Pulse duration in ns
        amplitude=0.5,  # Amplitude relative to instrument range
        relative_phase=0,  # Phase in radians
        envelope=Rectangular(),
    )

In this way, we defined a rectangular drive pulse using the generic Pulse object.
Alternatively, you can achieve the same result using the dedicated :class:`qibolab.pulses.Pulse` object:

.. testcode:: python

    from qibolab.pulses import Pulse, Rectangular

    pulse = Pulse(
        duration=40,  # timing, in all qibolab, is expressed in ns
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        relative_phase=0,  # phases are in radians
        envelope=Rectangular(),
    )

Both the Pulses objects and the PulseShape object have useful plot functions and several different various helper methods.

To organize pulses into sequences, Qibolab provides the :class:`qibolab.pulses.PulseSequence` object. Here's an example of how you can create and manipulate a pulse sequence:

.. testcode:: python

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
    sequence = PulseSequence.load(
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

    result = platform.execute([sequence], options=options)

Lastly, when conducting an experiment, it is not always required to define a pulse from scratch.
Usual pulses, such as pi-pulses or measurements, are already defined in the platform runcard and can be easily initialized with platform methods.
These are relying on parameters held in the :ref:`main_doc_native` data structures.
Typical experiments may include both pre-defined pulses and new ones:

.. testcode:: python

    from qibolab.pulses import Rectangular
    from qibolab.identifier import ChannelId

    natives = platform.natives.single_qubit[0]
    sequence = PulseSequence()
    sequence.concatenate(natives.RX.create_sequence())
    sequence.append(
        (
            "some/drive",
            Pulse(duration=10, amplitude=0.5, relative_phase=0, envelope=Rectangular()),
        )
    )
    sequence.concatenate(natives.MZ.create_sequence())

    results = platform.execute([sequence], options=options)

.. note::

   options is an :class:`qibolab.execution_parameters.ExecutionParameters` object, detailed in a separate section.


Sweepers
--------

Sweeper objects, represented by the :class:`qibolab.sweeper.Sweeper` class, stand as a crucial component in experiments and calibration tasks within the Qibolab framework.

Consider a scenario where a resonator spectroscopy experiment is performed. This process involves a sequence of steps:

1. Define a pulse sequence.
2. Define a readout pulse with frequency A.
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
A tipical resonator spectroscopy experiment could be defined with:

.. testcode:: python

    import numpy as np

    from qibolab.sweeper import Parameter, Sweeper

    natives = platform.natives.single_qubit

    sequence = PulseSequence()
    sequence.concatenate(
        natives[0].MZ.create_sequence()
    )  # readout pulse for qubit 0 at 4 GHz
    sequence.concatenate(
        natives[1].MZ.create_sequence()
    )  # readout pulse for qubit 1 at 5 GHz
    sequence.concatenate(
        natives[2].MZ.create_sequence()
    )  # readout pulse for qubit 2 at 6 GHz

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

It is possible to define and executes multiple sweepers at the same time.
For example:

.. testcode:: python

    from qibolab.pulses import Delay
    from qibolab.sequence import PulseSequence

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]
    sequence = PulseSequence()
    sequence.concatenate(natives.RX.create_sequence())
    sequence.append((qubit.probe, Delay(duration=sequence.duration)))
    sequence.concatenate(natives.MZ.create_sequence())

    f0 = platform.config(str(qubit.drive)).frequency
    sweeper_freq = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 100_000, f0 + 100_000, 10_000),
        channels=[qubit.drive],
    )
    sweeper_amp = Sweeper(
        parameter=Parameter.amplitude,
        range=(0, 0.43, 0.3),
        pulses=[next(iter(sequence.channel(qubit.drive)))],
    )

    results = platform.execute([sequence], options, [[sweeper_freq], [sweeper_amp]])

Let's say that the RX pulse has, from the runcard, a frequency of 4.5 GHz and an amplitude of 0.3, the parameter space probed will be:

- amplitudes: [0, 0.03, 0.06, 0.09, 0.12, ..., 0.39, 0.42]
- frequencies: [4.4999, 4.49991, 4.49992, ...., 4.50008, 4.50009] (GHz)

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

Within the Qibolab API, a variety of result types are available, contingent upon the chosen acquisition options. These results can be broadly classified into three main categories, based on the AcquisitionType:

- Integrated Results (:class:`qibolab.result.IntegratedResults`)
- Raw Waveform Results (:class:`qibolab.result.RawWaveformResults`)
- Sampled Results (:class:`qibolab.result.SampleResults`)

Furthermore, depending on whether results are averaged or not, they can be presented in an averaged version (as seen in :class:`qibolab.results.AveragedIntegratedResults`).

The result categories align as follows:

- AveragingMode: cyclic or sequential ->
    - AcquisitionType: integration -> :class:`qibolab.results.AveragedIntegratedResults`
    - AcquisitionType: raw -> :class:`qibolab.results.AveragedRawWaveformResults`
    - AcquisitionType: discrimination -> :class:`qibolab.results.AveragedSampleResults`
- AveragingMode: singleshot ->
    - AcquisitionType: integration -> :class:`qibolab.results.IntegratedResults`
    - AcquisitionType: raw -> :class:`qibolab.results.RawWaveformResults`
    - AcquisitionType: discrimination -> :class:`qibolab.results.SampleResults`

Let's now delve into a typical use case for result objects within the qibolab framework:

.. testcode:: python

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]

    sequence = PulseSequence()
    sequence.concatenate(natives.RX.create_sequence())
    sequence.append((qubit.probe, Delay(duration=sequence.duration)))
    sequence.concatenate(natives.MZ.create_sequence())

    options = ExecutionParameters(
        nshots=1000,
        relaxation_time=10,
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    res = platform.execute([sequence], options=options)

The ``res`` object will manifest as a dictionary, mapping the measurement pulse serial to its corresponding results.

The values related to the results will be find in the ``voltages`` attribute for IntegratedResults and RawWaveformResults, while for SampleResults  the values are in ``samples``.

While for execution of sequences the results represent single measurements, but what happens for sweepers?
the results will be upgraded: from values to arrays and from arrays to matrices.

The shape of the values of an integreted acquisition with 2 sweepers will be:

.. testcode:: python

    f0 = platform.config(str(qubit.drive)).frequency
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
    shape = (options.nshots, len(sweeper1.values), len(sweeper2.values))

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

Once a circuit has been compiled, it is converted to a :class:`qibolab.pulses.PulseSequence` by the :class:`qibolab.compilers.compiler.Compiler`.
This is a container of rules which define how each native gate can be translated to pulses.
A rule is a Python function that accepts a Qibo gate and a platform object and returns the :class:`qibolab.pulses.PulseSequence` implementing this gate and a dictionary with potential virtual-Z phases that need to be applied in later pulses.
Examples of rules can be found on :py:mod:`qibolab.compilers.default`, which defines the default rules used by Qibolab.

.. note::
   Rules return a :class:`qibolab.pulses.PulseSequence` for each gate, instead of a single pulse, because some gates such as the U3 or two-qubit gates, require more than one pulses to be implemented.

.. _main_doc_native:

Native
------

Each quantum platform supports a specific set of native gates, which are the quantum operations that have been calibrated.
If this set is universal any circuit can be transpiled and compiled to a pulse sequence which is then deployed in the given platform.

:py:mod:`qibolab.native` provides data containers for holding the pulse parameters required for implementing every native gate.
Every :class:`qibolab.qubits.Qubit` object contains a :class:`qibolab.native.SingleQubitNatives` object which holds the parameters of its native single-qubit gates,
while each :class:`qibolab.qubits.QubitPair` objects contains a :class:`qibolab.native.TwoQubitNatives` object which holds the parameters of the native two-qubit gates acting on the pair.

Each native gate is represented by a :class:`qibolab.pulses.Pulse` or :class:`qibolab.pulses.PulseSequence` which contain all the calibrated parameters.
Typical single-qubit native gates are the Pauli-X gate, implemented via a pi-pulse which is calibrated using Rabi oscillations and the measurement gate, implemented via a pulse sent in the readout line followed by an acquisition.
For a universal set of single-qubit gates, the RX90 (pi/2-pulse) gate is required, which is implemented by halving the amplitude of the calibrated pi-pulse.
U3, the most general single-qubit gate can be implemented using two RX90 pi-pulses and some virtual Z-phases which are included in the phase of later pulses.

Typical two-qubit native gates are the CZ and iSWAP, with their availability being platform dependent.
These are implemented with a sequence of flux pulses, potentially to multiple qubits, and virtual Z-phases.
Depending on the platform and the quantum chip architecture, two-qubit gates may require pulses acting on qubits that are not targeted by the gate.

.. _main_doc_instruments:

Instruments
-----------

One the key features of qibolab is its support for multiple different instruments.
A list of all the supported instruments follows:

Controllers (subclasses of :class:`qibolab.instruments.abstract.Controller`):
    - Dummy Instrument: :class:`qibolab.instruments.dummy.DummyInstrument`
    - Zurich Instruments: :class:`qibolab.instruments.zhinst.Zurich`
    - Quantum Machines: :class:`qibolab.instruments.qm.controller.QMController`

Other Instruments (subclasses of :class:`qibolab.instruments.abstract.Instrument`):
    - Erasynth++: :class:`qibolab.instruments.erasynth.ERA`
    - RohseSchwarz SGS100A: :class:`qibolab.instruments.rohde_schwarz.SGS100A`

Instruments all implement a set of methods:

- connect
- setup
- disconnect

While the controllers, the main instruments in a typical setup, add another, i.e.
execute.

Some more detail on the interal functionalities of instruments is given in :doc:`/tutorials/instrument`

The most important instruments are the controller, the following is a table of the current supported (or not supported) features, dev stands for `under development`:

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
