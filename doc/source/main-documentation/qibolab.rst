.. _main_doc_platform:

Platforms
---------

Qibolab provides support to different quantum laboratories.

Each lab configuration is implemented using a :class:`qibolab.platform.Platform` object which orchestrates instruments, qubits and channels and provides the basic features for executing pulses.
Therefore, the ``Platform`` enables the user to interface with all
the required lab instruments at the same time with minimum effort.

The API reference section provides a description of all the attributes and methods of the ``Platform``. Here, let's focus on the main elements.

In the platform, the main methods can be divided in different sections:

- functions save and change qubit parameters (``dump``, ``update``)
- functions to coordinate the instruments (``connect``, ``setup``, ``start``, ``stop``, ``disconnect``)
- functions to execute experiments (``execute_pulse_sequence``, ``execute_pulse_sequences``, ``sweep``)
- functions to initialize gates (``create_RX90_pulse``, ``create_RX_pulse``, ``create_CZ_pulse``, ``create_MZ_pulse``, ``create_qubit_drive_pulse``, ``create_qubit_readout_pulse``, ``create_RX90_drag_pulse``, ``create_RX_drag_pulse``)
- setters and getters of channel/qubit parameters (local oscillator parameters, attenuations, gain and biases)

The idea of the ``Platform`` is to serve as the only object exposed to the user,  so that we can deploy experiments, without any need of going into the low-level instrument-specific code.

For example, let's first define a platform (that we consider to be a single qubit platform) using the ``create`` method presented in :doc:`/tutorials/lab`:

.. testcode::  python

    from qibolab import create_platform

    platform = create_platform("dummy")

Now we connect and start the instruments (note that we, the user, do not need to know which instruments are connected).

.. testcode::  python

    platform.connect()
    platform.setup()
    platform.start()

We can easily print some of the parameters of the channels (similarly we can set those, if needed):

.. note::
   If the get_method does not apply to the platform (for example there is no local oscillator, to TWPA or no flux tunability...) a ``NotImplementedError`` will be raised.

.. testcode::  python

    print(f"Drive LO frequency: {platform.qubits[0].drive.lo_frequency}")
    print(f"Readout LO frequency: {platform.qubits[0].readout.lo_frequency}")
    print(f"TWPA LO frequency: {platform.qubits[0].twpa.lo_frequency}")

    print(f"Qubit bias: {platform.get_bias(0)}")
    print(f"Qubit attenuation: {platform.get_attenuation(0)}")

.. testoutput:: python
    :hide:

    Drive LO frequency: 0
    Readout LO frequency: 0
    Qubit bias: 0
    TWPA LO frequency: 1000000000.0
    Qubit attenuation: 0

Now we can create a simple sequence (again, without explicitly giving any qubit specific parameter, as these are loaded automatically from the platform, as defined in the runcard):

.. testcode::  python

   from qibolab.pulses import PulseSequence

   ps = PulseSequence()
   ps.add(platform.create_RX_pulse(qubit=0, start=0))  # start time is in ns
   ps.add(platform.create_RX_pulse(qubit=0, start=100))
   ps.add(platform.create_MZ_pulse(qubit=0, start=200))

Now we can execute the sequence on hardware:

.. testcode::  python

   results = platform.execute_pulse_sequence(ps)

Finally, we can stop instruments and close connections.

.. testcode::  python

    platform.stop()
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

- readout (from controller device to the qubits)
- feedback (from qubits to controller)
- twpa (pump to the TWPA)
- drive
- flux

The Qubit class allows you to set and manage several key parameters that influence qubit behavior.
These parameters are typically extracted from the runcard during platform initialization.

.. _main_doc_couplers:

Couplers
--------

The :class:`qibolab.couplers.Coupler` class serves as a comprehensive representation of a physical coupler qubit within the Qibolab framework.
It's a simplified :class:`qibolab.qubits.Qubit` to control couplers during 2q gate operation:

- :ref:`Channels <main_doc_channels>`: Physical Connection
- :class:`Parameters <qibolab.couplers.Coupler>`: Configurable Properties
- :ref:`Qubits <main_doc_qubits>`: Qubits the coupler acts on

We have a single required Channel for flux coupler control:

- flux

The Coupler class allows us to handle 2q interactions in coupler based architectures
in a simple way. They are usually associated with :class:`qibolab.qubits.QubitPair`
and usually extracted from the runcard during platform initialization.

.. _main_doc_channels:

Channels
--------

In Qibolab, channels serve as abstractions for physical wires within a laboratory setup.
Each :class:`qibolab.channels.Channel` object corresponds to a specific type of connection, simplifying the process of controlling quantum pulses across the experimental setup.

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

Let's explore an example using an RFSoC controller.
Note that while channels are defined in a device-independent manner, the port parameter varies based on the specific instrument.

.. testcode:: python

    from qibolab.channels import Channel, ChannelMap
    from qibolab.instruments.rfsoc import RFSoC

    controller = RFSoC(name="dummy", address="192.168.0.10", port="6000")
    channel1 = Channel("my_channel_name_1", port=controller[1])
    channel2 = Channel("my_channel_name_2", port=controller[2])
    channel3 = Channel("my_channel_name_3", port=controller[3])

Channels are then organized in :class:`qibolab.channels.ChannelMap` to be passed as a single argument to the platform.
Following the tutorial in :doc:`/tutorials/lab`, we can continue the initialization:

.. testcode:: python

    from qibolab.serialize import load_qubits, load_runcard

    ch_map = ChannelMap()
    ch_map |= channel1
    ch_map |= channel2
    ch_map |= channel3

    qubits, pairs = load_qubits, load_runcard
    runcard = load_runcard(runcard_path)
    qubits, pairs = load_qubits(runcard)

    qubits[0].drive = channel1
    qubits[0].readout = channel2
    qubits[0].feedback = channel3

Where, in the last lines, we assign the channels to the qubits.

To assign local oscillators, the procedure is simple:

.. testcode:: python

    from qibolab.instruments.erasynth import ERA as LocalOscillator

    LO_ADDRESS = "192.168.0.10"
    local_oscillator = LocalOscillator("NameLO", LO_ADDRESS)
    local_oscillator.frequency = 6e9  # Hz
    local_oscillator.power = 5  # dB
    channel2.local_oscillator = local_oscillator

.. _main_doc_pulses:

Pulses
------

In Qibolab, an extensive API is available for working with pulses and pulse sequences, a fundamental aspect of quantum experiments.
At the heart of this API is the :class:`qibolab.pulses.Pulse` object, which empowers users to define and customize pulses with specific parameters.

The API provides specialized subclasses tailored to the main types of pulses typically used in quantum experiments:

- Readout Pulses (:class:`qibolab.pulses.ReadoutPulse`)
- Drive Pulses (:class:`qibolab.pulses.DrivePulse`)
- Flux Pulses (:class:`qibolab.pulses.FluxPulse`)

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
        start=0,  # Timing, always in nanoseconds (ns)
        duration=40,  # Pulse duration in ns
        amplitude=0.5,  # Amplitude relative to instrument range
        frequency=1e8,  # Frequency in Hz
        relative_phase=0,  # Phase in radians
        shape=Rectangular(),
        channel="channel",
        type="qd",  # Enum type: :class:`qibolab.pulses.PulseType`
        qubit=0,
    )

In this way, we defined a rectangular drive pulse using the generic Pulse object.
Alternatively, you can achieve the same result using the dedicated :class:`qibolab.pulses.DrivePulse` object:

.. testcode:: python

    from qibolab.pulses import DrivePulse, Rectangular

    pulse = DrivePulse(
        start=0,  # timing, in all qibolab, is expressed in ns
        duration=40,
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        frequency=1e8,  # frequency are in Hz
        relative_phase=0,  # phases are in radians
        shape=Rectangular(),
        channel="channel",
        qubit=0,
    )

Both the Pulses objects and the PulseShape object have useful plot functions and several different various helper methods.

To organize pulses into sequences, Qibolab provides the :class:`qibolab.pulses.PulseSequence` object. Here's an example of how you can create and manipulate a pulse sequence:

.. testcode:: python

    from qibolab.pulses import PulseSequence

    sequence = PulseSequence()

    pulse1 = DrivePulse(
        start=0,  # timing, in all qibolab, is expressed in ns
        duration=40,
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        frequency=1e8,  # frequency are in Hz
        relative_phase=0,  # phases are in radians
        shape=Rectangular(),
        channel="channel",
        qubit=0,
    )
    pulse2 = DrivePulse(
        start=0,  # timing, in all qibolab, is expressed in ns
        duration=40,
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        frequency=1e8,  # frequency are in Hz
        relative_phase=0,  # phases are in radians
        shape=Rectangular(),
        channel="channel",
        qubit=0,
    )
    pulse3 = DrivePulse(
        start=0,  # timing, in all qibolab, is expressed in ns
        duration=40,
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        frequency=1e8,  # frequency are in Hz
        relative_phase=0,  # phases are in radians
        shape=Rectangular(),
        channel="channel",
        qubit=0,
    )
    pulse4 = DrivePulse(
        start=0,  # timing, in all qibolab, is expressed in ns
        duration=40,
        amplitude=0.5,  # this amplitude is relative to the range of the instrument
        frequency=1e8,  # frequency are in Hz
        relative_phase=0,  # phases are in radians
        shape=Rectangular(),
        channel="channel",
        qubit=0,
    )
    sequence.add(pulse1)
    sequence.add(pulse2)
    sequence.add(pulse3)
    sequence.add(pulse4)

    print(f"Total duration: {sequence.duration}")

    sequence_ch1 = sequence.get_channel_pulses("channel1")  # Selecting pulses on channel 1
    print(f"We have {sequence_ch1.count} pulses on channel 1.")

.. testoutput:: python
    :hide:

    Total duration: 40
    We have 0 pulses on channel 1.

.. warning::

    Pulses in PulseSequences are ordered automatically following the start time (and the channel if needed). Not by the definition order.

When conducting experiments on quantum hardware, pulse sequences are vital. Assuming you have already initialized a platform, executing an experiment is as simple as:

.. testcode:: python

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
    result = platform.execute_pulse_sequence(sequence, options=options)

Lastly, when conducting an experiment, it is not always required to define a pulse from scratch.
Usual pulses, such as pi-pulses or measurements, are already defined in the platform runcard and can be easily initialized with platform methods.
These are relying on parameters held in the :ref:`main_doc_native` data structures.
Typical experiments may include both pre-defined pulses and new ones:

.. testcode:: python

    from qibolab.pulses import Rectangular

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(0))
    sequence.add(
        DrivePulse(
            start=0,
            duration=10,
            amplitude=0.5,
            frequency=2500000000,
            relative_phase=0,
            shape=Rectangular(),
            channel="0",
        )
    )
    sequence.add(platform.create_MZ_pulse(0, start=0))

    results = platform.execute_pulse_sequence(sequence, options=options)

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

- Frequency
- Amplitude
- Duration
- Relative_phase
- Start

--

- Attenuation
- Gain
- Bias

The first group includes parameters of the pulses, while the second group include parameters of a different type that, in qibolab, are linked to a qubit object.

To designate the qubit or pulse to which a sweeper is applied, you can utilize the ``pulses`` or ``qubits`` parameter within the Sweeper object.

.. note::

   It is possible to simultaneously execute the same sweeper on different pulses or qubits. The ``pulses`` or ``qubits`` attribute is designed as a list, allowing for this flexibility.

To effectively specify the sweeping behavior, Qibolab provides the ``values`` attribute along with the ``type`` attribute.

The ``values`` attribute comprises an array of numerical values that define the sweeper's progression. To facilitate multi-qubit execution, these numbers can be interpreted in three ways:

- Absolute Values: Represented by `qibolab.sweeper.PulseType.ABSOLUTE`, these values are used directly.
- Relative Values with Offset: Utilizing `qibolab.sweeper.PulseType.OFFSET`, these values are relative to a designated base value, corresponding to the pulse or qubit value.
- Relative Values with Factor: Employing `qibolab.sweeper.PulseType.FACTOR`, these values are scaled by a factor from the base value, akin to a multiplier.

For offset and factor sweepers, the base value is determined by the respective pulse or qubit value.

Let's see some examples.
Consider now a system with three qubits (qubit 0, qubit 1, qubit 2) with resonator frequency at 4 GHz, 5 GHz and 6 GHz.
A tipical resonator spectroscopy experiment could be defined with:

.. testcode:: python

    import numpy as np

    from qibolab.sweeper import Parameter, Sweeper, SweeperType

    sequence = PulseSequence()
    sequence.add(platform.create_MZ_pulse(0, start=0))  # readout pulse for qubit 0 at 4 GHz
    sequence.add(platform.create_MZ_pulse(1, start=0))  # readout pulse for qubit 1 at 5 GHz
    sequence.add(platform.create_MZ_pulse(2, start=0))  # readout pulse for qubit 2 at 6 GHz

    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(-200_000, +200_000, 1),  # define an interval of swept values
        pulses=[sequence[0], sequence[1], sequence[2]],
        type=SweeperType.OFFSET,
    )

    results = platform.sweep(sequence, options, sweeper)

.. note::

   options is an :class:`qibolab.execution_parameters.ExecutionParameters` object, detailed in a separate section.

In this way, we first define a sweeper with an interval of 400 MHz (-200 MHz --- 200 MHz), assigning it to all three readout pulses and setting is as an offset sweeper. The resulting probed frequency will then be:
    - for qubit 0: [3.8 GHz, 4.2 GHz]
    - for qubit 1: [4.8 GHz, 5.2 GHz]
    - for qubit 2: [5.8 GHz, 6.2 GHz]

If we had used the :class:`qibolab.sweeper.SweeperType` absolute, we would have probed for all qubits the same frequencies [-200 MHz, 200 MHz].

.. note::

   The default :class:`qibolab.sweeper.SweeperType` is absolute!

For factor sweepers, usually useful when dealing with amplitudes, the base value is multipled by the values set.

It is possible to define and executes multiple sweepers at the same time.
For example:

.. testcode:: python

    sequence = PulseSequence()

    sequence.add(platform.create_RX_pulse(0))
    sequence.add(platform.create_MZ_pulse(0, start=sequence[0].finish))

    sweeper_freq = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(-100_000, +100_000, 10_000),
        pulses=[sequence[0]],
        type=SweeperType.OFFSET,
    )
    sweeper_amp = Sweeper(
        parameter=Parameter.amplitude,
        values=np.arange(0, 1.5, 0.1),
        pulses=[sequence[0]],
        type=SweeperType.FACTOR,
    )

    results = platform.sweep(sequence, options, sweeper_freq, sweeper_amp)

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

   res = platform.execute_pulse_sequence(sequence, options=options)
   res = platform.sweep(sequence, options=options)

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

    drive_pulse_1 = platform.create_MZ_pulse(0, start=0)
    measurement_pulse = platform.create_qubit_readout_pulse(0, start=0)

    sequence = PulseSequence()
    sequence.add(drive_pulse_1)
    sequence.add(measurement_pulse)

    options = ExecutionParameters(
        nshots=1000,
        relaxation_time=10,
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    res = platform.execute_pulse_sequence(sequence, options=options)

The ``res`` object will manifest as a dictionary, mapping the measurement pulse serial to its corresponding results.

The values related to the results will be find in the ``voltages`` attribute for IntegratedResults and RawWaveformResults, while for SampleResults  the values are in ``samples``.

While for execution of sequences the results represent single measurements, but what happens for sweepers?
the results will be upgraded: from values to arrays and from arrays to matrices.

The shape of the values of an integreted acquisition with 2 sweepers will be:

.. testcode:: python

    sweeper1 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(-100_000, +100_000, 1),  # define an interval of swept values
        pulses=[sequence[0]],
        type=SweeperType.OFFSET,
    )
    sweeper2 = Sweeper(
        parameter=Parameter.frequency,
        values=np.arange(-200_000, +200_000, 1),  # define an interval of swept values
        pulses=[sequence[0]],
        type=SweeperType.OFFSET,
    )
    shape = (options.nshots, len(sweeper1.values), len(sweeper2.values))

.. _main_doc_transpiler:

Transpiler and Compiler
-----------------------

While pulse sequences can be directly deployed using a platform, circuits need to first be transpiled and compiled to the equivalent pulse sequence.
This procedure typically involves the following steps:

1. The circuit needs to respect the chip topology, that is, two-qubit gates can only target qubits that share a physical connection. To satisfy this constraint SWAP gates may need to be added to rearrange the logical qubits.
2. All gates are transpiled to native gates, which represent the universal set of gates that can be implemented (via pulses) in the chip.
3. Native gates are compiled to a pulse sequence.

The transpilation and compilation process is taken care of automatically by the :class:`qibolab.backends.QibolabBackend` when a circuit is executed, using :class:`qibolab.transpilers.abstract.Transpiler` and :class:`qibolab.compilers.compiler.Compiler`.
The transpiler is responsible for steps 1 and 2, while the compiler for step 3 of the list above. In order to accomplish this, several transpilers are provided, some of which are listed below:

- :class:`qibolab.transpilers.gate_decompositions.NativeGates`: Transpiles single-qubit Qibo gates to Z, RZ, GPI2 or U3 and two-qubit gates to CZ and/or iSWAP (depending on platform support).
- :class:`qibolab.transpilers.star_connectivity.StarConnectivity`: Transforms a circuit to respect a 5-qubit star chip topology, with one middle qubit connected to each of the remaining four qubits.
- :class:`qibolab.transpilers.routing.ShortestPaths`: Transforms a circuit to respect a general chip topology given as a networkx graph, using a greedy algorithm.
- :class:`qibolab.transpilers.pipeline.Pipeline`: Applies a list of other transpilers sequentially.

Custom transpilers can be added by inheriting the abstract :class:`qibolab.transpilers.abstract.Transpiler` class.

Once a circuit has been transpiled, it is converted to a :class:`qibolab.pulses.PulseSequence` by the :class:`qibolab.compilers.compiler.Compiler`.
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

Each native gate is represented by a :class:`qibolab.native.NativePulse` or :class:`qibolab.native.NativeSequence` which contain all the calibrated parameters and can be converted to an actual :class:`qibolab.pulses.PulseSequence` that is then executed in the platform.
Typical single-qubit native gates are the Pauli-X gate, implemented via a pi-pulse which is calibrated using Rabi oscillations and the measurement gate, implemented via a pulse sent in the readout line followed by an acquisition.
For a universal set of single-qubit gates, the RX90 (pi/2-pulse) gate is required, which is implemented by halving the amplitude of the calibrated pi-pulse.
U3, the most general single-qubit gate can be implemented using two RX90 pi-pulses and some virtual Z-phases which are included in the phase of later pulses.

Typical two-qubit native gates are the CZ and iSWAP, with their availability being platform dependent.
These are implemented with a sequence of flux pulses, potentially to multiple qubits, and virtual Z-phases.
Depending on the platform and the quantum chip architecture, two-qubit gates may require pulses acting on qubits that are not targeted by the gate.
The :class:`qibolab.native.NativeType` flag is used for communicating the set of available native two-qubit gates to the transpiler.

.. _main_doc_instruments:

Instruments
-----------

One the key features of qibolab is its support for multiple different instruments.
A list of all the supported instruments follows:

Controllers (subclasses of :class:`qibolab.instruments.abstract.Controller`):
    - Dummy Instrument: :class:`qibolab.instruments.dummy.DummyInstrument`
    - Zurich Instruments: :class:`qibolab.instruments.zhinst.Zurich`
    - Quantum Machines: :class:`qibolab.instruments.qm.driver.QMOPX`
    - Qblox: :class:`qibolab.instruments.qblox.cluster.Cluster`
    - Xilinx RFSoCs: :class:`qibolab.instruments.rfsoc.driver.RFSoC`

Other Instruments (subclasses of :class:`qibolab.instruments.abstract.Instrument`):
    - Erasynth++: :class:`qibolab.instruments.erasynth.ERA`
    - RohseSchwarz SGS100A: :class:`qibolab.instruments.rohde_schwarz.SGS100A`
    - Qutech SPI rack: :class:`qibolab.instruments.qutech.SPI`

Instruments all implement a set of methods:

- connect
- setup
- start
- stop
- disconnect

While the controllers, the main instruments in a typical setup, add other two methods:

- execute_pulse_sequence
- sweep

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
    "RTS start",                    "yes","yes","yes","yes"
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

Qblox
^^^^^

Supports the following Instruments:

- Cluster
- Cluster QRM-RF
- Cluster QCM-RF
- Cluster QCM

Compatible with qblox-instruments driver 0.9.0 (28/2/2023).

RFSoCs
^^^^^^

Compatible and tested with:

- Xilinx RFSoC4x2
- Xilinx ZCU111
- Xilinx ZCU216

Technically compatible with any board running ``qibosoq``.
