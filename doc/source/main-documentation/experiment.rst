.. _main_doc_experiment:

Experiment
==========

Pulses
------

In Qibolab, an extensive API is available for working with pulses and pulse sequences, a fundamental aspect of quantum experiments.
At the heart of this API is the :class:`qibolab.Pulse` object, which empowers users to define and customize pulses with specific parameters.

Additionally, pulses are defined by an envelope shape, represented by a subclass of :class:`qibolab._core.pulses.envelope.BaseEnvelope`.
Qibolab offers a range of pre-defined pulse shapes which can be found in :py:mod:`qibolab._core.pulses.envelope`.

- Rectangular (:class:`qibolab.Rectangular`)
- Exponential (:class:`qibolab.Exponential`)
- Gaussian (:class:`qibolab.Gaussian`)
- Drag (:class:`qibolab.Drag`)
- IIR (:class:`qibolab.Iir`)
- SNZ (:class:`qibolab.Snz`)
- eCap (:class:`qibolab.ECap`)
- Custom (:class:`qibolab.Custom`)

To illustrate, here is an examples of how to instantiate a pulse using the Qibolab API:

.. testcode:: python

    from qibolab import Pulse, Rectangular

    pulse = Pulse(
        duration=40.0,  # Pulse duration in ns
        amplitude=0.5,  # Amplitude normalized to [-1, 1]
        relative_phase=0.0,  # Phase in radians
        envelope=Rectangular(),
    )

Here, we defined a rectangular drive pulse using the generic Pulse object.

Both the Pulses objects and the PulseShape object have useful plot functions and several different various helper methods.

To organize pulses into sequences, Qibolab provides the :class:`qibolab.PulseSequence` object. Here's an example of how you can create and manipulate a pulse sequence:

.. testcode:: python

    from qibolab import Pulse, PulseSequence, Rectangular


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

    from qibolab import Rectangular

    natives = platform.natives.single_qubit[0]
    sequence = natives.RX() | natives.MZ()

    results = platform.execute([sequence])


Sweepers
--------

Sweeper objects, represented by the :class:`qibolab.Sweeper` class, stand as a crucial component in experiments and calibration tasks within the Qibolab framework.

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

Sweeper objects in Qibolab are characterized by a :class:`qibolab.Parameter`. This parameter, crucial to the sweeping process, can be one of several types:

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

    from qibolab import Parameter, Sweeper

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

    results = platform.execute([sequence], [sweepers], **options)

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

    results = platform.execute([sequence], [[sweeper_freq], [sweeper_amp]], **options)

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

In the course of several examples, you've encountered the ``**options`` argument in function calls like:

.. testcode:: python

   res = platform.execute([sequence], **options)

Let's now delve into the details of the ``options`` and understand its parts.

The ``options`` extra arguments, is a vital element for every hardware execution.
It encompasses essential information that tailors the execution to specific requirements:

- ``nshots``: Specifies the number of experiment repetitions.
- ``relaxation_time``: Introduces a wait time between repetitions, measured in nanoseconds (ns).
- ``fast_reset``: Enables or disables fast reset functionality, if supported; raises an error if not supported.
- ``acquisition_type``: Determines the acquisition mode for results.
- ``averaging_mode``: Defines the mode for result averaging.

The first three parameters are straightforward in their purpose. However, let's take a closer look at the last two parameters.

Supported acquisition types, accessible via the :class:`qibolab.AcquisitionType` enumeration, include:

- Discrimination: Distinguishes states based on acquired voltages.
- Integration: Returns demodulated and integrated waveforms.
- Raw: Offers demodulated, yet unintegrated waveforms.

Supported averaging modes, available through the :class:`qibolab.AveragingMode` enumeration, consist of:

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


    ro_pulse = ro_sequence[0][1]
    result = platform.execute(
        [sequence],
        nshots=1000,
        relaxation_time=10,
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )


``result`` will be a dictionary with a single key ``ro_pulse.id`` and an array of
two elements, the averaged I and Q components of the integrated signal.
If instead, ``(AcquisitionType.INTEGRATION, AveragingMode.SINGLESHOT)`` was used, the array would have shape ``(options["nshots"], 2)``,
while for ``(AcquisitionType.DISCRIMINATION, AveragingMode.SINGLESHOT)`` the shape would be ``(options["nshots"],)`` with values 0 or 1.

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
    shape = (options["nshots"], len(sweeper1.values), len(sweeper2.values), 2)
