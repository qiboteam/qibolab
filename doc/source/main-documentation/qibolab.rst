Platforms
=========

Qibolab provides support to different quantum laboratories.

Each lab is implemented using a :class:`qibolab.platform.Platform` object which implements basic features and connects instruments, qubits and channels.
Therefore, the ``Platform`` enables the user to interface with all
the required lab instruments at the same time with minimum effort.

In the API reference section, a description of all the attributes and methods of the ``Platform`` is provided. here, let's focus on the main elements.

In the platform, the main methods can be divided in different sections:

- functions to load qubit parameters, save them and change them (``reload_settings``, ``dump``, ``update``)
- functions to coordinare the instruments (``connect``, ``setup``, ``start``, ``stop``, ``disconnect``)
- functions to execute experiments (``execute_pulse_sequence``, ``execute_pulse_sequences``, ``sweep``)
- functions to initialize gates (``create_RX90_pulse``, ``create_RX_pulse``, ``create_CZ_pulse``, ``create_MZ_pulse``, ``create_qubit_drive_pulse``, ``create_qubit_readout_pulse``, ``create_RX90_drag_pulse``, ``create_RX_drag_pulse``)
- setter and getter of channel/qubit parameters (local oscillator parameters, attenuations, gain and biases)

The idea of the ``Platform`` is to serve as the only exposed object to the user so, for example, we can easily write an example of experiment, whithout any need of going into the low-level instrument-dpecific code.

For example, let's first define a platform (that we consider to be a single qubit platform) using the ``create`` method presented in :doc:`/tutorials/lab`:

.. code-block::  python

    from qibolab import Platform
    platform = Platform("my_platform")

Now we connect and start the instruments (note that we, the user, do not need to know which instruments are connected).

.. code-block::  python

    platform.connect()
    platform.setup()
    platform.start()

We can easily print some of the parameters of the channels (similarly we can set those, if needed):

.. note::
   If the get_method does not apply to the platform (for example there is no local oscillator, to TWPA or no flux tunability...) a ``NotImplementedError`` will be raised.

.. code-block::  python

    print(f"Drive LO frequency: {platform.get_lo_drive_frequency(0)}")
    print(f"Readout LO frequency: {platform.get_lo_readout_frequency(0)}")
    print(f"TWPA LO frequency: {platform.get_lo_twpa_frequency(0)}")
    print(f"TWPA LO power: {platform.get_lo_twpa_power(0)}")

    print(f"Qubit bias: {platform.get_bias(0)}")
    print(f"Qubit attenuation: {platform.get_attenuation(0)}")

Now we can create a simple sequence (again, without explicitly give any qubit specific parameter, defined in the runcard):

.. code-block::  python

   from qibolab.pulses import PulseSequence

   ps = PulseSequence()
   ps.add(platform.create_RX_pulse(qubit=0, start=0))   # start time is in ns
   ps.add(platform.create_RX_pulse(qubit=0, start=100))
   ps.add(platform.create_MZ_pulse(qubit=0, start=200))

Now we can execute the sequence on hardware:

.. code-block::  python

   results = platform.execute_pulse_sequence(ps)

Finally, we can stop instruments and close connections.

.. code-block::  python

    platform.stop()
    platform.disconnect()


Qubits
======

The :class:`qibolab.qubits.Qubit` class serves as a comprehensive representation of a physical qubit within the Qibolab framework.
It encapsulates three fundamental elements crucial to qubit control and operation:

- Channels: Physical Connections
- Parameters: Configurable Properties
- Native Gates: Quantum Operations

Channels play a pivotal role in connecting the quantum system to the control infrastructure.
They are optional and encompass distinct types, each serving a specific purpose:

- readout (from controller device to the qubits)
- feedback (from qubits to controller)
- twpa (pump to the twpa)
- drive
- flux
- flux_coupler

The Qubit class allows you to set and manage several key parameters that influence qubit behavior.
These parameters are typically extracted from the runcard during platform initialization.
Notable settable parameters include:

- bare_resonator_frequency
- readout_frequency
- drive_frequency
- anharmonicity
- Ec
- Ej
- g
- assigment_fidelity
- sweetspot
- peak_vol`tage
- pi_pulse_amplitude
- T1
- T2
- T2_spin_echo
- state0_voltage
- state1_voltage
- mean_gnd_states
- mean_exc_states
- threshold
- iq_angle

Channels
========

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

.. code-block:: python

    from qibolab.channels import Channel, ChannelMap
    from qibolab.instruments.rfsoc import RFSoC

    controller = RFSoC()
    channel1 = Channel("my_channel_name_1", port=controller[1])
    channel2 = Channel("my_channel_name_2", port=controller[2])
    channel3 = Channel("my_channel_name_3", port=controller[3])

Channels are then organized in :class:`qibolab.channels.ChannelMap` to be passed as a single argument to the platform.
Following the tutorial in :doc:`/tutorials/lab`, we can continue the initialization:

.. code-block:: python

    ch_map = ChannelMap()
    ch_map |= channel1
    ch_map |= channel2
    ch_map |= channel3

    platform = Platform(Name, runcard, instruments, ch_map)

    platform.qubits[0].drive = channel1
    platform.qubits[0].readout = channel2
    platform.qubits[0].feedback = channel3

Where, in the last lines, we assign the channels to the qubits.

To assign local oscillators, the procedure is simple:

.. code-block:: python

    from qibolab.instruments.erasynth import ERA as LocalOscillator

    local_oscillator = LocalOscillator("NameLO", LO_ADDRESS)
    local_oscillator.frequency = 6e9  # Hz
    local_oscillator.power = 5  # dB
    channel2.local_oscillator = local_oscillator

Pulses
======

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

.. code-block:: python

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

.. code-block:: python

    from qibolab.pulses import *

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

.. code-block:: python

    from qibolab.pulses import PulseSequence

    sequence = PulseSequence()

    sequence.add(pulse1)
    sequence.add(pulse2)
    sequence.add(pulse3)
    sequence.add(pulse4)

    print(f"Total duration: {sequence.duration}")

    sequence_ch1 = sequence.get_channel_pulses("channel1")  # Selecting pulses on channel 1
    print(f"We have {sequence_ch1.count} pulses on channel 1.")

When conducting experiments on quantum hardware, pulse sequences are vital. Assuming you have already initialized a platform, executing an experiment is as simple as:

.. code-block:: python

   result = my_platform.execute_pulse_sequence(sequence)

Symbolic expressions
====================

Sweepers
========

Results
=======

Execution Parameters
====================

Transpiler
==========

Native
======

Instruments
===========
