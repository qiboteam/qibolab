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

The :class:`qibolab.qubits.Qubit` is a representation of a physical qubit, it mainly contains three elements:

    - channels
    - parameters
    - native_gates

The channels, better explained in the channels section, represent the physical wires in a laboratory.
The channels are all optional and come in different types:

    - readout (from controller device to the qubits)
    - feedback (from qubits to controller)
    - twpa (pump to the twpa)
    - drive
    - flux
    - flux_coupler

The settable parameters, that are read from the runcard when the platform is initialized, are:

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

Native
======

Channels
========

Pulses
======

In Qibolab we have a dedicated API to pulses and pulses sequence, which
at the moment works for both qblox and FPGAs setups.

The main component of the API is the :class:`qibolab.pulses.Pulse` object,
which enables the user to code a pulse with specific parameters. We provide
also a special object for the ``ReadoutPulse`` given its importance when dealing
with a quantum hardware. Moreover, we supports different kinds of :class:`qibolab.pulses.PulseShape`.

The :class:`qibolab.pulses.PulseSequence` class enables to combine different pulses
into a sequence through the ``add`` method.

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

Instruments
===========
