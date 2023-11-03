Pulses execution
================

First, we create the pulse sequence that will be executed. We can do this by
defining a :class:`qibolab.pulses.PulseSequence` object and adding different
pulses (:class:`qibolab.pulses.Pulse`) through the
:func:`qibolab.pulses.PulseSequence.add()` method:

.. testcode::  python

    from qibolab.pulses import (
        DrivePulse,
        ReadoutPulse,
        PulseSequence,
        Rectangular,
        Gaussian,
    )

    # Define PulseSequence
    sequence = PulseSequence()

    # Add some pulses to the pulse sequence
    sequence.add(
        DrivePulse(
            start=0,
            frequency=200000000,
            amplitude=0.3,
            duration=60,
            relative_phase=0,
            shape=Gaussian(5),
            qubit=0,
        )
    )
    sequence.add(
        ReadoutPulse(
            start=70,
            frequency=20000000.0,
            amplitude=0.5,
            duration=3000,
            relative_phase=0,
            shape=Rectangular(),
            qubit=0,
        )
    )

The next step consists in connecting to a specific lab in which the pulse
sequence will be executed. In order to do this we allocate a platform  object
via the :func:`qibolab.create_platform("name")` where ``name`` is the name of
the platform that will be used. The ``Platform`` constructor also takes care of
loading the runcard containing all the calibration settings for that specific
platform.

After connecting and setting up the platform's instruments using the
``connect()`` and ``setup()`` methods, the ``start`` method will turn on the
local oscillators and the ``execute`` method will execute the previous defined
pulse sequence according to the number of shots ``nshots`` specified.

.. testcode::  python

    from qibolab import create_platform
    from qibolab.execution_parameters import ExecutionParameters

    # Define platform and load specific runcard
    platform = create_platform("dummy")

    # Connects to lab instruments using the details specified in the calibration settings.
    platform.connect()

    # Configures instruments using the loaded calibration settings.
    platform.setup()

    # Turns on the local oscillators
    platform.start()

    # Executes a pulse sequence.
    options = ExecutionParameters(nshots=1000, relaxation_time=100)
    results = platform.execute_pulse_sequence(sequence, options=options)

    # Turn off lab instruments
    platform.stop()

    # Disconnect from the instruments
    platform.disconnect()

Remember to turn off the instruments and disconnect from the lab using the
``stop()`` and ``disconnect()`` methods of the platform.
