Basic examples
==============

Here are a few short basic `how to` examples.

How to execute a pulse sequence on a given platform?
----------------------------------------------------

First, we create the pulse sequence that will be executed.
We can do this by defining a ``PulseSequence`` object and adding different
pulses (:class:`qibolab.pulses.Pulse`) through the ``PulseSequence.add()`` method:

.. code-block::  python

    from qibolab.pulses import (
        Pulse,
        ReadoutPulse,
        PulseSequence,
        Rectangular,
        Gaussian
    )

    # Define PulseSequence
    sequence = PulseSequence()

    # Add some pulses to the pulse sequence
    sequence.add(Pulse(start=0,
                       frequency=200000000.0, # in Hz
                       amplitude=0.3,         # instruments dependant
                       duration=60,           # in ns
                       relative_phase=0,      # in rad
                       shape=Gaussian(5)))    # Gaussian shape with std = duration / 5

    sequence.add(ReadoutPulse(start=70,
                              frequency=20000000.0,
                              amplitude=0.5,
                              duration=3000,
                              relative_phase=0,
                              shape=Rectangular()))

The next step consists in connecting to a specific lab in which
the pulse sequence will be executed. In order to do this we
allocate a platform  object qith ``create_platform("name")`` where ``name`` is
the name of the platform that will be used. The ``Platform`` constructor
also takes care of loading the runcard containing all the calibration
settings for that specific platform.

After connecting and setting up the platform's instruments using the
``connect()`` and ``setup()`` methods, the ``start`` method will turn on
the local oscillators and the ``execute`` method will execute
the previous defined pulse sequence according to the number of shots ``nshots``
specified.

.. code-block::  python

    from qibolab import create_platform

    # Define platform and load specific runcard
    platform = create_platform("tii1q")  # load a platform named tii1q
    # Connects to lab instruments using the details specified in the calibration settings.
    platform.connect()
    # Configures instruments using the loaded calibration settings.
    platform.setup()
    # Turns on the local oscillators
    platform.start()
    # Executes a pulse sequence.
    results = platform.execute(ps, nshots=10)
    # Turn off lab instruments
    platform.stop()
    # Disconnect from the instruments
    platform.disconnect()

Remember to turn off the instruments and disconnect from the lab using the
``stop()`` and ``disconnect()`` methods of the platform.
