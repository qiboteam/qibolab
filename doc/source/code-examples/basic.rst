Basic examples
==============

Here are a few short basic `how to` examples.

How to execute a pulse sequence on a given platform?
----------------------------------------------------

First, we create the pulse sequence that will be executed.
We can do this by defining a ``PulseSequence`` object and adding different
pulses (:class:`qibolab.pulses.Pulse`) through the ``PulseSequence.add()`` method:

.. code-block::  python

    from qibolab.pulses import Pulse, ReadoutPulse
    from qibolab.circuit import PulseSequence
    from qibolab.pulse_shapes import Rectangular, Gaussian

    # Define PulseSequence
    sequence = PulseSequence()

    # Add some pulses to the pulse sequence
    sequence.add(Pulse(start=0,
                              frequency=200000000.0,
                              amplitude=0.3,
                              duration=60,
                              phase=0,
                              shape=Gaussian(60 / 5))) # Gaussian shape with std = duration / 5
    sequence.add(ReadoutPulse(start=70,
                                     frequency=20000000.0,
                                     amplitude=0.5,
                                     duration=3000,
                                     phase=0,
                                     shape=Rectangular()))

The next step consists in connecting to a specific lab in which
the pulse sequence will be executed. In order to do this we
allocate a platform  object ``Platform("name")`` where ``name`` is
the name of the platform that will be used. The ``Platform`` constructor
also takes care of loading the runcard containing all the calibration
settings for that specific platform.

After connecting and setting up the platform's instruments using the
``connect()`` and ``setup()`` methods, the ``execute`` method will execute
the previous defined pulse sequence according to the number of shots ``nshots``
specified.

.. code-block::  python

    from qibolab import Platform

    # Define platform and load specific runcard
    platform = Platform("tiiq")
    # Connect to the lab instruments
    platform.connect()
    # Configure instruments using runcard settings
    platform.setup()
    # Execute pulse sequence with a given number of shots
    # and retrieve the results
    results = platform.execute(ps, nshots=10)
    # Turn off lab instruments
    platform.stop()
    # Disconnect from the instruments
    platform.disconnect()

Remember to turn off the instruments and disconnect from the lab using the
``stop()`` and ``disconnect()`` method of the platform.