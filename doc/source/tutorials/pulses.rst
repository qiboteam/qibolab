Pulses execution
================

First, we create the pulse sequence that will be executed. We can do this by
defining a :class:`qibolab.pulses.PulseSequence` object and adding different
pulses (:class:`qibolab.pulses.Pulse`) through the
:func:`qibolab.pulses.PulseSequence.add()` method:

.. literalinclude:: ./includes/pulses/pulses0.py

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

.. literalinclude:: ./includes/pulses/pulses1.py

Remember to turn off the instruments and disconnect from the lab using the
``stop()`` and ``disconnect()`` methods of the platform.
