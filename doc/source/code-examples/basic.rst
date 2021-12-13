Basic examples
==============

Here are a few short basic `how to` examples.

How to define a pulse?
----------------------

.. code-block::  python

    from qibolab import pulses
    from qibolab.pulse_shapes import Rectangular, Gaussian

    # example with gaussian pulse
    pulse1 = pulses.Pulse(start=0,
                          frequency=200000000.0,
                          amplitude=0.3,
                          duration=60,
                          phase=0,
                          shape=Gaussian(60 / 5),
                          channel="qcm"))

    # rectangular pulse
    pulse2 = pulses.Pulse(start=70,
                          frequency=20000000.0,
                          amplitude=0.5,
                          duration=3000,
                          phase=0,
                          shape=Rectangular(),
                          channel="qrm"))

    # readout pulse (automatically allocated to QRM)
    pulse3 = pulses.ReadoutPulse(start=70,
                                 frequency=20000000.0,
                                 amplitude=0.5,
                                 duration=3000,
                                 phase=0,
                                 shape=Rectangular())


How to execute a circuit?
-------------------------


.. code-block::  python

    from qibolab import platform, pulses

    # platform.connect() is performed in the constructor

    platform.start() # turn on the instruments

    # define pulses as above
    pulse1 = pulses.Pulse(start=0,
                          frequency=200000000.0,
                          amplitude=0.3,
                          duration=60,
                          phase=0,
                          shape=Gaussian(60 / 5))
    pulse2 = pulses.ReadoutPulse(start=70,
                                 frequency=20000000.0,
                                 amplitude=0.5,
                                 duration=3000,
                                 phase=0,
                                 shape=Rectangular())

    # define PulseSequence object and add pulses
    pulseseq = pulses.PulseSequence()
    pulseseq.add(pulse1)
    pulseseq.add(pulse2)

    # execute and acquire result
    result = platform.execute(pulseseq)

    # stop the instruments
    platform.stop()

    # platform.disconnect() is performed in the destructor
