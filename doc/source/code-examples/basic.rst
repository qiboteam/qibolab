Basic examples
==============

Here are a few short basic `how to` examples.

How to execute a pulse sequence?
--------------------------------

.. code-block::  python

    from qibolab import platform, pulses # automatically connects to the platform
    from qibolab.pulse_shapes import Rectangular, Gaussian


    # define pulse sequence
    sequence = pulses.PulseSequence()
    sequence.add(pulses.Pulse(start=0,
                              frequency=200000000.0,
                              amplitude=0.3,
                              duration=60,
                              phase=0,
                              shape=Gaussian(60 / 5))) # Gaussian shape with std = duration / 5
    sequence.add(pulses.ReadoutPulse(start=70,
                                     frequency=20000000.0,
                                     amplitude=0.5,
                                     duration=3000,
                                     phase=0,
                                     shape=Rectangular()))

    # turn on instruments
    platform.start()
    # execute sequence and acquire results
    results = platform.execute(sequence)
    # turn off instruments
    platform.stop()
