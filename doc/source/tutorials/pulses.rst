.. admonition:: Work in progress

    This page is only partially updated from a previous version of Qibolab.

    In case of doubts, contact the `Qibo developers
    <https://github.com/qiboteam/qibo#contacts>`_.

Pulses execution
================

We can create pulse sequence using the Qibolab pulse API directly,
defining a :class:`qibolab.PulseSequence` object and adding different
pulses (:class:`qibolab.Pulse`) using the :func:`qibolab.PulseSequence.append()` method:

.. testcode::  python

    from qibolab import Delay, Gaussian, Pulse, PulseSequence, Rectangular

    # Define PulseSequence
    sequence = PulseSequence.load(
        [
            (
                "0/drive",
                Pulse(
                    amplitude=0.3,
                    duration=60,
                    relative_phase=0,
                    envelope=Gaussian(rel_sigma=0.2),
                ),
            ),
            ("1/drive", Delay(duration=100)),
            (
                "1/drive",
                Pulse(
                    amplitude=0.5, duration=3000, relative_phase=0, envelope=Rectangular()
                ),
            ),
        ]
    )


The next step consists in connecting to a specific lab in which the pulse
sequence will be executed. In order to do this we allocate a platform  object
via the :func:`qibolab.create_platform("name")` where ``name`` is the name of
the platform that will be used. The ``Platform`` constructor also takes care of
loading the runcard containing all the calibration settings for that specific
platform.

After connecting to the platform's instruments using the ``connect()``,
we can execute the previously defined sequence using the ``execute`` method:

.. testcode::  python

    from qibolab import create_platform

    # Define platform and load specific runcard
    platform = create_platform("dummy")

    # Connects to lab instruments using the details specified in the calibration settings.
    platform.connect()

    # Executes a pulse sequence.
    results = platform.execute([sequence], nshots=1000, relaxation_time=100)

    # Disconnect from the instruments
    platform.disconnect()

Remember to turn off and disconnect from the instruments using the
``disconnect()`` methods of the platform.

.. note::
    Calling ``platform.connect()`` automatically turns on auxilliary instruments such as local oscillators.

Alternatively, instead of using the pulse API directly, one can use the native gate data structures to write a pulse sequence:

.. testcode::  python

    import numpy as np

    from qibolab import Delay, Gaussian, Pulse, PulseSequence, Rectangular, create_platform

    platform = create_platform("dummy")
    q0 = platform.natives.single_qubit[0]
    sequence = q0.R(theta=np.pi / 2) | q0.MZ()
