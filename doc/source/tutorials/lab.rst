How to connect Qibolab to your lab?
===================================

In this section we will show how to let Qibolab communicate with your lab's
instruments and run an experiment.

The main required object, in this case, is the `Platform`. A Platform is defined
as a QPU (quantum processing unit with one or more qubits) controlled by one ore
more instruments.

How to define a platform for a self-hosted QPU?
-----------------------------------------------

The :class:`qibolab.platform.Platform` object holds all the information required
to execute programs, and in particular :class:`qibolab.pulses.PulseSequence` in
a real QPU. It is comprised by different objects that contain information about
the qubit characterization and connectivity, the native gates and the lab's
instrumentation.

The following cell shows how to define a single qubit platform from scratch,
using different Qibolab primitives.

.. literalinclude:: ./includes/lab/lab0.py

This code creates a platform with a single qubit that is controlled by the
:class:`qibolab.instruments.dummy.DummyInstrument`. In real applications, if
Qibolab provides drivers for the instruments in the lab, these can be directly
used in place of the ``DummyInstrument`` above, otherwise new drivers need to be
coded following the abstract :class:`qibolab.instruments.abstract.Instrument`
interface.

Furthermore, above we defined three channels that connect the qubit to the
control instrument and we assigned two native gates to the qubit. In this
example we neglected or characterization parameters associated to the qubit.
These can be passed when defining the :class:`qibolab.qubits.Qubit` objects.

When the QPU contains more than one qubit, some of the qubits are connected so
that two-qubit gates can be applied. For such connected pairs of qubits one
needs to additionally define :class:`qibolab.qubits.QubitPair` objects, which
hold the parameters of the two-qubit gates.

.. literalinclude:: ./includes/lab1.py

Some architectures may also have coupler qubits that mediate the interactions.
We can also interact with them defining the :class:`qibolab.couplers.Coupler` objects.
Then we add them to their corresponding :class:`qibolab.qubits.QubitPair` objects according
to the chip topology. We neglected characterization parameters associated to the
coupler but qibolab will take them into account when calling :class:`qibolab.native.TwoQubitNatives`.

.. literalinclude:: ./includes/lab2.py

The platform automatically creates the connectivity graph of the given chip
using the dictionary of :class:`qibolab.qubits.QubitPair` objects.

Registering platforms
^^^^^^^^^^^^^^^^^^^^^

The ``create()`` function defined in the above example can be called or imported
directly in any Python script. Alternatively, it is also possible to make the
platform available as

.. code-block::  python

    from qibolab import create_platform

    # Define platform and load specific runcard
    platform = create_platform("my_platform")


To do so, ``create()`` needs to be saved in a module called ``platform.py`` inside
a folder with the name of this platform (in this case ``my_platform``).
Moreover, the environment flag ``QIBOLAB_PLATFORMS`` needs to point to the directory
that contains this folder.
Examples of advanced platforms are available at `this
repository <https://github.com/qiboteam/qibolab_platforms_qrc>`_.

.. _using_runcards:

Using runcards
^^^^^^^^^^^^^^

Operating a QPU requires calibrating a set of parameters, the number of which
increases with the number of qubits. Hardcoding such parameters in the
``create()`` function, as shown in the above examples, is not scalable. However,
since ``create()`` is part of a Python module, is is possible to load parameters
from an external file or database.

Qibolab provides some utility functions, accessible through
:py:mod:`qibolab.serialize`, for loading calibration parameters stored in a JSON
file with a specific format. We call such file a runcard. Here is a runcard for
a two-qubit system:

.. code-block::  json

    {
        "nqubits": 2,
        "qubits": [
            0,
            1
        ],
        "settings": {
            "nshots": 1024,
            "sampling_rate": 1000000000,
            "relaxation_time": 50000
        },
        "topology": [
            [
                0,
                1
            ]
        ],
        "native_gates": {
            "single_qubit": {
                "0": {
                    "RX": {
                        "duration": 40,
                        "amplitude": 0.0484,
                        "frequency": 4855663000,
                        "shape": "Drag(5, -0.02)",
                        "type": "qd",
                        "start": 0,
                        "phase": 0
                    },
                    "MZ": {
                        "duration": 620,
                        "amplitude": 0.003575,
                        "frequency": 7453265000,
                        "shape": "Rectangular()",
                        "type": "ro",
                        "start": 0,
                        "phase": 0
                    }
                },
                "1": {
                    "RX": {
                        "duration": 40,
                        "amplitude": 0.05682,
                        "frequency": 5800563000,
                        "shape": "Drag(5, -0.04)",
                        "type": "qd",
                        "start": 0,
                        "phase": 0
                    },
                    "MZ": {
                        "duration": 960,
                        "amplitude": 0.00325,
                        "frequency": 7655107000,
                        "shape": "Rectangular()",
                        "type": "ro",
                        "start": 0,
                        "phase": 0
                    }
                }
            },
            "two_qubit": {
                "0-1": {
                    "CZ": [
                        {
                            "duration": 30,
                            "amplitude": 0.055,
                            "shape": "Rectangular()",
                            "qubit": 1,
                            "relative_start": 0,
                            "type": "qf"
                        },
                        {
                            "type": "virtual_z",
                            "phase": -1.5707963267948966,
                            "qubit": 0
                        },
                        {
                            "type": "virtual_z",
                            "phase": -1.5707963267948966,
                            "qubit": 1
                        }
                    ]
                }
            }
        },
        "characterization": {
            "single_qubit": {
                "0": {
                    "readout_frequency": 7453265000,
                    "drive_frequency": 4855663000,
                    "T1": 0.0,
                    "T2": 0.0,
                    "sweetspot": -0.047,
                    "threshold": 0.00028502261712637096,
                    "iq_angle": 1.283105298787488
                },
                "1": {
                    "readout_frequency": 7655107000,
                    "drive_frequency": 5800563000,
                    "T1": 0.0,
                    "T2": 0.0,
                    "sweetspot": -0.045,
                    "threshold": 0.0002694329123116206,
                    "iq_angle": 4.912447775569025
                }
            }
        }
    }

And in the case of having a chip with coupler qubits
we need the following changes to the previous runcard:

.. code-block::  json

    {
        "qubits": [
            0,
            1
        ],
        "couplers": [
            0
        ],
        "topology": {
            "0": [
                0,
                1
            ]
        },
        "native_gates": {
            "two_qubit": {
                "0-1": {
                    "CZ": [
                        {
                            "duration": 30,
                            "amplitude": 0.6025,
                            "shape": "Rectangular()",
                            "qubit": 1,
                            "relative_start": 0,
                            "type": "qf"
                        },
                        {
                            "type": "virtual_z",
                            "phase": -1,
                            "qubit": 0
                        },
                        {
                            "type": "virtual_z",
                            "phase": -3,
                            "qubit": 1
                        },
                        {
                            "type": "coupler",
                            "duration": 40,
                            "amplitude": 0.1,
                            "shape": "Rectangular()",
                            "coupler": 0,
                            "relative_start": 0
                        }
                    ]
                }
            }
        },
        "characterization": {
            "coupler": {
                "0": {
                    "sweetspot": 0.0
                }
            }
        }
    }

This file contains different sections: ``qubits`` is a list with the qubit
names, ``couplers`` one with the coupler names , ``settings`` defines default execution parameters, ``topology`` defines
the qubit connectivity (qubit pairs), ``native_gates`` specifies the calibrated
pulse parameters for implementing single and two-qubit gates and
``characterization`` provides the physical parameters associated to each qubit and coupler.
Note that such parameters may slightly differ depending on the QPU architecture,
however the pulses under ``native_gates`` should comply with the
:class:`qibolab.pulses.Pulse` API and the parameters under ``characterization``
should be a subset of :class:`qibolab.qubits.Qubit` attributes.

Providing the above runcard is not sufficient to instantiate a
:class:`qibolab.platform.Platform`. This should still be done using a
``create()`` method, however this is significantly simplified by
``qibolab.serialize``. The ``create()`` method should be put in a
file named ``platform.py`` inside the ``my_platform`` directory.
Here is the ``create()`` method that loads the parameters of
the above runcard:

.. testcode::  python

    # my_platform / platform.py

    from pathlib import Path
    from qibolab import Platform
    from qibolab.channels import ChannelMap, Channel
    from qibolab.serialize import load_runcard, load_qubits, load_settings
    from qibolab.instruments.dummy import DummyInstrument

    FOLDER = Path.cwd()
    # assumes runcard is storred in the same folder as platform.py


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # Create channel objects and assign to them the controller ports
        channels = ChannelMap()
        channels |= Channel("ch1out", port=instrument["o1"])
        channels |= Channel("ch1in", port=instrument["i1"])
        channels |= Channel("ch2", port=instrument["o2"])
        channels |= Channel("ch3", port=instrument["o3"])
        channels |= Channel("chf1", port=instrument["o4"])
        channels |= Channel("chf2", port=instrument["o5"])

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(folder)
        qubits, couplers, pairs = load_qubits(runcard)

        # assign channels to the qubit
        for q in range(2):
            qubits[q].readout = channels["ch1out"]
            qubits[q].feedback = channels["ch1in"]
            qubits[q].drive = channels[f"ch{q + 2}"]
            qubits[q].flux = channels[f"chf{q + 1}"]

        # create dictionary of instruments
        instruments = {instrument.name: instrument}
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform", qubits, pairs, instruments, settings, resonator_type="2D"
        )

With the following additions for coupler architectures:

.. testcode::  python

    # my_platform / platform.py


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # Create channel objects and assign to them the controller ports
        channels = ChannelMap()
        channels |= Channel("ch1out", port=instrument["o1"])
        channels |= Channel("ch1in", port=instrument["i1"])
        channels |= Channel("ch2", port=instrument["o2"])
        channels |= Channel("ch3", port=instrument["o3"])
        channels |= Channel("chf1", port=instrument["o4"])
        channels |= Channel("chf2", port=instrument["o5"])
        channels |= Channel("chfc0", port=instrument["o6"])

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(FOLDER)
        qubits, couplers, pairs = load_qubits(runcard)

        # assign channels to the qubit
        for q in range(2):
            qubits[q].readout = channels["ch1out"]
            qubits[q].feedback = channels["ch1in"]
            qubits[q].drive = channels[f"ch{q + 2}"]
            qubits[q].flux = channels[f"chf{q + 1}"]

        # assign channels to the coupler
        couplers[0].flux = channels["chfc0"]

        # create dictionary of instruments
        instruments = {instrument.name: instrument}
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform",
            qubits,
            pairs,
            instruments,
            settings,
            resonator_type="2D",
            couplers=couplers,
        )

Note that this assumes that the runcard is saved as ``<folder>/parameters.yml`` where ``<folder>``
is the directory containing ``platform.py``.


Instrument settings
^^^^^^^^^^^^^^^^^^^

The runcard of the previous example contains only parameters associated to the qubits
and their respective native gates. In some cases parameters associated to instruments
need to also be calibrated. An example is the frequency and the power of local oscillators,
such as the one used to pump a traveling wave parametric amplifier (TWPA).

The runcard can contain an ``instruments`` section that provides these parameters

.. code-block::  json

    {
        "nqubits": 2,
        "qubits": [
            0,
            1
        ],
        "settings": {
            "nshots": 1024,
            "sampling_rate": 1000000000,
            "relaxation_time": 50000
        },
        "topology": [
            [
                0,
                1
            ]
        ],
        "instruments": {
            "twpa_pump": {
                "frequency": 4600000000,
                "power": 5
            }
        },
        "native_gates": {
            "single_qubit": {
                "0": {
                    "RX": {
                        "duration": 40,
                        "amplitude": 0.0484,
                        "frequency": 4855663000,
                        "shape": "Drag(5, -0.02)",
                        "type": "qd",
                        "start": 0,
                        "phase": 0
                    },
                    "MZ": {
                        "duration": 620,
                        "amplitude": 0.003575,
                        "frequency": 7453265000,
                        "shape": "Rectangular()",
                        "type": "ro",
                        "start": 0,
                        "phase": 0
                    }
                },
                "1": {
                    "RX": {
                        "duration": 40,
                        "amplitude": 0.05682,
                        "frequency": 5800563000,
                        "shape": "Drag(5, -0.04)",
                        "type": "qd",
                        "start": 0,
                        "phase": 0
                    },
                    "MZ": {
                        "duration": 960,
                        "amplitude": 0.00325,
                        "frequency": 7655107000,
                        "shape": "Rectangular()",
                        "type": "ro",
                        "start": 0,
                        "phase": 0
                    }
                }
            },
            "two_qubit": {
                "0-1": {
                    "CZ": [
                        {
                            "duration": 30,
                            "amplitude": 0.055,
                            "shape": "Rectangular()",
                            "qubit": 1,
                            "relative_start": 0,
                            "type": "qf"
                        },
                        {
                            "type": "virtual_z",
                            "phase": -1.5707963267948966,
                            "qubit": 0
                        },
                        {
                            "type": "virtual_z",
                            "phase": -1.5707963267948966,
                            "qubit": 1
                        }
                    ]
                }
            }
        },
        "characterization": {
            "single_qubit": {
                "0": {
                    "readout_frequency": 7453265000,
                    "drive_frequency": 4855663000,
                    "T1": 0.0,
                    "T2": 0.0,
                    "sweetspot": -0.047,
                    "threshold": 0.00028502261712637096,
                    "iq_angle": 1.283105298787488
                },
                "1": {
                    "readout_frequency": 7655107000,
                    "drive_frequency": 5800563000,
                    "T1": 0.0,
                    "T2": 0.0,
                    "sweetspot": -0.045,
                    "threshold": 0.0002694329123116206,
                    "iq_angle": 4.912447775569025
                }
            }
        }
    }


These settings are loaded when creating the platform using :meth:`qibolab.serialize.load_instrument_settings`.
Note that the key used in the runcard should be the same with the name used when instantiating the instrument,
in this case ``"twpa_pump"``.

.. testcode::  python

    # my_platform / platform.py

    from pathlib import Path
    from qibolab import Platform
    from qibolab.channels import ChannelMap, Channel
    from qibolab.serialize import (
        load_runcard,
        load_qubits,
        load_settings,
        load_instrument_settings,
    )
    from qibolab.instruments.dummy import DummyInstrument
    from qibolab.instruments.oscillator import LocalOscillator

    FOLDER = Path.cwd()


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")
        twpa = LocalOscillator("twpa_pump", "0.0.0.1")

        # Create channel objects and assign to them the controller ports
        channels = ChannelMap()
        channels |= Channel("ch1out", port=instrument["o1"])
        channels |= Channel("ch2", port=instrument["o2"])
        channels |= Channel("ch3", port=instrument["o3"])
        channels |= Channel("ch1in", port=instrument["i1"])

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(FOLDER)
        qubits, pairs = load_qubits(runcard)

        # assign channels to the qubit
        for q in range(2):
            qubits[q].readout = channels["ch1out"]
            qubits[q].feedback = channels["ch1in"]
            qubits[q].drive = channels[f"ch{q + 2}"]

        # create dictionary of instruments
        instruments = {instrument.name: instrument, twpa.name: twpa}
        # load instrument settings from the runcard
        instruments = load_instrument_settings(runcard, instruments)
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform", qubits, pairs, instruments, settings, resonator_type="2D"
        )
