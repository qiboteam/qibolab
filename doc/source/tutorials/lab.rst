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

.. testcode::  python

    from qibolab import Platform
    from qibolab.components import IqChannel, AcquireChannel, IqConfig
    from qibolab.qubits import Qubit
    from qibolab.pulses import Gaussian, Pulse, Rectangular
    from qibolab.native import RxyFactory, FixedSequenceFactory, SingleQubitNatives
    from qibolab.instruments.dummy import DummyInstrument


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # create the qubit object
        qubit = Qubit(0)

        # assign channels to the qubit
        qubit.probe = IqChannel(name="probe", mixer=None, lo=None, acquisition="acquire")
        qubit.acquire = AcquireChannel(name="acquire", twpa_pump=None, probe="probe")
        qubit.drive = Iqchannel(name="drive", mixer=None, lo=None)

        # define configuration for channels
        configs = {}
        configs[qubit.drive.name] = IqConfig(frequency=3e9)
        configs[qubit.probe.name] = IqConfig(frequency=7e9)

        # create sequence that drives qubit from state 0 to 1
        drive_seq = PulseSequence()
        drive_seq[qubit.drive.name].append(
            Pulse(
                duration=40,
                amplitude=0.05,
                envelope=Gaussian(rel_sigma=0.2),
            )
        )

        # create sequence that can be used for measuring the qubit
        probe_seq = PulseSequence()
        probe_seq[qubit.probe.name].append(
            Pulse(
                duration=1000,
                amplitude=0.005,
                envelope=Rectangular(),
            )
        )

        # assign native gates to the qubit
        qubit.native_gates = SingleQubitNatives(
            RX=RxyFactory(drive_seq),
            MZ=FixedSequenceFactory(probe_seq),
        )

        # create dictionaries of the different objects
        qubits = {qubit.name: qubit}
        pairs = {}  # empty as for single qubit we have no qubit pairs
        instruments = {instrument.name: instrument}

        # allocate and return Platform object
        return Platform(
            "my_platform", qubits, pairs, configs, instruments, resonator_type="3D"
        )


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

.. testcode::  python

    from qibolab.components import IqChannel, AcquireChannel, DcChannel, IqConfig
    from qibolab.qubits import Qubit, QubitPair
    from qibolab.pulses import Gaussian, Pulse, PulseSequence, Rectangular
    from qibolab.native import (
        RxyFactory,
        FixedSequenceFactory,
        SingleQubitNatives,
        TwoQubitNatives,
    )

    # create the qubit objects
    qubit0 = Qubit(0)
    qubit1 = Qubit(1)

    # assign channels to the qubits
    qubit0.probe = IqChannel(name="probe_0", mixer=None, lo=None, acquisition="acquire_0")
    qubit0.acquire = AcquireChannel(name="acquire_0", twpa_pump=None, probe="probe_0")
    qubit0.drive = IqChannel(name="drive_0", mixer=None, lo=None)
    qubit0.flux = DcChannel(name="flux_0")
    qubit1.probe = IqChannel(name="probe_1", mixer=None, lo=None, acquisition="acquire_1")
    qubit1.acquire = AcquireChannel(name="acquire_1", twpa_pump=None, probe="probe_1")
    qubit1.drive = IqChannel(name="drive_1", mixer=None, lo=None)

    # assign single-qubit native gates to each qubit
    qubit0.native_gates = SingleQubitNatives(
        RX=RxyFactory(
            PulseSequence(
                {
                    qubit0.drive.name: [
                        Pulse(
                            duration=40,
                            amplitude=0.05,
                            envelope=Gaussian(rel_sigma=0.2),
                        )
                    ]
                }
            )
        ),
        MZ=FixedSequenceFactory(
            PulseSequence(
                {
                    qubit0.probe.name: [
                        Pulse(
                            duration=1000,
                            amplitude=0.005,
                            envelope=Rectangular(),
                        )
                    ]
                }
            )
        ),
    )
    qubit1.native_gates = SingleQubitNatives(
        RX=RxyFactory(
            PulseSequence(
                {
                    qubit1.drive.name: [
                        Pulse(
                            duration=40,
                            amplitude=0.05,
                            envelope=Gaussian(rel_sigma=0.2),
                        )
                    ]
                }
            )
        ),
        MZ=FixedSequenceFactory(
            PulseSequence(
                {
                    qubit1.probe.name: [
                        Pulse(
                            duration=1000,
                            amplitude=0.005,
                            envelope=Rectangular(),
                        )
                    ]
                }
            )
        ),
    )

    # define the pair of qubits
    pair = QubitPair(qubit0, qubit1)
    pair.native_gates = TwoQubitNatives(
        CZ=FixedSequenceFactory(
            PulseSequence(
                {
                    qubit0.flux.name: [
                        Pulse(
                            duration=30,
                            amplitude=0.005,
                            envelope=Rectangular(),
                        )
                    ],
                }
            )
        )
    )

Some architectures may also have coupler qubits that mediate the interactions.
We can also interact with them defining the :class:`qibolab.couplers.Coupler` objects.
Then we add them to their corresponding :class:`qibolab.qubits.QubitPair` objects according
to the chip topology. We neglected characterization parameters associated to the
coupler but qibolab will take them into account when calling :class:`qibolab.native.TwoQubitNatives`.


.. testcode::  python

    from qibolab.components import DcChannel
    from qibolab.couplers import Coupler
    from qibolab.qubits import Qubit, QubitPair
    from qibolab.pulses import Pulse, PulseSequence
    from qibolab.native import (
        FixedSequenceFactory,
        SingleQubitNatives,
        TwoQubitNatives,
    )

    # create the qubit and coupler objects
    qubit0 = Qubit(0)
    qubit1 = Qubit(1)
    coupler_01 = Coupler(0)

    # assign channel(s) to the coupler
    coupler_01.flux = DcChannel(name="flux_coupler_01")

    # assign single-qubit native gates to each qubit
    # Look above example

    # define the pair of qubits
    pair = QubitPair(qubit0, qubit1, coupler_01)
    pair.native_gates = TwoQubitNatives(
        CZ=FixedSequenceFactory(
            PulseSequence(
                {
                    coupler_01.flux.name: [
                        Pulse(
                            duration=30,
                            amplitude=0.005,
                            frequency=1e9,
                            envelope=Rectangular(),
                            qubit=qubit1.name,
                        )
                    ]
                },
            )
        )
    )

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
		"components": {
			"drive_0": {
				"frequency": 4855663000
			},
			"drive_1": {
				"frequency": 5800563000
			},
			"flux_0": {
				"bias": 0.0
			},
			"probe_0": {
				"frequency": 7453265000
			},
			"probe_1": {
				"frequency": 7655107000
			},
			"acquire_0": {
			  "delay": 0,
			  "smearing": 0
			},
			"acquire_1": {
			  "delay": 0,
			  "smearing": 0
			}
		}
        "native_gates": {
            "single_qubit": {
                "0": {
                    "RX": {
						"drive_0": [
							{
								"duration": 40,
								"amplitude": 0.0484,
								"envelope": {
									"kind": "drag",
									"rel_sigma": 0.2,
									"beta": -0.02,
								},
								"type": "qd",
							}
						]
					},
                    "MZ": {
						"probe_0": [
							{
							"duration": 620,
							"amplitude": 0.003575,
							"envelope": {"kind": "rectangular"},
							"type": "ro",
							}
						]
					}
                },
                "1": {
                    "RX": {
						"drive_1" : [
							{
							"duration": 40,
							"amplitude": 0.05682,
							"envelope": {
								"kind": "drag",
								"rel_sigma": 0.2,
								"beta": -0.04,
							},
							"type": "qd",
							}
						]
					},
                    "MZ": {
						"probe_1": [
							{
							"duration": 960,
							"amplitude": 0.00325,
							"envelope": {"kind": "rectangular"},
							"type": "ro",
							}
						]
					}
                }
            },
            "two_qubit": {
                "0-1": {
                    "CZ": [
                        {
                            "duration": 30,
                            "amplitude": 0.055,
                            "envelope": {"kind": "rectangular"},
                            "qubit": 1,
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
                    "T1": 0.0,
                    "T2": 0.0,
                    "threshold": 0.00028502261712637096,
                    "iq_angle": 1.283105298787488
                },
                "1": {
                    "T1": 0.0,
                    "T2": 0.0,
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
		"components": {
			"flux_coupler_01": {
				"bias": 0.12
			}
		}
        "native_gates": {
            "two_qubit": {
                "0-1": {
					"CZZ": {
						"flux_coupler_01": [
							{
								"type": "cf",
								"duration": 40,
								"amplitude": 0.1,
								"envelope": {"kind": "rectangular"},
								"coupler": 0,
							}
						]
						"flux_0": [
							{
								"duration": 30,
								"amplitude": 0.6025,
								"envelope": {"kind": "rectangular"},
								"type": "qf"
							}
						],
						"drive_0": [
							{
								"type": "virtual_z",
								"phase": -1,
								"qubit": 0
							}
						],
						"drive_1": [
							{
								"type": "virtual_z",
								"phase": -3,
								"qubit": 1
							}
						]
					}
                    "CZ": [

                    ]
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
    from qibolab.components import (
        AcquireChannel,
        DcChannel,
        IqChannel,
        AcquisitionConfig,
        DcConfig,
        IqConfig,
    )
    from qibolab.serialize import (
        load_runcard,
        load_qubits,
        load_settings,
    )
    from qibolab.instruments.dummy import DummyInstrument

    FOLDER = Path.cwd()
    # assumes runcard is storred in the same folder as platform.py


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(folder)
        qubits, _, pairs = load_qubits(runcard)

        # define channels and load component configs
        configs = {}
        component_params = runcard["components"]
        for q in range(2):
            drive_name = f"qubit_{q}/drive"
            configs[drive_name] = IqConfig(**component_params[drive_name])
            qubits[q].drive = IqChannel(drive_name, mixer=None, lo=None)

            flux_name = f"qubit_{q}/flux"
            configs[flux_name] = DcConfig(**component_params[flux_name])
            qubits[q].flux = DcChannel(flux_name)

            probe_name, acquire_name = f"qubit_{q}/probe", f"qubit_{q}/acquire"
            configs[probe_name] = IqConfig(**component_params[probe_name])
            qubits[q].probe = IqChannel(
                probe_name, mixer=None, lo=None, acquistion=acquire_name
            )

            configs[acquire_name] = AcquisitionConfig(**component_params[acquire_name])
            quibts[q].acquisition = AcquireChannel(
                acquire_name, twpa_pump=None, probe=probe_name
            )

        # create dictionary of instruments
        instruments = {instrument.name: instrument}
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform",
            qubits,
            pairs,
            configs,
            instruments,
            settings,
            resonator_type="2D",
        )

With the following additions for coupler architectures:

.. testcode::  python

    # my_platform / platform.py


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(folder)
        qubits, couplers, pairs = load_qubits(runcard)

        # define channels and load component configs
        configs = {}
        component_params = runcard["components"]
        for q in range(2):
            drive_name = f"qubit_{q}/drive"
            configs[drive_name] = IqConfig(**component_params[drive_name])
            qubits[q].drive = IqChannel(drive_name, mixer=None, lo=None)

            flux_name = f"qubit_{q}/flux"
            configs[flux_name] = DcConfig(**component_params[flux_name])
            qubits[q].flux = DcChannel(flux_name)

            probe_name, acquire_name = f"qubit_{q}/probe", f"qubit_{q}/acquire"
            configs[probe_name] = IqConfig(**component_params[probe_name])
            qubits[q].probe = IqChannel(
                probe_name, mixer=None, lo=None, acquistion=acquire_name
            )

            configs[acquire_name] = AcquisitionConfig(**component_params[acquire_name])
            quibts[q].acquisition = AcquireChannel(
                acquire_name, twpa_pump=None, probe=probe_name
            )

        coupler_flux_name = "coupler_0/flux"
        configs[coupler_flux_name] = DcConfig(**component_params[coupler_flux_name])
        couplers[0].flux = DcChannel(coupler_flux_name)

        # create dictionary of instruments
        instruments = {instrument.name: instrument}
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform",
            qubits,
            pairs,
            configs,
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
            "single_qubit": {},
            "two_qubit": {}
        },
        "characterization": {
            "single_qubit": {
                "0": {
                    "T1": 0.0,
                    "T2": 0.0,
                    "threshold": 0.00028502261712637096,
                    "iq_angle": 1.283105298787488
                },
                "1": {
                    "T1": 0.0,
                    "T2": 0.0,
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
    from qibolab.components import (
        AcquireChannel,
        DcChannel,
        IqChannel,
        AcquisitionConfig,
        DcConfig,
        IqConfig,
    )
    from qibolab.serialize import (
        load_runcard,
        load_qubits,
        load_settings,
    )
    from qibolab.instruments.dummy import DummyInstrument

    FOLDER = Path.cwd()
    # assumes runcard is storred in the same folder as platform.py


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(folder)
        qubits, _, pairs = load_qubits(runcard)

        # define channels and load component configs
        configs = {}
        component_params = runcard["components"]
        for q in range(2):
            drive_name = f"qubit_{q}/drive"
            configs[drive_name] = IqConfig(**component_params[drive_name])
            qubits[q].drive = IqChannel(drive_name, mixer=None, lo=None)

            flux_name = f"qubit_{q}/flux"
            configs[flux_name] = DcConfig(**component_params[flux_name])
            qubits[q].flux = DcChannel(flux_name)

            probe_name, acquire_name = f"qubit_{q}/probe", f"qubit_{q}/acquire"
            configs[probe_name] = IqConfig(**component_params[probe_name])
            qubits[q].probe = IqChannel(
                probe_name, mixer=None, lo=None, acquistion=acquire_name
            )

            configs[acquire_name] = AcquisitionConfig(**component_params[acquire_name])
            quibts[q].acquisition = AcquireChannel(
                acquire_name, twpa_pump=None, probe=probe_name
            )

        # create dictionary of instruments
        instruments = {instrument.name: instrument}
        # load instrument settings from the runcard
        instruments = load_instrument_settings(runcard, instruments)
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform",
            qubits,
            pairs,
            configs,
            instruments,
            settings,
            resonator_type="2D",
        )
