How to connect Qibolab to your lab?
===================================

In this section we will show how to let Qibolab communicate with your lab's
instruments and run an experiment.

The main required object, in this case, is the :class:`qibolab.Platform`.
A Platform is defined as a QPU (quantum processing unit with one or more qubits)
controlled by one ore more instruments.

How to define a platform for a self-hosted QPU?
-----------------------------------------------

The :class:`qibolab.Platform` object holds all the information required
to execute programs, and in particular :class:`qibolab.PulseSequence` in
a real QPU. It is comprised by different objects that contain information about
the native gates and the lab's instrumentation.

The following cell shows how to define a single qubit platform from scratch,
using different Qibolab primitives.

.. testcode::  python

    from qibolab import (
        Acquisition,
        AcquisitionChannel,
        Gaussian,
        IqChannel,
        Platform,
        PulseSequence,
        Pulse,
        Qubit,
        Readout,
        Rectangular,
    )
    from qibolab._core.components.configs import AcquisitionConfig, IqConfig
    from qibolab._core.native import FixedSequenceFactory, RxyFactory
    from qibolab._core.parameters import NativeGates, Parameters, SingleQubitNatives
    from qibolab.instruments import DummyInstrument


    def create():
        # Create the qubit objects
        qubits = {
            0: Qubit(
                drive="0/drive",
                probe="0/probe",
                acquisition="0/acquisition",
            )
        }

        # Create channels and connect to instrument ports
        channels = {}
        qubit = qubits[0]
        channels[qubit.probe] = IqChannel(
            device="controller",
            path="0",
            mixer=None,
            lo=None,
        )
        channels[qubit.acquisition] = AcquisitionChannel(
            device="controller", path="0", twpa_pump=None, probe="probe"
        )
        channels[qubit.drive] = IqChannel(
            device="controller", path="0", mixer=None, lo=None
        )

        # define configuration for channels
        configs = {}
        configs[qubit.drive] = IqConfig(frequency=3e9)
        configs[qubit.probe] = IqConfig(frequency=7e9)
        configs[qubit.acquisition] = AcquisitionConfig(delay=200.0, smearing=0.0)

        # create sequence that drives qubit from state 0 to 1
        drive_seq = PulseSequence(
            [
                (
                    qubit.drive,
                    Pulse(duration=40, amplitude=0.05, envelope=Gaussian(rel_sigma=0.2)),
                )
            ]
        )

        # create sequence that can be used for measuring the qubit
        measurement_seq = PulseSequence(
            [
                (
                    qubit.acquisition,
                    Readout(
                        acquisition=Acquisition(duration=1000),
                        probe=Pulse(duration=1000, amplitude=0.005, envelope=Rectangular()),
                    ),
                )
            ]
        )

        # assign native gates to the qubit
        native_gates = SingleQubitNatives(
            RX=RxyFactory(drive_seq),
            MZ=FixedSequenceFactory(measurement_seq),
        )

        # create a parameters instance
        parameters = Parameters(
            configs=configs,
            native_gates=NativeGates(single_qubit={0: native_gates}),
        )

        # Create a controller instrument
        instruments = {
            "my_instrument": DummyInstrument(
                name="my_instrument",
                address="0.0.0.0:0",
                channels=channels,
            )
        }

        # allocate and return Platform object
        return Platform("my_platform", parameters, instruments, qubits)


This code creates a platform with a single qubit that is controlled by the
:class:`qibolab.instruments.DummyInstrument`. In real applications, if
Qibolab provides drivers for the instruments in the lab, these can be directly
used in place of the ``DummyInstrument`` above, otherwise new drivers need to be
coded following the abstract :class:`qibolab._core.instruments.abstract.Instrument`
interface.

Furthermore, above we defined three channels that connect the qubit to the
control instrument and we assigned two native gates to the qubit.

When the QPU contains more than one qubit, some of the qubits are connected so
that two-qubit gates can be applied. These are called in a single dictionary, within
the native gates, but separately from the single-qubit ones.

.. testcode::  python

    from qibolab import (
        Acquisition,
        AcquisitionChannel,
        DcChannel,
        Gaussian,
        IqChannel,
        Pulse,
        PulseSequence,
        Qubit,
        Readout,
        Rectangular,
    )
    from qibolab._core.components.configs import AcquisitionConfig, IqConfig
    from qibolab._core.native import FixedSequenceFactory, RxyFactory
    from qibolab._core.parameters import (
        NativeGates,
        Parameters,
        SingleQubitNatives,
        TwoQubitContainer,
        TwoQubitNatives,
    )

    # Create the qubit objects
    qubits = {
        0: Qubit(
            drive="0/drive",
            flux="0/flux",
            probe="0/probe",
            acquisition="0/acquisition",
        ),
        1: Qubit(
            drive="1/drive",
            flux="1/flux",
            probe="1/probe",
            acquisition="1/acquisition",
        ),
    }

    # Create channels and connect to instrument ports
    channels = {}
    channels[qubits[0].probe] = IqChannel(
        device="controller",
        path="0",
        mixer=None,
        lo=None,
    )
    channels[qubits[0].acquisition] = AcquisitionChannel(
        device="controller", path="0", twpa_pump=None, probe="probe"
    )
    channels[qubits[0].drive] = IqChannel(
        device="controller", path="1", mixer=None, lo=None
    )
    channels[qubits[0].flux] = DcChannel(device="controller", path="2")

    channels[qubits[1].probe] = IqChannel(
        device="controller",
        path="3",
        mixer=None,
        lo=None,
    )
    channels[qubits[1].acquisition] = AcquisitionChannel(
        device="controller", path="3", twpa_pump=None, probe="probe"
    )
    channels[qubits[1].drive] = IqChannel(
        device="controller", path="4", mixer=None, lo=None
    )
    channels[qubits[1].flux] = DcChannel(device="controller", path="5")

    # define configuration for channels
    configs = {}
    configs[qubits[0].drive] = IqConfig(frequency=3e9)
    configs[qubits[0].probe] = IqConfig(frequency=7e9)
    configs[qubits[0].acquisition] = AcquisitionConfig(delay=200.0, smearing=0.0)

    # create native gates
    rx0 = PulseSequence(
        [
            (
                qubits[0].drive,
                Pulse(duration=40, amplitude=0.05, envelope=Gaussian(rel_sigma=0.2)),
            )
        ]
    )
    mz0 = PulseSequence(
        [
            (
                qubits[0].acquisition,
                Readout(
                    acquisition=Acquisition(duration=1000),
                    probe=Pulse(duration=1000, amplitude=0.005, envelope=Rectangular()),
                ),
            )
        ]
    )
    rx1 = PulseSequence(
        [
            (
                qubits[1].drive,
                Pulse(duration=40, amplitude=0.05, envelope=Gaussian(rel_sigma=0.2)),
            )
        ]
    )
    mz1 = PulseSequence(
        [
            (
                qubits[1].acquisition,
                Readout(
                    acquisition=Acquisition(duration=1000),
                    probe=Pulse(duration=1000, amplitude=0.005, envelope=Rectangular()),
                ),
            )
        ]
    )
    cz01 = PulseSequence(
        [
            (
                qubits[0].flux,
                Pulse(duration=30, amplitude=0.005, envelope=Rectangular()),
            ),
        ]
    )
    native_gates = NativeGates(
        single_qubit={
            0: SingleQubitNatives(
                RX=RxyFactory(rx0),
                MZ=FixedSequenceFactory(mz0),
            ),
            1: SingleQubitNatives(
                RX=RxyFactory(rx1),
                MZ=FixedSequenceFactory(mz1),
            ),
        },
        two_qubit=TwoQubitContainer(
            {"0-1": TwoQubitNatives(CZ=FixedSequenceFactory(cz01))}
        ),
    )

    # create a parameters instance
    parameters = Parameters(
        configs=configs,
        native_gates=native_gates,
    )

Some architectures may also have coupler qubits that mediate the interactions.
We neglected characterization parameters associated to the coupler but qibolab
will take them into account when calling :class:`qibolab.native.TwoQubitNatives`.


.. testcode::  python

    from qibolab import (
        DcChannel,
        Pulse,
        PulseSequence,
        Qubit,
        Rectangular,
    )
    from qibolab._core.parameters import TwoQubitContainer, TwoQubitNatives
    from qibolab._core.native import FixedSequenceFactory

    # create the qubit and coupler objects
    coupler_01 = Qubit(flux="c01/flux")

    channels = {}
    # assign channel(s) to the coupler
    channels[coupler_01.flux] = DcChannel(device="controller", path="5")

    # assign single-qubit native gates to each qubit
    # Look above example

    # define the pair of qubits
    two_qubit = TwoQubitContainer(
        {
            "0-1": TwoQubitNatives(
                CZ=FixedSequenceFactory(
                    PulseSequence(
                        [
                            (
                                coupler_01.flux,
                                Pulse(duration=30, amplitude=0.005, envelope=Rectangular()),
                            )
                        ],
                    )
                )
            ),
        }
    )

Couplers also need to be passed in a different dictionary than the qubits,
when instantiating the :class:`qibolab.Platform`

.. note::

    The platform automatically creates the connectivity graph of the given chip,
    using the keys of :class:`qibolab._core.parameters.TwoQubitContainer` map.


Registering platforms
^^^^^^^^^^^^^^^^^^^^^

The ``create()`` function defined in the above example can be called or imported
directly in any Python script. Alternatively, it is also possible to make the
platform available as

.. code-block::  python

    from qibolab import create_platform

    platform = create_platform("my_platform")


To do so, ``create()`` needs to be saved in a module called ``platform.py`` inside
a folder with the name of this platform (in this case ``my_platform``).
Moreover, the environment flag ``QIBOLAB_PLATFORMS`` needs to point to the directory
that contains this folder.
Examples of advanced platforms are available at `this
repository <https://github.com/qiboteam/qibolab_platforms_qrc>`_.

.. _parameters_json:

Loading platform parameters from JSON
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operating a QPU requires calibrating a set of parameters, the number of which
increases with the number of qubits. Hardcoding such parameters in the
``create()`` function, as shown in the above examples, is not scalable. However,
since ``create()`` is part of a Python module, is is possible to load parameters
from an external file or database.

Qibolab provides some utility functions, accessible through
:py:mod:`qibolab._core.parameters`, for loading calibration parameters stored in a JSON
file with a specific format. Here is an example

.. code-block::  json

    {
      "settings": {
        "nshots": 1024,
        "relaxation_time": 50000
      },
      "configs": {
        "0/drive": {
          "kind": "iq",
          "frequency": 4855663000
        },
        "1/drive": {
          "kind": "iq",
          "frequency": 5800563000
        },
        "0/flux": {
          "kind": "dc",
          "offset": 0.0
        },
        "1/flux": {
          "kind": "dc",
          "offset": 0.0
        },
        "0/probe": {
          "kind": "iq",
          "frequency": 7453265000
        },
        "1/probe": {
          "kind": "iq",
          "frequency": 7655107000
        },
        "0/acquisition": {
          "kind": "acquisition",
          "delay": 0,
          "smearing": 0
        },
        "1/acquisition": {
          "kind": "acquisition",
          "delay": 0,
          "smearing": 0
        },
        "01/coupler": {
          "kind": "dc",
          "offset": 0.12
        }
      },
      "native_gates": {
        "single_qubit": {
          "0": {
            "RX": [
              [
                "0/drive",
                {
                  "kind": "pulse",
                  "duration": 40,
                  "amplitude": 0.0484,
                  "envelope": {
                    "kind": "drag",
                    "rel_sigma": 0.2,
                    "beta": -0.02
                  }
                }
              ]
            ],
            "MZ": [
                [
                  "0/acquisition",
                  {
                      "kind": "readout",
                      "acquisition": {
                          "kind": "acquisition",
                          "duration": 620.0
                      },
                      "probe": {
                          "kind": "pulse",
                          "duration": 620.0,
                          "amplitude": 0.003575,
                          "envelope": {
                              "kind": "rectangular"
                          }
                      }
                  }
              ]
            ]
          },
          "1": {
            "RX": [
              [
                "1/drive",
                {
                  "kind": "pulse",
                  "duration": 40,
                  "amplitude": 0.05682,
                  "envelope": {
                    "kind": "drag",
                    "rel_sigma": 0.2,
                    "beta": -0.04
                  }
                }
              ]
            ],
            "MZ": [
              [
                "1/acquisition",
                {
                    "kind": "readout",
                    "acquisition": {
                        "kind": "acquisition",
                        "duration": 960.0
                    },
                    "probe": {
                        "kind": "pulse",
                        "duration": 960.0,
                        "amplitude": 0.00325,
                        "envelope": {
                            "kind": "rectangular"
                        }
                    }
                }
              ]
            ]
          }
        },
        "two_qubit": {
          "0-1": {
            "CZ": [
              [
                "01/coupler",
                {
                  "kind": "pulse",
                  "duration": 40,
                  "amplitude": 0.1,
                  "envelope": {
                    "kind": "rectangular"
                  }
                }
              ],
              [
                "0/flux",
                {
                  "kind": "pulse",
                  "duration": 30,
                  "amplitude": 0.6025,
                  "envelope": {
                    "kind": "rectangular"
                  }
                }
              ],
              [
                "0/drive",
                {
                  "kind": "virtualz",
                  "phase": -1
                }
              ],
              [
                "1/drive",
                {
                  "kind": "virtualz",
                  "phase": -3
                }
              ]
            ]
          }
        }
      }
    }

This file contains different sections: ``configs`` defines the default configuration of channel
parameters, while ``native_gates`` specifies the calibrated pulse parameters for implementing
single and two-qubit gates.
Note that such parameters may slightly differ depending on the QPU architecture.

Providing the above JSON is not sufficient to instantiate a
:class:`qibolab.Platform`. This should still be done using a
``create()`` method. The ``create()`` method should be put in a
file named ``platform.py`` inside the ``my_platform`` directory.
Here is the ``create()`` method that loads the parameters from the JSON:

.. testcode::  python

    # my_platform / platform.py

    from pathlib import Path
    from qibolab import (
        AcquisitionChannel,
        DcChannel,
        IqChannel,
        Platform,
        Qubit,
    )
    from qibolab.instruments import DummyInstrument


    FOLDER = Path.cwd()


    def create():
        qubits = {}
        for q in range(2):
            qubits[q] = Qubit(
                drive=f"{q}/drive",
                flux=f"{q}/flux",
                probe=f"{q}/probe",
                acquisition=f"{q}/acquisition",
            )

        couplers = {0: Qubit(flux="01/coupler")}

        channels = {}
        for q in range(2):
            channels[qubits[q].drive] = IqChannel(
                device="my_instrument", path="1", mixer=None, lo=None
            )
            channels[qubits[q].flux] = DcChannel(device="my_instrument", path="2")
            channels[qubits[q].probe] = IqChannel(
                device="my_instrument", path="0", mixer=None, lo=None
            )
            channels[qubits[q].acquisition] = AcquisitionChannel(
                device="my_instrument", path="0", twpa_pump=None, probe=qubits[q].probe
            )

        channels[couplers[0].flux] = DcChannel(device="my_instrument", path="5")

        instruments = {
            "my_instrument": DummyInstrument(
                name="my_instrument", address="0.0.0.0:0", channels=channels
            )
        }

        return Platform.load(FOLDER, instruments, qubits, couplers=couplers)

Note that this assumes that the JSON with parameters is saved as ``<folder>/parameters.json`` where ``<folder>``
is the directory containing ``platform.py``.


Instrument settings
^^^^^^^^^^^^^^^^^^^

The parameters of the previous example contains only parameters associated to the channel configuration
and the native gates. In some cases parameters associated to instruments also need to be calibrated.
An example is the frequency and the power of local oscillators,
such as the one used to pump a traveling wave parametric amplifier (TWPA).

The parameters JSON can contain such parameters in the ``configs`` section:

.. code-block::  json

    {
        "settings": {
            "nshots": 1024,
            "relaxation_time": 50000
        },
        "configs": {
            "twpa_pump": {
                "kind": "oscillator",
                "frequency": 4600000000,
                "power": 5
            }
        },
    }


Note that the key used in the JSON should be the same with the instrument name used
in the instrument dictionary when instantiating the :class:`qibolab.Platform`,
in this case ``"twpa_pump"``.

.. testcode::  python

    # my_platform / platform.py

    from pathlib import Path
    from qibolab import (
        AcquisitionChannel,
        DcChannel,
        IqChannel,
        Platform,
        Qubit,
    )
    from qibolab.instruments import DummyInstrument


    FOLDER = Path.cwd()


    def create():
        qubits = {}
        for q in range(2):
            qubits[q] = Qubit(
                drive=f"{q}/drive",
                flux=f"{q}/flux",
                probe=f"{q}/probe",
                acquisition=f"{q}/acquisition",
            )

        couplers = {0: Qubit(flux="01/coupler")}

        channels = {}
        for q in range(2):
            channels[qubits[q].drive] = IqChannel(
                device="my_instrument", path="1", mixer=None, lo=None
            )
            channels[qubits[q].flux] = DcChannel(device="my_instrument", path="2")
            channels[qubits[q].probe] = IqChannel(
                device="my_instrument", path="0", mixer=None, lo=None
            )
            channels[qubits[q].acquisition] = AcquisitionChannel(
                device="my_instrument", path="0", twpa_pump=None, probe=qubits[q].probe
            )

        channels[couplers[0].flux] = DcChannel(device="my_instrument", path="5")

        instruments = {
            "my_instrument": DummyInstrument(
                name="my_instrument", address="0.0.0.0:0", channels=channels
            ),
            "twpa_pump": DummyLocalOscillator(name="twpa_pump", address="0.0.0.1:0"),
        }

        return Platform.load(FOLDER, instruments, qubits, couplers=couplers)
