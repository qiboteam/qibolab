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

This is permanently stored as its constructing function, ``create()``, defined in a
source file, but whose data could be stored in a package-defined format, for which
loading (and even dumping) methods are provided.
The details of this process are explained in the following sections.

.. note::

   The main distinction between the content of the Python source file and the parameters
   stored as data is based on the possibility to automatically read and consume these
   parameters, and possibly even update them in a calibration process.

   More parameters may be introduced, and occasionally some platforms are defining them
   in the source, in a first stage.

   The general idea is to retain as much flexibility as possible, while avoiding the
   custom handling of commonly structured data by each platform, that would also
   complicate its handling by downstream projects.

Registering platforms
^^^^^^^^^^^^^^^^^^^^^

The ``create()`` function described in the above example can be called or imported
directly in any Python script. Alternatively, it is also possible to make the platform
available as

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

Operating a QPU requires calibrating a set of parameters, the number of which increases
with the number of qubits. Hardcoding such parameters in the ``create()`` function is
not scalable.
However, since ``create()`` is part of a Python module, is is possible to load
parameters from an external file or database.

Qibolab provides some utility functions, accessible through
:py:mod:`qibolab._core.parameters`, for loading calibration parameters stored in a JSON
file with a specific format.
Here is an example

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

The parameters of the previous example contains only parameters associated to the
channel configuration and the native gates. In some cases parameters associated to
instruments also need to be calibrated.
An example is the frequency and the power of local oscillators, such as the one used to
pump a traveling wave parametric amplifier (TWPA).

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


Note that the key used in the JSON have to be the same with the instrument name used in
the instrument dictionary when instantiating the :class:`qibolab.Platform`, in this case
``"twpa_pump"``.

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
