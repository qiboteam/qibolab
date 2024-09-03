Performing the first experiment
===============================

Define the platform
-------------------

To launch experiments on quantum hardware, users have first to define their platform.
To define a platform the user needs to provide a folder with the following structure:

.. code-block:: bash

    my_platform/
        platform.py
        parameters.json

where ``platform.py`` contains instruments information, ``parameters.json``
includes calibration parameters.

More information about defining platforms is provided in :doc:`../tutorials/lab` and several examples can be found at `TII dedicated repository <https://github.com/qiboteam/qibolab_platforms_qrc>`_.

For a first experiment, let's define a single qubit platform at the path previously specified.
In this example, the qubit is controlled by a Zurich Instruments' SHFQC instrument, although minimal changes are needed to use other devices.

.. testcode:: python

    # my_platform/platform.py

    import pathlib

    from laboneq.simple import DeviceSetup, SHFQC
    from qibolab.components import (
        AcquireChannel,
        IqChannel,
        IqConfig,
        AcquisitionConfig,
        OscillatorConfig,
    )

    from qibolab.instruments.zhinst import ZiChannel, Zurich
    from qibolab.parameters import Parameters
    from qibolab.platform import Platform

    NAME = "my_platform"  # name of the platform
    ADDRESS = "localhost"  # ip address of the ZI data server
    PORT = 8004  # port of the ZI data server

    # folder containing runcard with calibration parameters
    FOLDER = pathlib.Path.cwd()


    def create():
        # Define available instruments
        device_setup = DeviceSetup()
        device_setup.add_dataserver(host=ADDRESS, port=PORT)
        device_setup.add_instruments(SHFQC("device_shfqc", address="DEV12146"))

        # Load and parse the runcard (i.e. parameters.json)
        runcard = Parameters.load(FOLDER)
        qubits = runcard.native_gates.single_qubit
        pairs = runcard.native_gates.pairs
        qubit = qubits[0]

        # define component names, and load their configurations
        drive, probe, acquire = "q0/drive", "q0/probe", "q0/acquire"
        drive_lo, readout_lo = "q0/drive/lo", "q0/readout/lo"

        # assign channels to qubits
        qubit.drive = IqChannel(name=drive, lo=drive_lo, mixer=None)
        qubit.probe = IqChannel(name=probe, lo=readout_lo, mixer=None, acquisition=acquire)
        qubit.acquisition = AcquireChannel(name=acquire, probe=probe, twpa_pump=None)

        zi_channels = [
            ZiChannel(qubit.drive, device="device_shfqc", path="SGCHANNELS/0/OUTPUT"),
            ZiChannel(qubit.probe, device="device_shfqc", path="QACHANNELS/0/OUTPUT"),
            ZiChannel(qubit.acquisition, device="device_shfqc", path="QACHANNELS/0/INPUT"),
        ]

        controller = Zurich(NAME, device_setup=device_setup, channels=zi_channels)

        return Platform(
            name=NAME,
            runcard=runcard,
            instruments={controller: controller},
            resonator_type="3D",
        )


.. note::

    The ``platform.py`` file must contain a ``create_function`` with the following signature:

    .. code-block:: python

        import pathlib
        from qibolab.platform import Platform


        def create() -> Platform:
            """Function that generates Qibolab platform."""

And the we can define the runcard ``my_platform/parameters.json``:

.. code-block:: json

    {
    "nqubits": 1,
    "qubits": [
        0
    ],
    "topology": [],
    "settings": {
        "nshots": 1024,
        "relaxation_time": 70000,
        "sampling_rate": 9830400000
    },
    "components": {
        "qubit_0/drive": {
            "frequency": 4833726197,
            "power_range": 5
        },
        "qubit_0/drive/lo": {
            "frequency": 5200000000,
            "power": null
        },
        "qubit_0/probe": {
            "frequency": 7320000000,
            "power_range": 1
        },
        "qubit_0/readout/lo": {
            "frequency": 7300000000,
            "power": null
        },
        "qubit_0/acquire": {
            "delay": 0,
            "smearing": 0,
            "power_range": 10
        }
    }
    "native_gates": {
        "single_qubit": {
            "0": {
                "RX": {
                    "qubit_0/drive": [
                        {
                            "duration": 40,
                            "amplitude": 0.5,
                            "envelope": { "kind": "gaussian", "rel_sigma": 3.0 },
                            "type": "qd"
                        }
                    ]
                },
                "MZ": {
                    "qubit_0/probe": [
                        {
                            "duration": 2000,
                            "amplitude": 0.02,
                            "envelope": { "kind": "rectangular" },
                            "type": "ro"
                        }
                    ]
                }
            }
        },
        "two_qubits": {}
    }
    }


Setting up the environment
--------------------------

After defining the platform, we must instruct ``qibolab`` of the location of the platform(s).
We need to define the path that contains platform folders.
This can be done using an environment variable:
for Unix based systems:

.. code-block:: bash

    export QIBOLAB_PLATFORMS=<path-platform-folders>

for Windows:

.. code-block:: bash

    $env:QIBOLAB_PLATFORMS="<path-to-platform-folders>"

To avoid having to repeat this export command for every session, this line can be added to the ``.bashrc`` file (or alternatives as ``.zshrc``).


Run the experiment
------------------

Let's take the `Resonator spectroscopy experiment` defined and detailed in :doc:`../tutorials/calibration`.
Since it is a rather simple experiment, it can be used to perform a fast sanity-check on the platform.

We leave to the dedicated tutorial a full explanation of the experiment, but here it is the required code:

.. testcode:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from qibolab import create_platform
    from qibolab.sequence import PulseSequence
    from qibolab.result import magnitude
    from qibolab.sweeper import Sweeper, Parameter
    from qibolab.execution_parameters import (
        ExecutionParameters,
        AveragingMode,
        AcquisitionType,
    )

    # load the platform from ``dummy.py`` and ``dummy.json``
    platform = create_platform("dummy")

    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]
    # define the pulse sequence
    sequence = natives.MZ.create_sequence()

    # define a sweeper for a frequency scan
    f0 = platform.config(qubit.probe).frequency  # center frequency
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 2e8, f0 + 2e8, 1e6),
        channels=[qubit.probe],
    )

    # perform the experiment using specific options
    options = ExecutionParameters(
        nshots=1000,
        relaxation_time=50,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    results = platform.execute([sequence], options, [[sweeper]])
    _, acq = next(iter(sequence.acquisitions))

    # plot the results
    amplitudes = magnitude(results[acq.id][0])
    frequencies = sweeper.values

    plt.title("Resonator Spectroscopy")
    plt.xlabel("Frequencies [Hz]")
    plt.ylabel("Amplitudes [a.u.]")

    plt.plot(frequencies, amplitudes)

.. image:: ../tutorials/resonator_spectroscopy_light.svg
   :class: only-light
.. image:: ../tutorials/resonator_spectroscopy_dark.svg
   :class: only-dark
