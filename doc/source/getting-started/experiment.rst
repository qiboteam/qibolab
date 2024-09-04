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

where ``platform.py`` contains instruments information and ``parameters.json`` includes calibration parameters.

More information about defining platforms is provided in :doc:`../tutorials/lab` and several examples can be found
at the `TII QRC lab dedicated repository <https://github.com/qiboteam/qibolab_platforms_qrc>`_.

For a first experiment, let's define a single qubit platform at the path previously specified.
In this example, the qubit is controlled by a Quantum Machines cluster that contains Octaves,
although minimal changes are needed to use other devices.

.. testcode:: python

    # my_platform/platform.py

    import pathlib

    from qibolab.components import AcquisitionChannel, Channel, DcChannel, IqChannel
    from qibolab.identifier import ChannelId
    from qibolab.instruments.qm import Octave, QmConfigs, QmController
    from qibolab.parameters import ConfigKinds
    from qibolab.platform import Platform
    from qibolab.platform.platform import QubitMap
    from qibolab.qubits import Qubit

    # folder containing runcard with calibration parameters
    FOLDER = pathlib.Path.cwd()

    # Register QM-specific configurations for parameters loading
    ConfigKinds.extend([QmConfigs])


    def create():
        # Define qubit
        qubits: QubitMap = {
            0: Qubit(
                drive="0/drive",
                probe="0/probe",
                acquisition="0/acquisition",
            )
        }

        # Create channels and connect to instrument ports
        channels: dict[ChannelId, Channel] = {}
        qubit = qubits[0]
        # Readout
        channels[qubit.probe] = IqChannel(
            device="octave1", path="1", mixer=None, lo="0/probe/lo"
        )
        # Acquire
        channels[qubit.acquisition] = AcquisitionChannel(
            device="octave1", path="1", probe=qubit.probe
        )
        # Drive
        channels[qubit.drive] = IqChannel(
            device="octave1", path="2", mixer=None, lo="0/drive/lo"
        )

        # Define Quantum Machines instruments
        octaves = {
            "octave1": Octave("octave5", port=101, connectivity="con1"),
        }
        controller = QmController(
            name="qm",
            address="192.168.0.101:80",
            octaves=octaves,
            channels=channels,
            calibration_path=FOLDER,
        )

        # Define and return platform
        return Platform.load(
            path=FOLDER, instruments=[controller], qubits=qubits, resonator_type="3D"
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
        "settings": {
            "nshots": 1024,
            "relaxation_time": 70000
        },
        "configs": {
            "0/drive": {
                "kind": "iq",
                "frequency": 4833726197
            },
            "0/drive/lo": {
                "kind": "oscillator",
                "frequency": 5200000000,
                "power": 0
            },
            "0/probe": {
                "kind": "iq",
                "frequency": 7320000000
            },
            "0/probe/lo": {
                "kind": "oscillator",
                "frequency": 7300000000,
                "power": 0
            },
            "0/acquisition": {
                "kind": "qm-acquisition",
                "delay": 224,
                "smearing": 0,
                "threshold": 0.002100861788865835,
                "iq_angle": -0.7669877581038627,
                "gain": 10,
                "offset": 0.0
            }
        },
        "native_gates": {
            "single_qubit": {
                "0": {
                    "RX": {
                        "0/drive": [
                            {
                                "duration": 40,
                                "amplitude": 0.5,
                                "envelope": { "kind": "gaussian", "rel_sigma": 3.0 },
                                "type": "qd"
                            }
                        ]
                    },
                    "MZ": [
                        [
                            "0/acquisition",
                            {
                                "kind": "readout",
                                "acquisition": {
                                    "kind": "acquisition",
                                    "duration": 2000.0
                                },
                                "probe": {
                                    "kind": "pulse",
                                    "duration": 2000.0,
                                    "amplitude": 0.003,
                                    "envelope": {
                                        "kind": "rectangular"
                                    }
                                }
                            }
                        ]
                    ]
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

To avoid having to repeat this export command for every session, this line can be added to the ``.bashrc`` file (or alternatives such as ``.zshrc``).


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
    signal = results[acq.id]
    amplitudes = signal[..., 0] + 1j * signal[..., 1]
    frequencies = sweeper.values

    plt.title("Resonator Spectroscopy")
    plt.xlabel("Frequencies [Hz]")
    plt.ylabel("Amplitudes [a.u.]")

    plt.plot(frequencies, amplitudes)

.. image:: ../tutorials/resonator_spectroscopy_light.svg
   :class: only-light
.. image:: ../tutorials/resonator_spectroscopy_dark.svg
   :class: only-dark
