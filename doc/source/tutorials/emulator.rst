How to emulate a QPU using Qibolab?
===================================

In this section we are going to explain how to setup a platform for pulse emulation
and how to use it in Qibolab. To setup a platform we first recommend to have a look at
:ref:`tutorial_platform`.

We are now going to explain a few distinctive features of an emulation
platform. All parameters related to the Hamiltonian to be simulated are coded directly in the JSON file under
the section ``configs``. Here is an example

.. code-block::  json

    {
        "settings": {
            "nshots": 1024,
            "relaxation_time": 0
            },
        "configs": {
            "emulator/bounds": {
            "kind": "bounds",
            "waveforms": 1000000,
            "readout": 50,
            "instructions": 200
            },
        "hamiltonian":{
            "transmon_levels": 2,
            "single_qubit": {
                "0": {
                    "frequency": 5e9,
                    "anharmonicity": -200e6,
                    "t1": {
                        "0-1": 1000
                        },
                    "t2": {
                        "0-1": 1900
                        }
                    }
                },
            "kind": "hamiltonian"
            },
            "0/drive": {
                "kind": "iq",
                "frequency": 5e9
                },
            "0/drive12": {
                "kind": "iq",
                "frequency": 4.8e9
                },
            "0/probe": {
                "kind": "iq",
                "frequency": 5200000000.0
                },
            "0/acquisition": {
                "kind": "acquisition",
                "delay": 0.0,
                "smearing": 0.0,
                "threshold": 0.0,
                "iq_angle": 0.0,
                "kernel": null
                }
            },
        "native_gates": {
            "single_qubit": {
                "0": {
                    "RX": [
                    [
                        "0/drive",
                        {
                            "duration": 40,
                            "amplitude": 0.1594,
                            "envelope": {
                                "kind": "gaussian",
                                "rel_sigma": 0.2
                            },
                            "relative_phase": 0.0,
                            "kind": "pulse"
                        }
                    ]
                    ],
                    "RX90": [
                    [
                        "0/drive",
                        {
                            "duration": 40,
                            "amplitude": 0.07975,
                            "envelope": {
                                "kind": "gaussian",
                                "rel_sigma": 0.2
                            },
                            "relative_phase": 0.0,
                            "kind": "pulse"
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
                                "duration": 100.0
                            },
                        "probe": {
                            "duration": 100.0,
                            "amplitude": 0.1,
                            "envelope": {
                                "kind": "gaussian_square",
                                "rel_sigma": 0.2,
                                "width": 0.75
                                },
                            "relative_phase": 0.0,
                            "kind": "pulse"
                        }
                        }
                    ]
                    ],
                    "CP": null
                }
                }
                }
            }


We are defining an Hamiltonian with a transmon with two-levels, with a frequency of :math:`\omega_q / 2 \pi = 5 \ \text{GHz}` and
anharmoncity :math:`\alpha/2 \pi = - 200 \ \text{MHz}`,
with :math:`T_1 = 1 \  \mu s` and :math:`T_2 = 1.9 \ \mu s`.
Everything else follows the usual Qibolab conventions. Keep in mind that you still need to define also a readout pulse even if all
parameters ignored in the current emulator except when the readout pulse is played.

We are now going to give an example on how to setup the `platform.py` file.


.. testcode::  python

    # emulator / platform.py


    from qibolab import ConfigKinds, IqChannel, Hardware, Qubit
    from qibolab.instruments.emulator import EmulatorController, HamiltonianConfig

    ConfigKinds.extend([HamiltonianConfig])


    def create() -> Hardware:
        """Create an emulator hardware."""
        qubits = {Qubit.default(q) for q in range(1)}
        channels = {
            qubit.drive: IqChannel(mixer=None, lo=None) for qubit in qubits.values()
        }

        # register the instruments
        instruments = {
            "dummy": EmulatorController(address="0.0.0.0", channels=channels),
        }

        return Hardware(
            instruments=instruments,
            qubits=qubits,
        )

We can observe that we need to allocate an object of type ``EmulatorController`` where we load the channels.
Note that in order to enables the config to support the Hamiltonian configuration we are adding it explicitly
in the statement ``ConfigKinds.extend([HamiltonianConfig])``.
