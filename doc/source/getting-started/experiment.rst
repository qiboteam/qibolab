First Experiment
================

In this example, we'll run a simple single-qubit quantum experiment: single-shot classification.

Setup the Platform
------------------

First, define a single-qubit platform with minimal hardware:

.. testcode:: python

    from qibolab import AcquisitionChannel, Hardware, IqChannel, Qubit, Platform, Parameters
    from qibolab.instruments.dummy import DummyInstrument

    # Define qubit with drive, probe, and acquisition channels
    qubit = Qubit.default(0)

    # Create channels
    channels = {
        qubit.probe: IqChannel(mixer=None, lo=None),
        qubit.acquisition: AcquisitionChannel(probe=qubit.probe, twpa_pump=None),
        qubit.drive: IqChannel(mixer=None, lo=None),
    }

    # Create dummy instrument
    controller = DummyInstrument(address="192.168.0.101:80", channels=channels)

    # Create hardware
    hardware = Hardware(
        instruments={"dummy": controller},
        qubits={0: qubit},
    )

Platform parameters are stored in a dictionary (see below for details):

.. testcode:: python

    parameters_dict = {
        "settings": {"nshots": 1000, "relaxation_time": 70000},
        "configs": {
            "0/drive": {"kind": "iq", "frequency": 4833726197},
            "0/probe": {"kind": "iq", "frequency": 7320000000},
            "0/acquisition": {
                "kind": "acquisition",
                "delay": 224,
                "smearing": 0,
                "threshold": 0.002,
                "iq_angle": -0.767,
            },
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
                                "amplitude": 0.5,
                                "envelope": {"kind": "gaussian", "rel_sigma": 3.0},
                            },
                        ],
                    ],
                    "MZ": [
                        [
                            "0/acquisition",
                            {
                                "kind": "readout",
                                "acquisition": {
                                    "kind": "acquisition",
                                    "duration": 2000.0,
                                },
                                "probe": {
                                    "kind": "pulse",
                                    "duration": 2000.0,
                                    "amplitude": 0.003,
                                    "envelope": {"kind": "rectangular"},
                                },
                            },
                        ]
                    ],
                }
            },
            "two_qubit": {},
        },
    }

    # Create platform
    params = Parameters.model_validate(parameters_dict)
    platform = Platform(
        name="my_platform",
        parameters=params,
        **vars(hardware),
    )

Run the Experiment
------------------

Execute a simple single-shot classification:

.. testcode:: python

    import matplotlib.pyplot as plt
    from qibolab import AcquisitionType

    # Get native gates
    gates = platform.natives.single_qubit[0]

    # Run two experiments: measure |0> and |1>
    results = []
    for sequence in [gates.MZ(), gates.RX() | gates.MZ()]:
        signal = platform.execute(
            [sequence],
            nshots=1000,
            acquisition_type=AcquisitionType.INTEGRATION,
        )
        # Extract acquisition pulse ID
        _, acq = next(iter(sequence.acquisitions))
        sig = signal[acq.id]
        results.append([sig[..., 0], sig[..., 1]])

    # Plot results
    plt.title("Single-Shot Classification")
    plt.xlabel("In-phase [a.u.]")
    plt.ylabel("Quadrature [a.u.]")
    plt.scatter(*results[0], label="↓ (no RX)")
    plt.scatter(*results[1], label="↑ (after RX)")
    plt.legend()

What's Happening?
-----------------

The code above:

1. Defines a single-qubit platform with a dummy instrument
2. Runs two experiments:
   - **MZ()**: Just measure (detect state |0>)
   - **RX() | MZ()**: Apply π/2 rotation, then measure (detect state |1>)
3. Collects I and Q quadrature data from both experiments
4. Plots the results on the IQ plane

With the dummy instrument, results are random noise. Use the :ref:`emulator <main_doc_emulator>`
for realistic quantum simulation, or connect to real hardware.

Next Steps
----------

- See :ref:`main_doc_experiment` for complete Experiment API reference
- Try the :ref:`emulator tutorial <tutorial_emulator>` for realistic simulation
- Read :ref:`tutorial_platform` to build custom platforms
- Explore :ref:`tutorial_calibration` for gate calibration
