import itertools

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.dummy import DummyInstrument
from qibolab.platform import Platform

NAME = "dummy2"
RUNCARD = {
    "nqubits": 5,
    "description": "Dummy2 5-qubits + couplers on star topology platform.",
    "qubits": [
        0,
        1,
        2,
        3,
        4,
    ],
    "couplers": [
        "c0",
        "c1",
        "c3",
        "c4",
    ],
    "settings": {"sampling_rate": 1000000000, "relaxation_time": 0, "nshots": 1024},
    "resonator_type": "2D",
    "topology": [[0, 2], [1, 2], [2, 3], [2, 4]],
    "native_gates": {
        "single_qubit": {
            0: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.1,
                    "frequency": 4.0e9,
                    "shape": "Gaussian(5)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 2000,
                    "amplitude": 0.1,
                    "frequency": 5.2e9,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            1: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.3,
                    "frequency": 4.2e9,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 2000,
                    "amplitude": 0.1,
                    "frequency": 4.9e9,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            2: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.3,
                    "frequency": 4.5e9,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 2000,
                    "amplitude": 0.1,
                    "frequency": 6.1e9,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            3: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.3,
                    "frequency": 4.15e9,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 2000,
                    "amplitude": 0.1,
                    "frequency": 5.8e9,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            4: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.3,
                    "frequency": 4155663000,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 2000,
                    "amplitude": 0.1,
                    "frequency": 5.5e9,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
        },
        "two_qubit": {
            "0-2": {
                "CZ": [
                    {
                        "duration": 30,
                        "amplitude": 0.05,
                        "shape": "Rectangular()",
                        "qubit": 0,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": 0.0, "qubit": 0},
                    {"type": "virtual_z", "phase": 0.0, "qubit": 2},
                ]
            },
            "1-2": {
                "CZ": [
                    {
                        "duration": 30,
                        "amplitude": 0.05,
                        "shape": "Rectangular()",
                        "qubit": 0,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": 0.0, "qubit": 1},
                    {"type": "virtual_z", "phase": 0.0, "qubit": 2},
                ]
            },
            "3-2": {
                "CZ": [
                    {
                        "duration": 30,
                        "amplitude": 0.05,
                        "shape": "Rectangular()",
                        "qubit": 0,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": 0.0, "qubit": 3},
                    {"type": "virtual_z", "phase": 0.0, "qubit": 2},
                ]
            },
            "4-2": {
                "CZ": [
                    {
                        "duration": 30,
                        "amplitude": 0.05,
                        "shape": "Rectangular()",
                        "qubit": 0,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": 0.0, "qubit": 4},
                    {"type": "virtual_z", "phase": 0.0, "qubit": 2},
                ]
            },
        },
    },
    "characterization": {
        "single_qubit": {
            0: {
                "readout_frequency": 5.2e9,
                "drive_frequency": 4.0e9,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "mean_gnd_states": (0 + 1j),
                "mean_exc_states": (1 + 0j),
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            1: {
                "readout_frequency": 4.9e9,
                "drive_frequency": 4.2e9,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "mean_gnd_states": (0.25 + 0j),
                "mean_exc_states": (0 + 0.25j),
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            2: {
                "readout_frequency": 6.1e9,
                "drive_frequency": 4.5e9,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "mean_gnd_states": (0.5 + 0j),
                "mean_exc_states": (0 + 0.5j),
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            3: {
                "readout_frequency": 5.8e9,
                "drive_frequency": 4.15e9,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "mean_gnd_states": (0.75 + 0j),
                "mean_exc_states": (0 + 0.75j),
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            4: {
                "readout_frequency": 5.5e9,
                "drive_frequency": 4.1e9,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "mean_gnd_states": (1 + 0j),
                "mean_exc_states": (0 + 1j),
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            "c0": {"sweetspot": 0.0},
            "c1": {"sweetspot": 0.0},
            "c3": {"sweetspot": 0.0},
            "c4": {"sweetspot": 0.0},
        }
    },
}


def create_dummy2():
    """Create a dummy platform using the dummy instrument."""
    # Create dummy controller
    instrument = DummyInstrument(NAME, 0)
    # Create channel objects
    nqubits = RUNCARD["nqubits"]
    channels = ChannelMap()
    channels |= Channel("readout", port=instrument["readout"])
    channels |= (Channel(f"drive-{i}", port=instrument[f"drive-{i}"]) for i in range(nqubits))
    channels |= (Channel(f"flux-{i}", port=instrument[f"flux-{i}"]) for i in range(nqubits))
    channels |= (
        Channel(f"flux_coupler-c{c}", port=instrument[f"flux_coupler-c{c}"])
        for c in itertools.chain(range(0, 2), range(3, 5))
    )

    # Create platform
    platform = Platform(NAME, RUNCARD, [instrument], channels)

    instrument.sampling_rate = platform.sampling_rate * 1e-9

    # map channels to qubits
    for qubit in platform.qubits:
        platform.qubits[qubit].readout = channels["readout"]
        platform.qubits[qubit].drive = channels[f"drive-{qubit}"]
        platform.qubits[qubit].flux = channels[f"flux-{qubit}"]
        channels["readout"].attenuation = 0

    for coupler in platform.couplers:
        platform.couplers[coupler].flux = channels[f"flux_coupler-{coupler}"]

    # FIXME: This could be a way of getting qubits to coupler or we could get couplers to qubits
    # FIXME: Nicer way of getting this range
    # assign couplers to qubits
    for c in itertools.chain(range(0, 2), range(3, 5)):
        platform.qubits[c].flux_coupler[f"c{c}"] = platform.couplers[f"c{c}"]
        platform.qubits[2].flux_coupler[f"c{c}"] = platform.couplers[f"c{c}"]

    # assign qubits to couplers
    for c in itertools.chain(range(0, 2), range(3, 5)):
        platform.couplers[f"c{c}"].qubits[c] = [platform.qubits[c]]
        platform.couplers[f"c{c}"].qubits[c].append(platform.qubits[2])

    return platform
