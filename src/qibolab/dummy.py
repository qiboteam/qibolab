from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.dummy import DummyInstrument
from qibolab.instruments.oscillator import LocalOscillator
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_settings

NAME = "dummy"
RUNCARD = {
    "nqubits": 4,
    "qubits": [
        0,
        1,
        2,
        3,
    ],
    "settings": {"sampling_rate": 1000000000, "relaxation_time": 0, "nshots": 1024},
    "topology": [[0, 1], [1, 2], [0, 3]],
    "native_gates": {
        "single_qubit": {
            0: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.005,
                    "frequency": 4700000000,
                    "shape": "Gaussian(5)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "RX12": {
                    "duration": 40,
                    "amplitude": 0.005,
                    "frequency": 4700000000,
                    "shape": "Gaussian(5)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 1000,
                    "amplitude": 0.0025,
                    "frequency": 7226500000,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            1: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.0484,
                    "frequency": 4855663000,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "RX12": {
                    "duration": 40,
                    "amplitude": 0.0484,
                    "frequency": 4855663000,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 620,
                    "amplitude": 0.003575,
                    "frequency": 7453265000,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            2: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.005,
                    "frequency": 2700000000,
                    "shape": "Gaussian(5)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "RX12": {
                    "duration": 40,
                    "amplitude": 0.005,
                    "frequency": 2700000000,
                    "shape": "Gaussian(5)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 1000,
                    "amplitude": 0.0025,
                    "frequency": 5226500000,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
            3: {
                "RX": {
                    "duration": 40,
                    "amplitude": 0.0484,
                    "frequency": 5855663000,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "RX12": {
                    "duration": 40,
                    "amplitude": 0.0484,
                    "frequency": 5855663000,
                    "shape": "Drag(5, -0.02)",
                    "type": "qd",
                    "start": 0,
                    "phase": 0,
                },
                "MZ": {
                    "duration": 620,
                    "amplitude": 0.003575,
                    "frequency": 8453265000,
                    "shape": "Rectangular()",
                    "type": "ro",
                    "start": 0,
                    "phase": 0,
                },
            },
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
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": -1.5707963267948966, "qubit": 1},
                    {"type": "virtual_z", "phase": -1.5707963267948966, "qubit": 0},
                ]
            },
            "1-2": {
                "CZ": [
                    {
                        "duration": 30,
                        "amplitude": 0.055,
                        "shape": "Rectangular()",
                        "qubit": 1,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": -1.5707963267948966, "qubit": 2},
                    {"type": "virtual_z", "phase": -1.5707963267948966, "qubit": 1},
                ]
            },
            "0-3": {
                "CZ": [
                    {
                        "duration": 34,
                        "amplitude": 0.055,
                        "shape": "SNZ(1)",
                        "qubit": 3,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {
                        "duration": 4,
                        "amplitude": 0.055,
                        "shape": "SNZ(1)",
                        "qubit": 3,
                        "relative_start": 14,
                        "type": "qf",
                    },
                    {
                        "duration": 34,
                        "amplitude": 0.1,
                        "shape": "Rectangular()",
                        "qubit": 3,
                        "relative_start": 0,
                        "type": "qf",
                    },
                    {"type": "virtual_z", "phase": -1.5707963267948966, "qubit": 3},
                    {"type": "virtual_z", "phase": -1.5707963267948966, "qubit": 0},
                ]
            },
        },
    },
    "characterization": {
        "single_qubit": {
            0: {
                "readout_frequency": 7226500000,
                "drive_frequency": 4700000000,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            1: {
                "readout_frequency": 7453265000,
                "drive_frequency": 4855663000,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": -0.047,
                "threshold": 0.00028502261712637096,
                "iq_angle": 1.283105298787488,
            },
            2: {
                "readout_frequency": 5226500000,
                "drive_frequency": 2700000000,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": 0.0,
                "threshold": 0.0,
                "iq_angle": 0.0,
            },
            3: {
                "readout_frequency": 8453265000,
                "drive_frequency": 5855663000,
                "T1": 0.0,
                "T2": 0.0,
                "sweetspot": -0.047,
                "threshold": 0.00028502261712637096,
                "iq_angle": 1.283105298787488,
            },
        }
    },
}


def create_dummy():
    """Create a dummy platform on two qubits using the dummy instrument."""
    # Create dummy controller
    instrument = DummyInstrument(NAME, 0)

    # Create local oscillator
    twpa_pump = LocalOscillator(name="twpa_pump", address=0)
    twpa_pump.frequency = 1e9
    twpa_pump.power = 10

    # Create channel objects
    nqubits = RUNCARD["nqubits"]
    channels = ChannelMap()
    channels |= Channel("readout", port=instrument["readout"])
    channels |= (Channel(f"drive-{i}", port=instrument[f"drive-{i}"]) for i in range(nqubits))
    channels |= (Channel(f"flux-{i}", port=instrument[f"flux-{i}"]) for i in range(nqubits))
    channels |= Channel("twpa", port=None)
    channels["readout"].attenuation = 0
    channels["twpa"].local_oscillator = twpa_pump

    qubits, pairs = load_qubits(RUNCARD)
    settings = load_settings(RUNCARD)

    # map channels to qubits
    for q, qubit in qubits.items():
        qubit.readout = channels["readout"]
        qubit.drive = channels[f"drive-{q}"]
        qubit.flux = channels[f"flux-{q}"]
        qubit.twpa = channels["twpa"]

    instruments = {instrument.name: instrument, twpa_pump.name: twpa_pump}
    instrument.sampling_rate = settings.sampling_rate * 1e-9
    return Platform(NAME, qubits, pairs, instruments, settings, resonator_type="2D")
