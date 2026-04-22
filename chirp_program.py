
# Single QUA script generated at 2026-04-22 12:15:47.801107
# QUA library version: 1.2.6


from qm import CompilerOptionArguments
from qm.qua import *

with program() as prog:
    v1 = declare(int, )
    with for_(v1,0,(v1<1000),(v1+1)):
        align()
        play("-6956312437102561334", "drive", chirp=(750,None,"Hz/nsec"))
        wait(25000, )

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "filter": {
                        "feedback": [],
                        "feedforward": [],
                    },
                },
            },
            "digital_outputs": {},
            "analog_inputs": {},
        },
    },
    "octaves": {},
    "elements": {
        "drive": {
            "singleInput": {
                "port": ('con1', 1),
            },
            "intermediate_frequency": 0,
            "operations": {
                "-6956312437102561334": "-6956312437102561334",
            },
        },
    },
    "pulses": {
        "-6956312437102561334": {
            "length": 400000,
            "waveforms": {
                "single": "-6956312437102561334",
            },
            "digital_marker": "ON",
            "operation": "control",
        },
    },
    "waveforms": {
        "-6956312437102561334": {
            "sample": 0.1,
            "type": "constant",
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {},
    "mixers": {},
}

loaded_config = {
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                "1": {
                    "offset": 0.0,
                    "delay": 0,
                    "shareable": False,
                    "filter": {
                        "feedforward": [],
                        "feedback": [],
                    },
                    "crosstalk": {},
                },
            },
            "analog_inputs": {},
            "digital_outputs": {},
            "digital_inputs": {},
        },
    },
    "oscillators": {},
    "elements": {
        "drive": {
            "digitalInputs": {},
            "digitalOutputs": {},
            "outputs": {},
            "operations": {
                "-6956312437102561334": "-6956312437102561334",
            },
            "hold_offset": {
                "duration": 0,
            },
            "sticky": {
                "analog": False,
                "digital": False,
                "duration": 4,
            },
            "singleInput": {
                "port": ('con1', 1, 1),
            },
            "intermediate_frequency": 0,
        },
    },
    "pulses": {
        "-6956312437102561334": {
            "length": 400000,
            "waveforms": {
                "single": "-6956312437102561334",
            },
            "integration_weights": {},
            "operation": "control",
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "-6956312437102561334": {
            "type": "constant",
            "sample": 0.1,
        },
    },
    "digital_waveforms": {
        "ON": {
            "samples": [(1, 0)],
        },
    },
    "integration_weights": {},
    "mixers": {},
}

from qm import QuantumMachinesManager, SimulationConfig
from qm_saas import QmSaas
import matplotlib.pyplot as plt
import numpy as np
import copy

client = QmSaas(email="nandan24@nus.edu.sg", password="TpXa-LqRe-ZmWs-BkFy")

with client.simulator("v2_6_0") as instance:
    qmm = QuantumMachinesManager(
        host=instance.host,
        port=instance.port,
        connection_headers=instance.default_connection_headers,
    )

    # Fix filter format - just remove it entirely for the emulator
    sim_config = copy.deepcopy(config)
    for con in sim_config["controllers"].values():
        for port in con.get("analog_outputs", {}).values():
            port.pop("filter", None)

    job = qmm.simulate(sim_config, prog, SimulationConfig(int(4e5 // 4)))
    samples = job.get_simulated_samples()
    x = samples.con1.analog["1"]

    NFFT = 2**10
    Fs = 1e9
    fig, ax1 = plt.subplots()
    ax1.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=100, cmap=plt.cm.gist_heat)
    ax1.set_xlabel("t [us]")
    ax1.set_ylabel("f [MHz]")
    plt.title("Linear Chirp")
    plt.savefig("chirp_spectrogram.png", dpi=150, bbox_inches="tight")
    print("Saved to chirp_spectrogram.png")