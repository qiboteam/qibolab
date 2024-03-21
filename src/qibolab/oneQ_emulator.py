from qibolab.emulator import create_runcard_emulator
from qibolab.instruments.simulator.models.general_no_coupler_model import (
    generate_model_config_oneQ,
)
from qibolab.instruments.simulator.pulse_simulator import (
    PulseSimulator,
    get_default_simulation_config,
)

# device parameters: frequency in Hz, time in s
device_name = "ibmfakebelemQ0"
sampling_rate = 4500000000.0
readout_error = [0.01, 0.02]
lo_freq = 5090167234.445013
anharmonicity = -336123005.1821652
drive_freq = 5090167234.445013
rabi_freq = 125457538.19061986
T1 = 8.857848970762537e-05
T2 = 0.00010679794866226273
T2e = 0.0
nlevel = 3

# simulation parameters
sim_sampling_boost = 10
simulate_dissipation = True
instant_measurement = True


def create_oneQ_emulator(runcard_folder: str):
    """Create a one qubit emulator platform."""

    model_params_dict = {
        "device_name": device_name,
        "topology": [],
        "nqubits": 1,
        "ncouplers": 0,
        "qubits_list": ["0"],
        "couplers_list": [],
        "sampling_rate": sampling_rate,
        "readout_error": {0: readout_error},
        "drive_freq": {"0": drive_freq},
        "T1": {"0": T1},
        "T2": {"0": T2},
        "T2e": {"0": T2e},
        "lo_freq": {"0": lo_freq},
        "rabi_freq": {"0": rabi_freq},
        "anharmonicity": {"0": anharmonicity},
        "coupling_strength": {},
    }

    model_config = generate_model_config_oneQ(model_params_dict, nlevel)
    simulation_config = get_default_simulation_config(sim_sampling_boost)
    simulation_config.update({"simulate_dissipation": simulate_dissipation})
    simulation_config.update({"instant_measurement": instant_measurement})

    pulse_simulator = PulseSimulator(simulation_config, model_config)

    return create_runcard_emulator(runcard_folder, pulse_simulator)
