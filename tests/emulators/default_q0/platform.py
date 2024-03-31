import pathlib
import logging
from qibolab.channels import ChannelMap
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_runcard, load_settings

from qibolab.instruments.simulator.models import general_no_coupler_model
from qibolab.instruments.simulator.pulse_simulator import (
    PulseSimulator,
    get_default_simulation_config,
)


log = logging.getLogger()
log.setLevel(logging.ERROR)

# set number of levels of qubit
nlevel = 3

# simulation parameters
sim_sampling_boost = 10
simulate_dissipation = True
instant_measurement = True


def create_emulator(runcard_folder: str):
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
        "lo_freq": {"0": lo_freq},
        "rabi_freq": {"0": rabi_freq},
        "anharmonicity": {"0": anharmonicity},
        "coupling_strength": {},
    }

    model_config = general_no_coupler_model.generate_model_config(
        model_params_dict, nlevels_q=[nlevel]
    )
    simulation_config = get_default_simulation_config(sim_sampling_boost)
    simulation_config.update({"simulate_dissipation": simulate_dissipation})
    simulation_config.update({"instant_measurement": instant_measurement})

    pulse_simulator = PulseSimulator(simulation_config, model_config)

    # load runcard
    runcard_folder_path = pathlib.Path(runcard_folder)
    original_runcard = load_runcard(runcard_folder_path)
    runcard = load_runcard(runcard_folder_path)

    # todo: check PulseSimulator topology is superset of runcard topology

    model_config = pulse_simulator.model_config
    emulator_name = pulse_simulator.emulator_name
    qubits_list = model_config["qubits_list"]
    couplers_list = model_config["couplers_list"]
    runcard_qubits_list = original_runcard["qubits"]
    runcard_couplers_list = original_runcard["couplers"]

    log.info(emulator_name)
    log.info(f"emulator qubits: {qubits_list}")
    log.info(f"emulator couplers: {couplers_list}")
    log.info(f"runcard qubits: {runcard_qubits_list}")
    log.info(f"runcard couplers: {runcard_couplers_list}")
    log.info(f"sampling rate: {pulse_simulator.sampling_rate}GHz")
    log.info(f"simulation sampling boost: {pulse_simulator.sim_sampling_boost}")

    # Create channel object
    channels = ChannelMap()
    channels |= (f"readout-{q}" for q in qubits_list)
    channels |= (f"drive-{q}" for q in qubits_list)

    # extract quantities from runcard for platform declaration
    qubits, couplers, pairs = load_qubits(runcard)
    settings = load_settings(runcard)

    # Specify emulator controller
    instruments = {"pulse_simulator": pulse_simulator}

    # map channels to qubits
    for q in runcard_qubits_list:
        # print(q)
        qubits[q].readout = channels[f"readout-{q}"]
        qubits[q].drive = channels[f"drive-{q}"]

        channels[f"drive-{q}"].qubit = qubits[q]
        qubits[q].sweetspot = 0  # not used

    return Platform(
        emulator_name, qubits, pairs, instruments, settings, resonator_type="2D"
    )

