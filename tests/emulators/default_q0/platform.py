import pathlib
import logging

from qibolab.channels import ChannelMap
from qibolab.instruments.simulator.pulse_simulator import (
    PulseSimulator,
    get_default_simulation_config,
)
from qibolab.instruments.simulator.models import general_no_coupler_model
from qibolab.instruments.simulator.models.methods import load_model_params
from qibolab.platform import Platform
from qibolab.serialize import (
    load_qubits,
    load_runcard,
    load_settings,
)

#from qibolab.emulator import create_runcard_emulator
log = logging.getLogger()
#log.setLevel(logging.ERROR)
log.setLevel(logging.INFO)

FOLDER = pathlib.Path(__file__).parent

# simulation parameters
sim_sampling_boost = 10
simulate_dissipation = True
instant_measurement = True


#def create_oneQ_emulator(runcard_folder: str):
def create_emulator(nlevel:int=3):
    """Create a one qubit emulator platform."""

    # load runcard
    original_runcard = load_runcard(FOLDER)
    runcard = load_runcard(FOLDER)
    model_params = load_model_params(FOLDER)

    model_config = general_no_coupler_model.generate_model_config(
        model_params, nlevels_q=[nlevel]
    )
    simulation_config = get_default_simulation_config(sim_sampling_boost)
    simulation_config.update({"simulate_dissipation": simulate_dissipation})
    simulation_config.update({"instant_measurement": instant_measurement})

    pulse_simulator = PulseSimulator(simulation_config, model_config)

    #return create_runcard_emulator(runcard_folder, pulse_simulator)

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

