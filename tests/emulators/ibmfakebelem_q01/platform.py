import pathlib
import logging
from qibolab.channels import ChannelMap
from qibolab.instruments.simulator.pulse_simulator import PulseSimulator
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_runcard, load_settings


log = logging.getLogger()
log.setLevel(logging.ERROR)

def create_runcard_emulator(runcard_folder: str, pulse_simulator: PulseSimulator):
    """Create a emulator platform using the emulator instrument."""
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
