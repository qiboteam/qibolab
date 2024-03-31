import json
from typing import Union
from pathlib import Path

GHz = 1e9
ns = 1e-9

MODEL_PARAMS = "model.json"

def load_model_params(path: Path) -> dict:
    """Load model parameters JSON to a dictionary."""
    return json.loads((path / MODEL_PARAMS).read_text())


def default_noflux_platform2simulator_channels(
    qubits_list: list, couplers_list: list
) -> dict:
    """Returns the default dictionary that maps platform channel names to simulator channel names.
    Args:
        qubits_list (list): List of qubit names to be included in the simulation.
        couplers_list (list): List of coupler names to be included in the simulation.

    Returns:
        dict: Mapping between platform channel names to simulator chanel names.
    """
    platform2simulator_channels = {}
    for qubit in qubits_list:
        platform2simulator_channels.update({f"drive-{qubit}": f"D-{qubit}"})
        platform2simulator_channels.update({f"readout-{qubit}": f"R-{qubit}"})
    for coupler in couplers_list:
        platform2simulator_channels.update({f"drive-{coupler}": f"D-{coupler}"})

    return platform2simulator_channels
