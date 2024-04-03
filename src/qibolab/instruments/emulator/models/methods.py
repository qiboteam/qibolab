import json
import operator
from functools import reduce
from pathlib import Path

GHz = 1e9

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
    return reduce(
        operator.or_,
        [{f"drive-{q}": f"D-{q}", f"readout-{q}": f"R-{q}"} for q in qubits_list]
        + [{f"drive-{c}": f"D-{c}"} for c in couplers_list],
    )
