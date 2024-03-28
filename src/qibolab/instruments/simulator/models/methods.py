from typing import Union

import yaml

MHz = 1e6
GHz = 1e9
us = 1e-6
ns = 1e-9


def load_model_params(model_params: Union[dict, str]) -> dict:
    """Load yaml to a dictionary or returns the input if it is already a
    dictionary."""
    if isinstance(model_params, dict):
        params = model_params
    else:
        with open(model_params) as file:
            params = yaml.safe_load(file)
    return params


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
