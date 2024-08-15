import operator
import numpy as np
from functools import reduce


def default_noflux_platform_to_simulator_channels(
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


def default_platform_to_simulator_channels(
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
        [{f"drive-{q}": f"D-{q}", f"readout-{q}": f"R-{q}", f"flux-{q}": f"F-{q}"} for q in qubits_list]
        + [{f"drive-{c}": f"D-{c}", f"flux-{c}": f"F-{c}"} for c in couplers_list],
    )


def flux_detuning(
    flux_pulse_amplitude: np.ndarray,
    flux_quanta: float,
    max_frequency: float,
    current_frequency: float,
) -> float:
    """Function that returns detuned qubit frequency due to flux pulse."""
    phase = flux_pulse_amplitude/flux_quanta 
    
    return max_frequency * np.sqrt(np.abs(np.cos(phase))) - current_frequency