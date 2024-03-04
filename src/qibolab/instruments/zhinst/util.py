"""Utility methods."""

from qibolab.qubits import Qubit

SAMPLING_RATE = 2
NANO_TO_SECONDS = 1e-9


def measure_channel_name(qubit: Qubit) -> str:
    """Construct and return a name for qubit's measure channel.

    FIXME: We cannot use channel name directly, because currently channels are named after wires, and due to multiplexed readout
    multiple qubits have the same channel name for their readout. Should be fixed once channels are refactored.
    """
    return f"{qubit.readout.name}_{qubit.name}"


def acquire_channel_name(qubit: Qubit) -> str:
    """Construct and return a name for qubit's acquire channel.

    FIXME: We cannot use acquire channel name, because qibolab does not have a concept of acquire channel. This function shall be removed
    once all channel refactoring is done.
    """
    return f"acquire{qubit.name}"
