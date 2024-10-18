"""Utils for result management."""

from functools import reduce

import keysight.qcs as qcs  # pylint: disable=E0401
import numpy as np

from qibolab._core.execution_parameters import AcquisitionType

__all__ = ["fetch_result"]


def fetch_result(
    results: qcs.Results,
    channel: qcs.Channels,
    acquisition_type: AcquisitionType,
    averaging: bool,
    sweeper_swaps_required: list[tuple[int, int]],
) -> dict[qcs.Channels, np.ndarray]:
    """Processes the QCS result object to return the appropiate results.

    Arguments:
        results (qcs.Results): QCS result object.
        channel (qcs.Channels): Virtual acquisition channel to obtain results from.
        acquisition_type (AcquisitionType): Acquisition type to be used.
        averaging (bool): Flag for averaging results
        sweeper_swaps_required (list[tuple[int, int]]): Array of axes pairs corresponding to swapped sweepers.

    Returns:
        raw (dict[qcs.Channels, np.ndarray]): Map of virtual channel to acquisition results.
    """
    if acquisition_type is AcquisitionType.RAW:
        raw = results.get_trace(channel, averaging)
    elif acquisition_type is AcquisitionType.INTEGRATION:
        raw = results.get_iq(channel, averaging)
    elif acquisition_type is AcquisitionType.DISCRIMINATION:
        raw = results.get_classified(channel, averaging)
    else:
        raise ValueError("Acquisition type unrecognized")

    if len(sweeper_swaps_required) > 0:
        swap = lambda array, pair_indices: np.swapaxes(array, *pair_indices)
        for key, value in raw.items():
            raw[key] = reduce(swap, sweeper_swaps_required, value)

    return raw
