"""Utils for result management."""

import numpy as np
from keysight import qcs

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)


def fetch_result(
    results: qcs.Results,
    channel: qcs.Channels,
    acquisition_type: AcquisitionType,
    averaging: bool,
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
        raw = results.get_iq(channel, averaging, acq_index=None)
    elif acquisition_type is AcquisitionType.DISCRIMINATION:
        if averaging:
            raw = results.get_qubit_state_counts(channel, acq_index=None)
        else:
            raw = results.get_classified(channel, acq_index=None)
    else:
        raise ValueError("Acquisition type unrecognized")
    return raw


def parse_result(
    result: np.ndarray, options: ExecutionParameters, singleshot_dim=None
) -> np.ndarray:
    """Parses resulting numpy array into Qibolab expected array shape.

    Arguments:
        result (np.ndarray): Result array from QCS.
        options (ExecutionParameters): Execution settings.
        singleshot_dim (int): Axis position of the number of shots, used for single-shot with hardware sweepers.

    Returns:
        parsed_result (np.ndarray): Parsed numpy array.
    """

    if (
        options.averaging_mode is AveragingMode.SINGLESHOT
        and singleshot_dim is not None
    ):
        # Current result shape is software_sweepers x nshots x hardware_sweepers
        # Qibolab expects the shape of nshots x sweepers
        result = np.moveaxis(result, singleshot_dim, 0)

    # IQ data
    if options.acquisition_type is AcquisitionType.INTEGRATION:
        # Current result dtype is complex, and we need to unwrap it into the I and Q components
        return np.stack([np.real(result), np.imag(result)], axis=-1)

    # Qubit state data
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        # If no averaging was requested, we can pass the data as-is
        if options.averaging_mode is AveragingMode.SINGLESHOT:
            return result
        # Otherwise, the data is in the form of state counts
        else:
            if result.shape == (2,):
                # If there are no sweepers, we can just average directly
                return result[1] / sum(result)
            else:
                # If there are sweepers, the shape is (sweepers x state_count_dim)
                # We average over the last dimension to get the excited state probability
                return result[..., 1] / np.sum(result, axis=-1)

    return result
