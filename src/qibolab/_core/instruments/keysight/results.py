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
) -> np.ndarray:
    """Processes the QCS result object to return the appropiate results.

    Arguments:
        results (qcs.Results): QCS result object.
        channel (qcs.Channels): Virtual acquisition channel to obtain results from.
        acquisition_type (AcquisitionType): Acquisition type to be used.
        averaging (bool): Flag for averaging results
        sweeper_swaps_required (list[tuple[int, int]]): Array of axes pairs corresponding to swapped sweepers.

    Returns:
        raw (np.ndarray): Acquisition results.
    """
    if acquisition_type is AcquisitionType.RAW:
        raw = results.get_trace(channel, averaging)
    elif acquisition_type is AcquisitionType.INTEGRATION:
        raw = results.get_iq(channel, averaging, acq_index=None)
    elif acquisition_type is AcquisitionType.DISCRIMINATION:
        if averaging:
            # At worst, raw currently holds an array with shape of (sweepers, measurements, states)
            raw = results.get_qubit_state_counts(channel, acq_index=None)
            # We shrink the last dimension to be consistent with the IQ acquisition
            raw = {key: val[..., 1] / np.sum(val, axis=-1) for key, val in raw.items()}
        else:
            raw = results.get_classified(channel, avg=False, acq_index=None)
    else:
        raise ValueError("Acquisition type unrecognized")

    # Since we request the results for a single channel, this is a dict with only one key
    return raw[channel]


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

    return result
