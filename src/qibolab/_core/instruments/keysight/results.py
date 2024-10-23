"""Utils for result management."""

import numpy as np
from keysight import qcs

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.sweeper import ParallelSweepers


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
        raw = results.get_iq(channel, averaging)
    elif acquisition_type is AcquisitionType.DISCRIMINATION:
        raw = results.get_classified(channel, averaging)
    else:
        raise ValueError("Acquisition type unrecognized")
    return raw


def parse_result(
    result: np.ndarray, options: ExecutionParameters, sweepers: list[ParallelSweepers]
) -> np.ndarray:
    """Parses resulting numpy array into Qibolab expected array shape.

    Arguments:
        result (np.ndarray): Result array from QCS.
        options (ExecutionParameters): Execution settings.
        sweepers (list[ParallelSweepers]): Array of array of sweepers.

    Returns:
        parsed_result (np.ndarray): Parsed numpy array.
    """
    # For single shot, qibolab expects result format (nshots, ...)
    # QCS returns (..., nshots), so we need to shuffle the arrays
    if options.averaging_mode is AveragingMode.SINGLESHOT and len(sweepers) > 0:
        tmp = np.zeros(options.results_shape(sweepers))
        # For IQ data, QCS returns complex results
        if options.acquisition_type is AcquisitionType.INTEGRATION:
            for k in range(options.nshots):
                tmp[k, ..., 0] = np.real(result[..., k])
                tmp[k, ..., 1] = np.imag(result[..., k])
        else:
            for k in range(options.nshots):
                tmp[k, ...] = result[..., k]

    # For IQ data, QCS returns complex results
    elif options.acquisition_type is AcquisitionType.INTEGRATION:
        tmp = np.zeros(result.shape + (2,))
        tmp[..., 0] = np.real(result)
        tmp[..., 1] = np.imag(result)
    else:
        return result
    return tmp
