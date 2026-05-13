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
        # As of QCS 2.6.4, get_classified does not work for averaging=True, this will be handled by parse_result
        # TODO: Remove this after Keysight patches it
        raw = results.get_classified(channel, False, acq_index=None)
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

    # If the state discrimination is expected, we can directly return the result or average it in software
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        if options.averaging_mode is not AveragingMode.SINGLESHOT:
            # TODO: Refactor this after Keysight patches it
            return (
                np.average(result, axis=0)
                if singleshot_dim is None
                else np.average(result, axis=singleshot_dim)
            )
        else:
            return result

    # Else, the IQ integrated result is expected
    # Current result dtype is complex, and we need to unwrap it into the I and Q components
    return np.stack([np.real(result), np.imag(result)], axis=-1)
