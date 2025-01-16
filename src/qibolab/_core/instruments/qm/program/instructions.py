from typing import Optional

import numpy as np
import numpy.typing as npt
from qm import qua
from qm.qua import Cast, Variable, declare, fixed, for_, for_each_

from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Align, Delay, Pulse, Readout, VirtualZ
from qibolab._core.sweeper import ParallelSweepers, Parameter, Sweeper

from ..config import operation
from .acquisition import Acquisition
from .arguments import ExecutionArguments, Parameters
from .sweepers import INT_TYPE, NORMALIZERS, SWEEPER_METHODS, normalize_phase


def _delay(pulse: Delay, element: str, parameters: Parameters):
    # TODO: How to play delays on multiple elements?
    if parameters.duration is None:
        duration = max(int(pulse.duration) // 4 + 1, 4)
        qua.wait(duration, element)
    elif parameters.interpolated:
        duration = parameters.duration + 1
        qua.wait(duration, element)
    else:
        duration = parameters.duration / 4
        with qua.if_(duration < 4):
            qua.wait(4, element)
        with qua.else_():
            qua.wait(duration, element)


def _play_multiple_waveforms(element: str, parameters: Parameters):
    """Sweeping pulse duration using distinctly uploaded waveforms."""
    assert not parameters.interpolated
    assert parameters.interpolated_op is None
    with qua.switch_(parameters.duration, unsafe=True):
        for value, sweep_op in parameters.duration_ops:
            if parameters.amplitude is not None:
                sweep_op = sweep_op * parameters.amplitude
            with qua.case_(value):
                qua.play(sweep_op, element)


def _play_single_waveform(
    op: str,
    element: str,
    parameters: Parameters,
    acquisition: Optional[Acquisition] = None,
):
    if parameters.amplitude is not None:
        op = parameters.amplitude_op * parameters.amplitude
    if acquisition is not None:
        acquisition.measure(op)
    else:
        if parameters.duration is not None:
            # sweeping duration using interpolation
            # distinctly uploaded waveforms are handled by ``_play_multiple_waveforms``
            assert len(parameters.duration_ops) == 0
            qua.play(parameters.interpolated_op, element, duration=parameters.duration)
        else:
            qua.play(op, element)


def _play(
    op: str,
    element: str,
    parameters: Parameters,
    acquisition: Optional[Acquisition] = None,
):
    if parameters.phase is not None:
        qua.frame_rotation_2pi(parameters.phase, element)

    if len(parameters.duration_ops) > 0:
        _play_multiple_waveforms(element, parameters)
    else:
        _play_single_waveform(op, element, parameters, acquisition)

    if parameters.phase is not None:
        qua.reset_frame(element)


def play(args: ExecutionArguments):
    """Part of QUA program that plays an arbitrary pulse sequence.

    Should be used inside a ``program()`` context.
    """
    qua.align()

    # keep track of ``Align`` command that were already played
    # because the same ``Align`` will appear on multiple channels
    # in the sequence
    processed_aligns = set()

    for channel_id, pulse in args.sequence:
        element = str(channel_id)
        op = operation(pulse)
        params = args.parameters[pulse.id]
        if isinstance(pulse, Delay):
            _delay(pulse, element, params)
        elif isinstance(pulse, Pulse):
            _play(op, element, params)
        elif isinstance(pulse, Readout):
            acquisition = args.acquisitions.get((op, element))
            _play(op, element, params, acquisition)
        elif isinstance(pulse, VirtualZ):
            qua.frame_rotation_2pi(normalize_phase(pulse.phase), element)
        elif isinstance(pulse, Align) and pulse.id not in processed_aligns:
            channel_ids = args.sequence.pulse_channels(pulse.id)
            qua.align(*(str(ch) for ch in channel_ids))
            processed_aligns.add(pulse.id)

    if args.relaxation_time > 0:
        qua.wait(args.relaxation_time // 4)


def _process_sweeper(sweeper: Sweeper, args: ExecutionArguments):
    parameter = sweeper.parameter
    if parameter not in SWEEPER_METHODS:
        raise NotImplementedError(f"Sweeper for {parameter} is not implemented.")

    if parameter in INT_TYPE:
        variable = declare(int)
        values = sweeper.values.astype(int)
    else:
        variable = declare(fixed)
        values = sweeper.values

    if parameter is Parameter.frequency:
        lo_frequency = args.parameters[sweeper.channels[0]].lo_frequency
        values = NORMALIZERS[parameter](values, lo_frequency)
    elif parameter in NORMALIZERS:
        values = NORMALIZERS[parameter](values)

    return variable, values


def _qua_for_loop(variables: list[Variable], values: list[npt.NDArray]):
    """Generate QUA ``for_`` loop command for the sweeps.

    Partly copied from ``qualang_tools.from_array``.
    """
    if len(variables) > 1:
        return for_each_(variables, values)

    var = variables[0]
    array = values[0]

    if len(array) == 0:
        raise ValueError("Sweeper values must have length > 0.")
    elif len(array) == 1:
        return for_(var, array[0], var <= array[0], var + 1)

    if not isinstance(var, Variable):
        raise TypeError("The first argument must be a QUA variable.")
    if (not isinstance(array[0], (np.generic, int, float))) or (
        isinstance(array[0], bool)
    ):
        raise TypeError("The array must be an array of python variables.")

    start = array[0]
    stop = array[-1]
    if var.is_fixed():
        if not (-8 <= start < 8) or not (-8 <= stop < 8):
            raise ValueError("fixed numbers are bounded to [-8, 8).")
    elif not var.is_int():
        raise TypeError(
            "This variable type is not supported. Please use a QUA 'int' or 'fixed' when sweeping."
        )

    # Linear increment
    if np.isclose(np.std(np.diff(array)), 0) == "lin":
        step = array[1] - array[0]

        if var.is_int():
            # Check that the array is an array of int with integer increments
            if not all(float(x).is_integer() for x in (step, start, stop)):
                raise TypeError(
                    "When looping over a QUA int variable, the step and array elements must be integers."
                )

            if step > 0:
                return for_(var, int(start), var <= int(stop), var + int(step))
            else:
                return for_(var, int(start), var >= int(stop), var + int(step))

        elif var.is_fixed():
            # Generate the loop parameters for positive and negative steps
            if step > 0:
                return for_(
                    var,
                    float(start),
                    var < float(stop) + float(step) / 2,
                    var + float(step),
                )
            else:
                return for_(
                    var,
                    float(start),
                    var > float(stop) + float(step) / 2,
                    var + float(step),
                )

    # Logarithmic increment
    if np.isclose(np.std(array[1:] / array[:-1]), 0, atol=1e-3):
        step = array[1] / array[0]

        if var.is_int():
            if step > 1:
                if int(round(start) * float(step)) == int(round(start)):
                    return for_each_(var, array)
                else:
                    return for_(
                        var,
                        round(start),
                        var < round(stop) * np.sqrt(float(step)),
                        Cast.mul_int_by_fixed(var, float(step)),
                    )
            else:
                return for_(
                    var,
                    round(start),
                    var > round(stop) / np.sqrt(float(step)),
                    Cast.mul_int_by_fixed(var, float(step)),
                )

        elif var.is_fixed():
            if step > 1:
                return for_(
                    var,
                    float(start),
                    var < float(stop) * np.sqrt(float(step)),
                    var * float(step),
                )
            else:
                return for_(
                    var,
                    float(start),
                    var > float(stop) * np.sqrt(float(step)),
                    var * float(step),
                )

    # Non-(linear or logarithmic) increment requires ``for_each_``
    return for_each_(var, array)


def sweep(
    sweepers: list[ParallelSweepers],
    args: ExecutionArguments,
):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops.

    Uses recursion to handle nested sweepers.
    """
    if len(sweepers) > 0:
        parallel_sweepers = sweepers[0]

        variables, values = zip(
            *(_process_sweeper(sweeper, args) for sweeper in parallel_sweepers)
        )
        loop = _qua_for_loop(variables, values)

        with loop:
            for sweeper, variable in zip(parallel_sweepers, variables):
                method = SWEEPER_METHODS[sweeper.parameter]
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        params = args.parameters[pulse.id]
                        method(variable, params)
                else:
                    for channel in sweeper.channels:
                        params = args.parameters[channel]
                        method(variable, params)

            sweep(sweepers[1:], args)

    else:
        play(args)


def program(
    args: ExecutionArguments,
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
    offsets: list[tuple[ChannelId, float]],
):
    """QUA program implementing the required experiment."""
    with qua.program() as experiment:
        # FIXME: force offset setting due to a bug in QUA 1.2.1a2 and OPX1000
        for channel_id, offset in offsets:
            qua.set_dc_offset(channel_id, "single", offset)

        n = declare(int)
        # declare acquisition variables
        for acquisition in args.acquisitions.values():
            acquisition.declare()
        # execute pulses
        with for_(n, 0, n < options.nshots, n + 1):
            sweep(list(sweepers), args)
        # download acquisitions
        has_iq = options.acquisition_type is AcquisitionType.INTEGRATION
        buffer_dims = options.results_shape(sweepers)[::-1][int(has_iq) :]
        with qua.stream_processing():
            for acquisition in args.acquisitions.values():
                acquisition.download(*buffer_dims)
    return experiment
