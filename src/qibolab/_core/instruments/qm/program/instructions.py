from typing import Optional

from qm import qua
from qm.qua import declare, fixed, for_
from qualang_tools.loops import from_array

from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
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
        qua.play(op, element, duration=parameters.duration)


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
        params = args.parameters[op]
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

    variable = declare(int) if parameter in INT_TYPE else declare(fixed)
    values = sweeper.values
    if parameter is Parameter.frequency:
        lo_frequency = args.parameters[sweeper.channels[0]].lo_frequency
        values = NORMALIZERS[parameter](values, lo_frequency)
    elif parameter in NORMALIZERS:
        values = NORMALIZERS[parameter](values)

    return variable, values


def sweep(
    sweepers: list[ParallelSweepers],
    args: ExecutionArguments,
):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops.

    Uses recursion to handle nested sweepers.
    """
    if len(sweepers) > 0:
        parallel_sweepers = sweepers[0]

        variables, all_values = zip(
            *(_process_sweeper(sweeper, args) for sweeper in parallel_sweepers)
        )
        if len(parallel_sweepers) > 1:
            loop = qua.for_each_(variables, all_values)
        else:
            loop = for_(*from_array(variables[0], all_values[0]))

        with loop:
            for sweeper, variable, values in zip(
                parallel_sweepers, variables, all_values
            ):
                method = SWEEPER_METHODS[sweeper.parameter]
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        params = args.parameters[operation(pulse)]
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
):
    """QUA program implementing the required experiment."""
    with qua.program() as experiment:
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
