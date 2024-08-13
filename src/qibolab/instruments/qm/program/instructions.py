from typing import Optional

from qm import qua
from qm.qua import declare, fixed, for_
from qualang_tools.loops import from_array

from qibolab.components import Config
from qibolab.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab.pulses import Delay
from qibolab.sweeper import ParallelSweepers

from ..config import operation
from .acquisition import Acquisition
from .arguments import ExecutionArguments, Parameters
from .sweepers import INT_TYPE, NORMALIZERS, SWEEPER_METHODS


def _delay(pulse: Delay, element: str, parameters: Parameters):
    # TODO: How to play delays on multiple elements?
    if parameters.duration is None:
        duration = int(pulse.duration) // 4
    else:
        duration = parameters.duration
    qua.wait(duration + 1, element)


def _play_multiple_waveforms(element: str, parameters: Parameters):
    """Sweeping pulse duration using distinctly uploaded waveforms."""
    with qua.switch_(parameters.duration, unsafe=True):
        for value, sweep_op in parameters.pulses:
            with qua.case_(value // 4):
                qua.play(sweep_op, element)


def _play(
    op: str,
    element: str,
    parameters: Parameters,
    acquisition: Optional[Acquisition] = None,
):
    if parameters.phase is not None:
        qua.frame_rotation_2pi(parameters.phase, element)
    if parameters.amplitude is not None:
        op = op * parameters.amplitude

    if len(parameters.pulses) > 0:
        _play_multiple_waveforms(element, parameters)
    else:
        if acquisition is not None:
            acquisition.measure(op)
        else:
            qua.play(op, element, duration=parameters.duration)

    if parameters.phase is not None:
        qua.reset_frame(element)


def play(args: ExecutionArguments):
    """Part of QUA program that plays an arbitrary pulse sequence.

    Should be used inside a ``program()`` context.
    """
    qua.align()
    for channel_id, pulse in args.sequence:
        element = str(channel_id)
        op = operation(pulse)
        params = args.parameters[op]
        if isinstance(pulse, Delay):
            _delay(pulse, element, params)
        elif isinstance(pulse, Pulse):
            acquisition = args.acquisitions.get((op, element))
            _play(op, element, params, acquisition)

    if args.relaxation_time > 0:
        qua.wait(args.relaxation_time // 4)


def _sweep_recursion(sweepers, configs, args):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops
    using recursion."""
    if len(sweepers) > 0:
        parallel_sweepers = sweepers[0]
        if len(parallel_sweepers) > 1:
            raise NotImplementedError

        sweeper = parallel_sweepers[0]
        parameter = sweeper.parameter
        if parameter not in SWEEPER_METHODS:
            raise NotImplementedError(f"Sweeper for {parameter} is not implemented.")

        variable = declare(int) if parameter in INT_TYPE else declare(fixed)
        values = sweeper.values
        if parameter in NORMALIZERS:
            values = NORMALIZERS[parameter](sweeper.values)

        method = SWEEPER_METHODS[parameter]
        with for_(*from_array(variable, values)):
            if sweeper.pulses is not None:
                method(sweeper.pulses, values, variable, configs, args)
            else:
                method(sweeper.channels, values, variable, configs, args)

            _sweep_recursion(sweepers[1:], configs, args)

    else:
        play(args)


def sweep(
    sweepers: list[ParallelSweepers],
    configs: dict[str, Config],
    args: ExecutionArguments,
):
    """Public sweep function that is called by the driver."""
    # for sweeper in sweepers:
    #    if sweeper.parameter is Parameter.duration:
    #        _update_baked_pulses(sweeper, instructions, config)
    _sweep_recursion(sweepers, configs, args)


def program(
    configs: dict[str, Config],
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
            sweep(list(sweepers), configs, args)
        # download acquisitions
        has_iq = options.acquisition_type is AcquisitionType.INTEGRATION
        buffer_dims = options.results_shape(sweepers)[::-1][int(has_iq) :]
        with qua.stream_processing():
            for acquisition in args.acquisitions.values():
                acquisition.download(*buffer_dims)
    return experiment
