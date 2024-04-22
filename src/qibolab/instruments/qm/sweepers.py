import math

import numpy as np
from qibo.config import raise_error
from qm import qua
from qm.qua import declare, fixed, for_
from qualang_tools.loops import from_array

from qibolab.instruments.channels import check_max_offset
from qibolab.instruments.qm.sequence import BakedPulse
from qibolab.pulses import PulseType
from qibolab.sweeper import Parameter


def maximum_sweep_value(values, value0):
    """Calculates maximum value that is reached during a sweep.

    Useful to check whether a sweep exceeds the range of allowed values.
    Note that both the array of values we sweep and the center value can
    be negative, so we need to make sure that the maximum absolute value
    is within range.

    Args:
        values (np.ndarray): Array of values we will sweep over.
        value0 (float, int): Center value of the sweep.
    """
    return max(abs(min(values) + value0), abs(max(values) + value0))


def _update_baked_pulses(sweeper, qmsequence, config):
    """Updates baked pulse if duration sweeper is used."""
    qmpulse = qmsequence.pulse_to_qmpulse[sweeper.pulses[0].id]
    is_baked = isinstance(qmpulse, BakedPulse)
    for pulse in sweeper.pulses:
        qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
        if isinstance(qmpulse, BakedPulse):
            if not is_baked:
                raise_error(
                    TypeError,
                    "Duration sweeper cannot contain both baked and not baked pulses.",
                )
            values = np.array(sweeper.values).astype(int)
            qmpulse.bake(config, values)


def sweep(sweepers, qubits, qmsequence, relaxation_time, config):
    """Public sweep function that is called by the driver."""
    for sweeper in sweepers:
        if sweeper.parameter is Parameter.duration:
            _update_baked_pulses(sweeper, qmsequence, config)
    _sweep_recursion(sweepers, qubits, qmsequence, relaxation_time)


def _sweep_recursion(sweepers, qubits, qmsequence, relaxation_time):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops
    using recursion."""
    if len(sweepers) > 0:
        parameter = sweepers[0].parameter.name
        func_name = f"_sweep_{parameter}"
        if func_name in globals():
            globals()[func_name](sweepers, qubits, qmsequence, relaxation_time)
        else:
            raise_error(
                NotImplementedError, f"Sweeper for {parameter} is not implemented."
            )
    else:
        qmsequence.play(relaxation_time)


def _sweep_frequency(sweepers, qubits, qmsequence, relaxation_time):
    sweeper = sweepers[0]
    freqs0 = []
    for pulse in sweeper.pulses:
        qubit = qubits[pulse.qubit]
        if pulse.type is PulseType.DRIVE:
            lo_frequency = math.floor(qubit.drive.lo_frequency)
        elif pulse.type is PulseType.READOUT:
            lo_frequency = math.floor(qubit.readout.lo_frequency)
        else:
            raise_error(
                NotImplementedError,
                f"Cannot sweep frequency of pulse of type {pulse.type}.",
            )
        # convert to IF frequency for readout and drive pulses
        f0 = math.floor(pulse.frequency - lo_frequency)
        freqs0.append(declare(int, value=f0))
        # check if sweep is within the supported bandwidth [-400, 400] MHz
        max_freq = maximum_sweep_value(sweeper.values, f0)
        if max_freq > 4e8:
            raise_error(
                ValueError,
                f"Frequency {max_freq} for qubit {qubit.name} is beyond instrument bandwidth.",
            )

    # is it fine to have this declaration inside the ``nshots`` QUA loop?
    f = declare(int)
    with for_(*from_array(f, sweeper.values.astype(int))):
        for pulse, f0 in zip(sweeper.pulses, freqs0):
            qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
            qua.update_frequency(qmpulse.element, f + f0)

        _sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)


def _sweep_amplitude(sweepers, qubits, qmsequence, relaxation_time):
    sweeper = sweepers[0]
    # TODO: Consider sweeping amplitude without multiplication
    if min(sweeper.values) < -2:
        raise_error(
            ValueError, "Amplitude sweep values are <-2 which is not supported."
        )
    if max(sweeper.values) > 2:
        raise_error(ValueError, "Amplitude sweep values are >2 which is not supported.")

    a = declare(fixed)
    with for_(*from_array(a, sweeper.values)):
        for pulse in sweeper.pulses:
            qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
            if isinstance(qmpulse, BakedPulse):
                qmpulse.amplitude = a
            else:
                qmpulse.operation = qmpulse.operation * qua.amp(a)

        _sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)


def _sweep_relative_phase(sweepers, qubits, qmsequence, relaxation_time):
    sweeper = sweepers[0]
    relphase = declare(fixed)
    with for_(*from_array(relphase, sweeper.values / (2 * np.pi))):
        for pulse in sweeper.pulses:
            qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
            qmpulse.relative_phase = relphase

        _sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)


def _sweep_bias(sweepers, qubits, qmsequence, relaxation_time):
    sweeper = sweepers[0]
    offset0 = []
    for qubit in sweeper.qubits:
        b0 = qubit.flux.offset
        max_offset = qubit.flux.max_offset
        max_value = maximum_sweep_value(sweeper.values, b0)
        check_max_offset(max_value, max_offset)
        offset0.append(declare(fixed, value=b0))
    b = declare(fixed)
    with for_(*from_array(b, sweeper.values)):
        for qubit, b0 in zip(sweeper.qubits, offset0):
            with qua.if_((b + b0) >= 0.49):
                qua.set_dc_offset(f"flux{qubit.name}", "single", 0.49)
            with qua.elif_((b + b0) <= -0.49):
                qua.set_dc_offset(f"flux{qubit.name}", "single", -0.49)
            with qua.else_():
                qua.set_dc_offset(f"flux{qubit.name}", "single", (b + b0))

        _sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)


def _sweep_start(sweepers, qubits, qmsequence, relaxation_time):
    sweeper = sweepers[0]
    start = declare(int)
    values = (np.array(sweeper.values) // 4).astype(int)

    if len(np.unique(values[1:] - values[:-1])) > 1:
        loop = qua.for_each_(start, values)
    else:
        loop = for_(*from_array(start, values))

    with loop:
        for pulse in sweeper.pulses:
            qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
            # find all pulses that are connected to ``qmpulse`` and update their starts
            to_process = {qmpulse}
            while to_process:
                next_qmpulse = to_process.pop()
                to_process |= next_qmpulse.next_
                next_qmpulse.wait_time_variable = start

        _sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)


def _sweep_duration(sweepers, qubits, qmsequence, relaxation_time):
    sweeper = sweepers[0]
    qmpulse = qmsequence.pulse_to_qmpulse[sweeper.pulses[0].id]
    if isinstance(qmpulse, BakedPulse):
        values = np.array(sweeper.values).astype(int)
    else:
        values = np.array(sweeper.values).astype(int) // 4

    dur = declare(int)
    with for_(*from_array(dur, values)):
        for pulse in sweeper.pulses:
            qmpulse = qmsequence.pulse_to_qmpulse[pulse.id]
            qmpulse.swept_duration = dur
            # find all pulses that are connected to ``qmpulse`` and align them
            if not isinstance(qmpulse, BakedPulse):
                to_process = set(qmpulse.next_)
                while to_process:
                    next_qmpulse = to_process.pop()
                    to_process |= next_qmpulse.next_
                    qmpulse.elements_to_align.add(next_qmpulse.element)
                    next_qmpulse.wait_time -= qmpulse.wait_time + qmpulse.duration // 4

        _sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)
