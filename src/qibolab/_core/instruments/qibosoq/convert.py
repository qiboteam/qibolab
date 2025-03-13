"""Convert helper functions for qibosoq driver."""

from copy import deepcopy
from dataclasses import asdict
from functools import singledispatch
from typing import Any, cast

import numpy as np
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses
from scipy.constants import mega, micro, nano

from qibolab._core.components.configs import Config, DcConfig
from qibolab._core.pulses.envelope import (
    Drag,
    Envelope,
    Exponential,
    Gaussian,
    Rectangular,
)
from qibolab._core.qubits import Qubit
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper


def replace_pulse_shape(
    rfsoc_pulse: rfsoc_pulses.Pulse, envelope: Envelope, samples: int
) -> rfsoc_pulses.Pulse:
    """Set pulse shape parameters in rfsoc_pulses pulse object."""
    if isinstance(envelope, Rectangular):
        return rfsoc_pulses.Rectangular(**asdict(rfsoc_pulse))
    if isinstance(envelope, Gaussian):
        return rfsoc_pulses.Gaussian(
            **asdict(rfsoc_pulse), rel_sigma=envelope.rel_sigma
        )
    if isinstance(envelope, Drag):
        return rfsoc_pulses.Drag(
            **asdict(rfsoc_pulse), rel_sigma=envelope.rel_sigma, beta=envelope.beta
        )
    if isinstance(envelope, Exponential):
        return rfsoc_pulses.FluxExponential(
            **asdict(rfsoc_pulse),
            tau=envelope.tau,
            upsilon=envelope.upsilon,
            weight=envelope.g,
        )

    return rfsoc_pulses.Arbitrary(
        **asdict(rfsoc_pulse),
        i_values=envelope.i(samples).tolist(),
        q_values=envelope.q(samples).tolist(),
    )


def convert_units_sweeper(
    sweeper: rfsoc.Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]
) -> rfsoc.Sweeper:
    """Convert units for `qibosoq.abstract.Sweeper` considering also LOs."""
    sweeper = deepcopy(sweeper)
    for idx, jdx in enumerate(sweeper.indexes):
        parameter = sweeper.parameters[idx]
        if parameter is rfsoc.Parameter.FREQUENCY:
            lo_frequency = ...
            sweeper.starts[idx] = (sweeper.starts[idx] - lo_frequency) / mega
            sweeper.stops[idx] = (sweeper.stops[idx] - lo_frequency) / mega
        elif parameter is rfsoc.Parameter.DELAY:
            sweeper.starts[idx] *= nano / micro
            sweeper.stops[idx] *= nano / micro
        elif parameter is rfsoc.Parameter.RELATIVE_PHASE:
            sweeper.starts[idx] = np.degrees(sweeper.starts[idx])
            sweeper.stops[idx] = np.degrees(sweeper.stops[idx])
    return sweeper


@singledispatch
def convert(*args) -> Any:
    """Convert from qibolab obj to qibosoq obj, overloaded."""
    raise ValueError(f"Convert function received bad parameters ({type(args[0])}).")


@convert.register
def _(qubit: Qubit, configs: dict[str, Config], ports: dict[str, int]) -> rfsoc.Qubit:
    """Convert `qibolab.platforms.abstract.Qubit` to
    `qibosoq.abstract.Qubit`."""
    if qubit.flux is not None:
        flux = cast(DcConfig, configs[qubit.flux])
        return rfsoc.Qubit(flux.offset, ports[qubit.flux])
    return rfsoc.Qubit(0.0, None)


@convert.register
def _(
    sequence: PulseSequence, qubits: dict[int, Qubit], sampling_rate: float
) -> list[rfsoc_pulses.Pulse]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""
    # last_pulse_start = 0
    list_sequence = []
    # for pulse in sorted(sequence):
    #     start_delay = (pulse.start - last_pulse_start) * nano / micro
    #     pulse_dict = asdict(convert(pulse, qubits, start_delay, sampling_rate))
    #     list_sequence.append(pulse_dict)
    #
    #     last_pulse_start = pulse.start
    return list_sequence


# @convert.register
# def _(
#     pulse: Pulse, qubits: dict[int, Qubit], start_delay: float, sampling_rate: float
# ) -> rfsoc_pulses.Pulse:
#     """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""
#     pulse_type = pulse.type.name.lower()
#     dac = getattr(qubits[pulse.qubit], pulse_type).port.name
#     adc = qubits[pulse.qubit].feedback.port.name if pulse_type == "readout" else None
#     lo_frequency = pulse_lo_frequency(pulse, qubits)
#
#     rfsoc_pulse = rfsoc_pulses.Pulse(
#         frequency=(pulse.frequency - lo_frequency) / mega,
#         amplitude=pulse.amplitude,
#         relative_phase=np.degrees(pulse.relative_phase),
#         start_delay=start_delay,
#         duration=pulse.duration * nano / micro,
#         dac=dac,
#         adc=adc,
#         name=pulse.serial,
#         type=pulse_type,
#     )
#     return replace_pulse_shape(rfsoc_pulse, pulse.shape, sampling_rate)


@convert.register
def _(par: Parameter) -> rfsoc.Parameter:
    """Convert a qibolab sweeper.Parameter into a qibosoq.Parameter."""
    return getattr(rfsoc.Parameter, par.name.upper())


@convert.register
def _(
    sweeper: Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]
) -> rfsoc.Sweeper:
    """Convert `qibolab.sweeper.Sweeper` to `qibosoq.abstract.Sweeper`.

    Note that any unit conversion is not done in this function (to avoid
    to do it multiple times). Conversion will be done in
    `convert_units_sweeper`.
    """
    parameters = []
    starts = []
    stops = []
    indexes = []

    if sweeper.parameter is Parameter.offset:
        # for qubit in sweeper.qubits:
        #     parameters.append(rfsoc.Parameter.BIAS)
        #     indexes.append(list(qubits.values()).index(qubit))
        #     base_value = qubit.flux.offset
        #     values = sweeper.get_values(base_value)
        #     starts.append(values[0])
        #     stops.append(values[-1])

        if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
            raise ValueError("Sweeper amplitude is set to reach values higher than 1")
    else:
        # for pulse in sweeper.pulses:
        #     idx_sweep = sequence.index(pulse)
        #     indexes.append(idx_sweep)
        #     start, stop, step = sweeper.irange
        #     starts.append(start)
        #     stops.append(step)
        #
        #     if sweeper.parameter is Parameter.duration:
        #         parameters.append(rfsoc.Parameter.DURATION)
        #
        #         if len(sequence) > idx_sweep + 1:
        #             # if duration-swept pulse is not last
        #             indexes.append(idx_sweep + 1)
        #             t_start = sequence[idx_sweep + 1].start - sequence[idx_sweep].start
        #             parameters.append(rfsoc.Parameter.DELAY)
        #             starts.append(t_start + delta_start)
        #             stops.append(t_start + delta_stop)
        #     else:
        #         parameters.append(convert(sweeper.parameter))
        pass

    return rfsoc.Sweeper(
        parameters=parameters,
        indexes=indexes,
        starts=np.asarray(starts),
        stops=np.asarray(stops),
        expts=0,  # len(sweeper.values),
    )
