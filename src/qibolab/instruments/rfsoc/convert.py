"""Convert helper functions for rfsoc driver."""

from dataclasses import asdict
from functools import singledispatch

import numpy as np
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses

from qibolab.platform import Qubit
from qibolab.pulses import Pulse, PulseSequence, PulseShape
from qibolab.sweeper import BIAS, DURATION, Parameter, Sweeper

HZ_TO_MHZ = 1e-6
NS_TO_US = 1e-3


def replace_pulse_shape(
    rfsoc_pulse: rfsoc_pulses.Pulse, shape: PulseShape, sampling_rate: float
) -> rfsoc_pulses.Pulse:
    """Set pulse shape parameters in rfsoc_pulses pulse object."""
    if shape.name not in {"Gaussian", "Drag", "Rectangular", "Exponential"}:
        new_pulse = rfsoc_pulses.Arbitrary(
            **asdict(rfsoc_pulse),
            i_values=shape.envelope_waveform_i(sampling_rate),
            q_values=shape.envelope_waveform_q(sampling_rate),
        )
        return new_pulse
    new_pulse_cls = getattr(rfsoc_pulses, shape.name)
    if shape.name == "Rectangular":
        return new_pulse_cls(**asdict(rfsoc_pulse))
    if shape.name == "Gaussian":
        return new_pulse_cls(**asdict(rfsoc_pulse), rel_sigma=shape.rel_sigma)
    if shape.name == "Drag":
        return new_pulse_cls(
            **asdict(rfsoc_pulse), rel_sigma=shape.rel_sigma, beta=shape.beta
        )
    if shape.name == "Exponential":
        return new_pulse_cls(
            **asdict(rfsoc_pulse), tau=shape.tau, upsilon=shape.upsilon, weight=shape.g
        )


def pulse_lo_frequency(pulse: Pulse, qubits: dict[int, Qubit]) -> int:
    """Return local_oscillator frequency (HZ) of a pulse."""
    pulse_type = pulse.type.name.lower()
    try:
        lo_frequency = getattr(
            qubits[pulse.qubit], pulse_type
        ).local_oscillator.frequency
    except AttributeError:
        lo_frequency = 0
    return lo_frequency


def convert_units_sweeper(
    sweeper: rfsoc.Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]
):
    """Convert units for `qibosoq.abstract.Sweeper` considering also LOs."""
    for idx, jdx in enumerate(sweeper.indexes):
        parameter = sweeper.parameters[idx]
        if parameter is rfsoc.Parameter.FREQUENCY:
            pulse = sequence[jdx]
            lo_frequency = pulse_lo_frequency(pulse, qubits)
            sweeper.starts[idx] = (sweeper.starts[idx] - lo_frequency) * HZ_TO_MHZ
            sweeper.stops[idx] = (sweeper.stops[idx] - lo_frequency) * HZ_TO_MHZ
        elif parameter is rfsoc.Parameter.DELAY:
            sweeper.starts[idx] *= NS_TO_US
            sweeper.stops[idx] *= NS_TO_US
        elif parameter is rfsoc.Parameter.RELATIVE_PHASE:
            sweeper.starts[idx] = np.degrees(sweeper.starts[idx])
            sweeper.stops[idx] = np.degrees(sweeper.stops[idx])


@singledispatch
def convert(*args):
    """Convert from qibolab obj to qibosoq obj, overloaded."""
    raise ValueError(f"Convert function received bad parameters ({type(args[0])}).")


@convert.register
def _(qubit: Qubit) -> rfsoc.Qubit:
    """Convert `qibolab.platforms.abstract.Qubit` to
    `qibosoq.abstract.Qubit`."""
    if qubit.flux:
        return rfsoc.Qubit(qubit.flux.offset, qubit.flux.port.name)
    return rfsoc.Qubit(0.0, None)


@convert.register
def _(
    sequence: PulseSequence, qubits: dict[int, Qubit], sampling_rate: float
) -> list[rfsoc_pulses.Pulse]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""
    last_pulse_start = 0
    list_sequence = []
    for pulse in sorted(sequence, key=lambda item: item.start):
        start_delay = (pulse.start - last_pulse_start) * NS_TO_US
        pulse_dict = asdict(convert(pulse, qubits, start_delay, sampling_rate))
        list_sequence.append(pulse_dict)

        last_pulse_start = pulse.start
    return list_sequence


@convert.register
def _(
    pulse: Pulse, qubits: dict[int, Qubit], start_delay: float, sampling_rate: float
) -> rfsoc_pulses.Pulse:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""
    pulse_type = pulse.type.name.lower()
    dac = getattr(qubits[pulse.qubit], pulse_type).port.name
    adc = qubits[pulse.qubit].feedback.port.name if pulse_type == "readout" else None
    lo_frequency = pulse_lo_frequency(pulse, qubits)

    rfsoc_pulse = rfsoc_pulses.Pulse(
        frequency=(pulse.frequency - lo_frequency) * HZ_TO_MHZ,
        amplitude=pulse.amplitude,
        relative_phase=np.degrees(pulse.relative_phase),
        start_delay=start_delay,
        duration=pulse.duration * NS_TO_US,
        dac=dac,
        adc=adc,
        name=pulse.id,
        type=pulse_type,
    )
    return replace_pulse_shape(rfsoc_pulse, pulse.shape, sampling_rate)


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

    if sweeper.parameter is BIAS:
        for qubit in sweeper.qubits:
            parameters.append(rfsoc.Parameter.BIAS)
            indexes.append(list(qubits.values()).index(qubit))
            base_value = qubit.flux.offset
            values = sweeper.get_values(base_value)
            starts.append(values[0])
            stops.append(values[-1])

        if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
            raise ValueError("Sweeper amplitude is set to reach values higher than 1")
    else:
        for pulse in sweeper.pulses:
            idx_sweep = sequence.index(pulse)
            indexes.append(idx_sweep)
            base_value = getattr(pulse, sweeper.parameter.name)
            if idx_sweep != 0 and sweeper.parameter is START:
                # do the conversion from start to delay
                base_value = base_value - sequence[idx_sweep - 1].start
            values = sweeper.get_values(base_value)
            starts.append(values[0])
            stops.append(values[-1])

            if sweeper.parameter is START:
                parameters.append(rfsoc.Parameter.DELAY)
            elif sweeper.parameter is DURATION:
                parameters.append(rfsoc.Parameter.DURATION)
                delta_start = values[0] - base_value
                delta_stop = values[-1] - base_value

                if len(sequence) > idx_sweep + 1:
                    # if duration-swept pulse is not last
                    indexes.append(idx_sweep + 1)
                    t_start = sequence[idx_sweep + 1].start - sequence[idx_sweep].start
                    parameters.append(rfsoc.Parameter.DELAY)
                    starts.append(t_start + delta_start)
                    stops.append(t_start + delta_stop)
            else:
                parameters.append(convert(sweeper.parameter))

    return rfsoc.Sweeper(
        parameters=parameters,
        indexes=indexes,
        starts=starts,
        stops=stops,
        expts=len(sweeper.values),
    )
