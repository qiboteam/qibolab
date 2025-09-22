"""Convert helper functions for qibosoq driver."""

from copy import deepcopy
from dataclasses import asdict
from functools import singledispatch
from typing import Any, cast

import numpy as np
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses
from scipy.constants import mega, micro, nano

from qibolab._core.components.channels import AcquisitionChannel, Channel, IqChannel
from qibolab._core.components.configs import Config, DcConfig, OscillatorConfig
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses.envelope import (
    Drag,
    Envelope,
    Exponential,
    Gaussian,
    Rectangular,
)
from qibolab._core.pulses.pulse import Delay, PulseLike
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


def get_lo_frequency(ch: Channel, configs: dict[str, Config]) -> float:
    """Return LO frequency from channel, if applicable."""
    if isinstance(ch, IqChannel):
        if ch.lo is not None:
            conf = configs[ch.lo]
            if isinstance(conf, OscillatorConfig):
                return conf.frequency
    return 0


def convert_units_sweeper(
    sweeper: Sweeper,
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
) -> Sweeper:
    """Convert units for `qibosoq.abstract.Sweeper` considering also LOs."""
    sweeper = deepcopy(sweeper)

    # if sweeper.parameter is Parameter.delay TODO

    if sweeper.parameter is Parameter.frequency:
        start, stop, step = sweeper.irange

        assert sweeper.channels is not None
        lo_frequency = get_lo_frequency(channels[sweeper.channels[0]], configs)

        new_start = (start - lo_frequency) / mega
        new_stop = (start - lo_frequency) / mega
        new_step = step / mega

        sweeper.range = (new_start, new_stop, new_step)

    elif sweeper.parameter in (Parameter.phase, Parameter.relative_phase):
        start, stop, step = sweeper.irange

        new_start = np.degrees(start)
        new_stop = np.degrees(stop)
        new_step = np.degrees(step)

        sweeper.range = (new_start, new_stop, new_step)
        return sweeper

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
    sequence: PulseSequence,
    sampling_rate: float,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
) -> list[rfsoc_pulses.Pulse]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""

    start_delay = 0
    list_sequence = []
    for ch, pulse in sequence:
        if isinstance(pulse, Delay):
            start_delay = pulse.duration * nano / micro
        else:
            pulse_dict = asdict(
                convert(pulse, start_delay, ch, channels, sampling_rate, configs)
            )
            start_delay = 0
            list_sequence.append(pulse_dict)

    return list_sequence


@convert.register
def _(
    pulse: PulseLike,
    start_delay: float,
    ch_id: ChannelId,
    channels: dict[ChannelId, Channel],
    sampling_rate: float,
    configs: dict[str, Config],
) -> rfsoc_pulses.Pulse:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""

    ch = channels[ch_id]

    adc = None  # Maybe I always need to assign an adc, in order to force correct freqs
    if pulse.kind in ("readout", "acquisition"):
        assert isinstance(ch, AcquisitionChannel)
        probe_id = ch.probe
        assert probe_id is not None
        probe_ch = channels[probe_id]
        adc = int(probe_ch.path)

        amp = pulse.probe.amplitude
        rel_ph = pulse.probe.relative_phase
        envelope = pulse.probe.envelope
        type = "readout"
    else:
        amp = pulse.amplitude
        rel_ph = pulse.relative_phase
        envelope = pulse.envelope
        type = "drive"

    dac = int(ch.path)  # In any case, add pulse channel for DAC
    lo_frequency = get_lo_frequency(ch, configs)

    freq = getattr(configs[ch_id], "frequency", 0)
    freq = (freq - lo_frequency) / mega

    rfsoc_pulse = rfsoc_pulses.Pulse(
        frequency=freq,
        amplitude=amp,
        relative_phase=np.degrees(rel_ph),
        start_delay=start_delay,
        duration=pulse.duration * nano / micro,
        dac=dac,
        adc=adc,
        name=str(pulse.id),
        type=type,
    )
    num_samples = int(sampling_rate * pulse.duration)
    return replace_pulse_shape(rfsoc_pulse, envelope, num_samples)


@convert.register
def _(par: Parameter) -> rfsoc.Parameter:
    """Convert a qibolab sweeper.Parameter into a qibosoq.Parameter."""
    return getattr(rfsoc.Parameter, par.name.upper())


@convert.register
def _(
    sweeperlist: list,  # Should be ParallelSweepers, but functools encounters an error
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
) -> rfsoc.Sweeper:
    """Convert `qibolab.sweeper.Sweeper` to `qibosoq.abstract.Sweeper`.

    Note that any unit conversion is not done in this function (to avoid
    to do it multiple times). Conversion will be done in
    `convert_units_sweeper`.
    """

    pulse_sequence = [pulse for ch, pulse in sequence]

    parameters = []
    starts = []
    stops = []
    indexes = []
    expts = 0
    for sweeper in sweeperlist:
        if sweeper.parameter is Parameter.offset:
            assert sweeper.channels is not None
            for ch_id in sweeper.channels:
                parameters.append(rfsoc.Parameter.BIAS)
                indexes.append(int(channels[ch_id].path))
                start, stop, step = sweeper.irange
                starts.append(start)
                stops.append(stop)
                expts = len(sweeper.values)

            if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
                raise ValueError(
                    "Sweeper amplitude is set to reach values higher than 1"
                )

        elif sweeper.parameter is Parameter.frequency:
            assert sweeper.channels is not None
            for ch_id in sweeper.channels:
                parameters.append(rfsoc.Parameter.FREQUENCY)

                pulse = sequence.channel(ch_id)[0]
                # TODO what happens if more than one pulse are on the same channel?
                indexes.append(int(channels[ch_id].path))

                start, stop, step = sweeper.irange
                starts.append(start)
                stops.append(stop)
                expts = len(sweeper.values)

        elif sweeper.parameter is Parameter.amplitude:
            assert sweeper.pulses is not None
            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.AMPLITUDE)
                indexes.append(pulse_sequence.index(pulse))
                start, stop, step = sweeper.irange
                starts.append(start)
                stops.append(stop)
                expts = len(sweeper.values)

            if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
                raise ValueError(
                    "Sweeper amplitude is set to reach values higher than 1"
                )

        elif sweeper.parameter in (Parameter.phase, Parameter.relative_phase):
            assert sweeper.pulses is not None
            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.RELATIVE_PHASE)
                indexes.append(pulse_sequence.index(pulse))
                start, stop, step = sweeper.irange
                starts.append(start)
                stops.append(stop)
                expts = len(sweeper.values)
        # elif sweeper.parameter is Parameter.duration:
        else:
            # pass
            raise RuntimeError(
                "In sweeper conversion function, I received a non-convertible sweeper."
            )

    return rfsoc.Sweeper(
        parameters=parameters,
        indexes=indexes,
        starts=np.asarray(starts),
        stops=np.asarray(stops),
        expts=expts,
    )
