"""Convert helper functions for qibosoq driver."""

from dataclasses import asdict
from functools import singledispatch
from typing import Any

import numpy as np
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses
from scipy.constants import mega, micro, nano

from qibolab._core.components.channels import AcquisitionChannel, Channel, IqChannel
from qibolab._core.components.configs import Config, OscillatorConfig
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses.envelope import (
    Drag,
    Envelope,
    Exponential,
    Gaussian,
    Rectangular,
)
from qibolab._core.pulses.pulse import (
    Acquisition,
    Align,
    Delay,
    PulseLike,
    Readout,
    VirtualZ,
)
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
            **asdict(rfsoc_pulse), rel_sigma=1 / envelope.rel_sigma
        )
    if isinstance(envelope, Drag):
        return rfsoc_pulses.Drag(
            **asdict(rfsoc_pulse), rel_sigma=1 / envelope.rel_sigma, beta=envelope.beta
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
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
) -> Sweeper:
    """Convert units for `qibosoq.abstract.Sweeper` considering also LOs."""

    start, stop, step = sweeper.irange
    if sweeper.parameter is Parameter.frequency:
        assert sweeper.channels is not None
        lo_frequency = get_lo_frequency(channels[sweeper.channels[0]], configs)

        new_start = (start - lo_frequency) / mega
        new_stop = (stop - lo_frequency) / mega
        new_step = step / mega

        return Sweeper(
            parameter=sweeper.parameter,
            range=(new_start, new_stop, new_step),
            channels=sweeper.channels,
        )

    new_start, new_stop, new_step = sweeper.irange
    if sweeper.parameter in (Parameter.phase, Parameter.relative_phase):
        new_start = np.degrees(start)
        new_stop = np.degrees(stop)
        new_step = np.degrees(step)

    elif sweeper.parameter is Parameter.duration:
        new_start = start / micro * nano
        new_stop = stop / micro * nano
        new_step = step / micro * nano

    return Sweeper(
        parameter=sweeper.parameter,
        range=(new_start, new_stop, new_step),
        pulses=sweeper.pulses,
    )


@singledispatch
def convert(*args) -> Any:
    """Convert from qibolab obj to qibosoq obj, overloaded."""
    raise ValueError(f"Convert function received bad parameters ({type(args[0])}).")


@convert.register
def _(
    sequence: PulseSequence,
    sampling_rate: float,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
) -> list[rfsoc_pulses.Element]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""

    start_delay = 0
    list_sequence = []
    pulse_sequence = [p for _, p in sequence]
    for ch, pulse in sequence:
        if isinstance(pulse, Delay):
            # multiple consecutive delays are summed
            start_delay += pulse.duration * nano / micro
        elif isinstance(pulse, Align):
            reached = False
            # find last real pulse before align and delay with its duration
            for rev_p in pulse_sequence[::-1]:
                if reached:
                    if not (isinstance(rev_p, Delay) or isinstance(rev_p, Align)):
                        start_delay = rev_p.duration * nano / micro
                        break
                else:
                    reached = rev_p == pulse
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
) -> rfsoc_pulses.Element:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""

    ch = channels[ch_id]
    lo_frequency = get_lo_frequency(ch, configs)

    adc = 0  # Assign adc 0 to ensure frequency matching
    if isinstance(pulse, Acquisition):
        freq = (getattr(configs[ch_id], "frequency") - lo_frequency) / mega
        return rfsoc_pulses.Measurement(
            type="readout",
            frequency=freq,
            start_delay=start_delay,
            duration=pulse.duration * nano / micro,
            dac=0,  # Fix frequency matching to 0-dac
            adc=int(ch.path),
        )
    if isinstance(pulse, Readout):
        assert isinstance(ch, AcquisitionChannel)
        probe_id = ch.probe
        assert probe_id is not None
        probe_ch = channels[probe_id]
        adc = int(probe_ch.path)
        ptype = "readout"
        freq = (getattr(configs[probe_id], "frequency") - lo_frequency) / mega
        amp = pulse.probe.amplitude
        rel_ph = pulse.probe.relative_phase
        envelope = pulse.probe.envelope
    else:
        assert not isinstance(pulse, VirtualZ), (
            "VirtualZ pulse is not convertible to qibosoq.pulse (should not reach this line)"
        )
        assert not isinstance(pulse, Delay), (
            "Delay pulse is currently not convertible to qibosoq.pulse (should not reach this line)"
        )
        assert not isinstance(pulse, Align), (
            "Align pulse is currently not convertible to qibosoq.pulse (should not reach this line)"
        )

        amp = pulse.amplitude
        rel_ph = pulse.relative_phase
        envelope = pulse.envelope
        freq = getattr(configs[ch_id], "frequency", 0)
        ptype = "drive" if freq != 0 else "flux"
        if freq != 0:
            freq = (freq - lo_frequency) / mega

    dac = int(ch.path)  # In any case, add pulse channel for DAC

    rfsoc_pulse = rfsoc_pulses.Pulse(
        frequency=freq,
        amplitude=amp,
        relative_phase=np.degrees(rel_ph),
        start_delay=start_delay,
        duration=pulse.duration * nano / micro,
        dac=dac,
        adc=adc,
        name=str(pulse.id),
        type=ptype,
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

    pulse_sequence = [pulse for _, pulse in sequence]

    parameters = []
    starts = []
    stops = []
    indexes = []
    expts = 0
    for sweeper in sweeperlist:
        start, stop, _ = sweeper.irange
        expts = len(sweeper.values)
        if sweeper.parameter is Parameter.offset:
            assert sweeper.channels is not None
            for ch_id in sweeper.channels:
                parameters.append(rfsoc.Parameter.BIAS)
                indexes.append(int(channels[ch_id].path))
                starts.append(start)
                stops.append(stop)

            if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
                raise ValueError(
                    "Sweeper amplitude is set to reach values higher than 1"
                )

        elif sweeper.parameter is Parameter.frequency:
            assert sweeper.channels is not None
            for ch_id in sweeper.channels:
                parameters.append(rfsoc.Parameter.FREQUENCY)
                pulse = list(sequence.channel(ch_id))[0]
                # TODO what happens if more than one pulse are on the same channel?
                indexes.append(int(channels[ch_id].path))
                starts.append(start)
                stops.append(stop)

        elif sweeper.parameter is Parameter.amplitude:
            assert sweeper.pulses is not None
            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.AMPLITUDE)
                indexes.append(pulse_sequence.index(pulse))
                starts.append(start)
                stops.append(stop)

            if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
                raise ValueError(
                    "Sweeper amplitude is set to reach values higher than 1"
                )

        elif sweeper.parameter in (Parameter.phase, Parameter.relative_phase):
            assert sweeper.pulses is not None
            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.RELATIVE_PHASE)
                indexes.append(pulse_sequence.index(pulse))
                starts.append(start)
                stops.append(stop)

        elif sweeper.parameter is Parameter.duration:
            assert sweeper.pulses is not None
            if not isinstance(sweeper.pulses[0], Delay):
                raise RuntimeError("Only delay sweepers are convertible.")
            pulse_idx = 0
            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.DELAY)

                # counting the index of the pulse (without delay pulses)
                is_this_delay = False
                for p in pulse_sequence:
                    if is_this_delay:
                        if not isinstance(p, Delay):
                            indexes.append(pulse_idx)
                    elif p == pulse:
                        is_this_delay = True
                    if not isinstance(p, Delay):
                        pulse_idx += 1

                starts.append(start)
                stops.append(stop)
        else:
            raise RuntimeError(
                f"In the sweeper conversion function, sweeper of type {type(sweeper)} is not convertible."
            )

    return rfsoc.Sweeper(
        parameters=parameters,
        indexes=indexes,
        starts=np.asarray(starts),
        stops=np.asarray(stops),
        expts=expts,
    )
