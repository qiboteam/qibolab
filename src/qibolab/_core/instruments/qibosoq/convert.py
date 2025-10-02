"""Convert helper functions for qibosoq driver."""

from dataclasses import asdict
from functools import singledispatch
from typing import Any

import numpy as np
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses
from scipy.constants import mega, micro, nano

from qibolab._core.components.channels import (
    AcquisitionChannel,
    Channel,
    DcChannel,
    IqChannel,
)
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
    PulseId,
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
    new_start, new_stop, new_step = sweeper.irange
    if sweeper.parameter is Parameter.frequency:
        assert sweeper.channels is not None
        lo_frequency = get_lo_frequency(channels[sweeper.channels[0]], configs)

        new_start = (start - lo_frequency) / mega
        new_stop = (stop - lo_frequency) / mega
        new_step = step / mega

    elif sweeper.parameter in (Parameter.phase, Parameter.relative_phase):
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
        channels=sweeper.channels,
    )


def order_pulse_sequence(
    sequence: PulseSequence,
) -> tuple[list[tuple[ChannelId, PulseLike, float]], dict[PulseId, list[PulseId]]]:
    """Order pulse sequence by execution time."""

    channel_time: dict[str, float] = {}
    channel_order = {ch: i for i, ch in enumerate(sequence.by_channel.keys())}
    result = []

    delay_equivalence = {}

    for ch, subseq in sequence.by_channel.items():
        channel_time[ch] = 0
        delays_before_pulse = []
        for idx, pulse in enumerate(subseq):
            if isinstance(pulse, Delay):
                start = channel_time[ch]
                channel_time[ch] = stop = start + pulse.duration
                # result.append((ch, pulse, start, stop, idx))
                delays_before_pulse.append(pulse.id)

            elif isinstance(pulse, Align):
                start = channel_time[ch]
                stop = max(channel_time.values())
                for c in channel_time:
                    channel_time[c] = stop
                delays_before_pulse.append(pulse.id)
                # result.append((ch, pulse, start, stop, idx))

            else:
                start = channel_time[ch]
                channel_time[ch] = stop = start + pulse.duration
                result.append((ch, pulse, start, stop, idx))
                if len(delays_before_pulse) > 0:
                    delay_equivalence[pulse.id] = delays_before_pulse
                delays_before_pulse = []

    def sort_key(x):
        """Sort pulse first by time and then by type (delays/align later)."""
        ch, _, start, stop, idx = x
        return (start, stop, idx, channel_order[ch])

    result.sort(key=sort_key)

    rt_result = []
    rst = 0  # relative start time
    last_start = 0
    for ch, pulse, start, stop, _ in result:
        rst = start - last_start
        rt_result.append((ch, pulse, rst))
        last_start = start

    return rt_result, delay_equivalence


def simplify_delays(
    timed_sequence: list[tuple[ChannelId, PulseLike, float]],
) -> tuple[list[tuple[ChannelId, PulseLike, float]], dict[PulseId, list[PulseId]]]:
    """Merge consecutive delays at the same start time by subtracting the minimum delay."""
    compressed = []
    delay_equivalence = {}
    last_id = None

    last_delay = 0
    for ch, pulse, t in timed_sequence:
        if not isinstance(pulse, Delay):
            compressed.append((ch, pulse, t))
            last_delay = 0
            last_id = None
            continue
        if pulse.duration <= last_delay:
            if last_id is None:
                delay_equivalence[pulse.id] = [pulse.id]
                last_id = pulse.id
            else:
                delay_equivalence[last_id].append(pulse.id)
            continue
        if pulse.duration > last_delay:
            new_duration = pulse.duration - last_delay
            last_delay = pulse.duration
            pulse = Delay(duration=new_duration, id_=pulse.id)
            compressed.append((ch, pulse, t))

            if last_id is None:
                delay_equivalence[pulse.id] = [pulse.id]
                last_id = pulse.id
            else:
                delay_equivalence[last_id].append(pulse.id)
    return compressed, delay_equivalence


def get_index(sequence: list[PulseLike], pulse: PulseLike) -> int:
    """Get pulse index from a sequence."""
    for idx, p in enumerate(sequence):
        if p.id == pulse.id:
            return idx
    raise RuntimeError("Pulse not in sequence.")


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

    list_sequence = []
    ordered_sequence, _ = order_pulse_sequence(sequence)

    for ch, pulse, rst in ordered_sequence:
        pulse_dict = asdict(
            convert(pulse, rst * nano / micro, ch, channels, sampling_rate, configs)
        )
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
        # probe_ch = channels[probe_id]
        adc = int(ch.path)
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

    rfsoc_pulse = rfsoc_pulses.Pulse(
        frequency=freq,
        amplitude=amp,
        relative_phase=np.degrees(rel_ph),
        start_delay=start_delay,
        duration=pulse.duration * nano / micro,
        dac=int(ch.path),  # Add DAC in any case, for frequency matching
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

    ordered_sequence, delay_equivalence = order_pulse_sequence(sequence)
    pulse_sequence = [pulse for _, pulse, _ in ordered_sequence]

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
                qubit_idx = 0
                for ch in channels:
                    if isinstance(channels[ch], DcChannel):
                        if channels[ch] == channels[ch_id]:
                            indexes.append(qubit_idx)
                        else:
                            qubit_idx += 1
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
                pulse = [p for ch, p, _ in ordered_sequence if ch == ch_id][0]
                # TODO what happens if more than one pulse are on the same channel?
                indexes.append(get_index(pulse_sequence, pulse))
                starts.append(start)
                stops.append(stop)

        elif sweeper.parameter is Parameter.amplitude:
            assert sweeper.pulses is not None
            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.AMPLITUDE)
                indexes.append(get_index(pulse_sequence, pulse))
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
                indexes.append(get_index(pulse_sequence, pulse))
                starts.append(start)
                stops.append(stop)

        elif sweeper.parameter is Parameter.duration:
            assert sweeper.pulses is not None
            if not isinstance(sweeper.pulses[0], Delay):
                raise RuntimeError("Only delay sweepers are convertible.")

            for pulse in sweeper.pulses:
                parameters.append(rfsoc.Parameter.DELAY)
                # counting the index of the pulse (without delay pulses)
                for idx, p in enumerate(pulse_sequence):
                    if p.id in delay_equivalence:
                        if pulse.id in delay_equivalence[p.id]:
                            indexes.append(idx)

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
