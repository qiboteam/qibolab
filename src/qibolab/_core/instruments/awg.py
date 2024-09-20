from abc import abstractmethod
from collections.abc import Iterable

import numpy as np

from qibolab._core.components import AcquisitionChannel, Config, DcChannel, IqChannel
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseLike, Waveform
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter


class AWG(Controller):
    """Arbitrary waveform generators (AWGs) are instruments that play an array
    of samples and do not possess hardware sweepers.

    We implement common methods such as waveform generation and
    recursive Python-based software sweepers for ease of driver
    implementation into Qibolab.
    """

    def generate_waveforms(
        self, sequence: PulseSequence, configs: dict[str, Config]
    ) -> dict[ChannelId, Waveform]:
        """Generates waveform arrays for a given pulse sequence."""

        channel_waveforms = {}
        new_sequence = sequence.align_to_delays()
        for channel in new_sequence.channels:
            sequence_duration = new_sequence.channel_duration(channel)
            # Optional if channel has its own individual sampling rate
            sampling_rate = getattr(
                configs[channel], "sampling_rate", self.sampling_rate
            )
            time_interval = 1 / sampling_rate
            num_samples = int(sequence_duration / time_interval)
            pulses = new_sequence.channel(channel)

            instrument_channel = self.channels[channel]
            if isinstance(instrument_channel, DcChannel):
                channel_waveforms[channel] = self.generate_flux_waveform(
                    pulses, sampling_rate, num_samples
                )
            else:
                if isinstance(instrument_channel, AcquisitionChannel):
                    probe_channel = instrument_channel.probe
                    lo = self.channels[probe_channel].lo
                    frequency = configs[probe_channel].frequency
                elif isinstance(instrument_channel, IqChannel):
                    lo = instrument_channel.lo
                    frequency = configs[channel].frequency
                else:
                    raise ValueError("Channel not supported")
                # Downconvert/upconvert as necessary
                if lo is not None:
                    frequency = abs(frequency - configs[lo].frequency)
                channel_waveforms[channel] = self.generate_iq_waveform(
                    pulses, sampling_rate, num_samples, frequency
                )

        return channel_waveforms

    def generate_flux_waveform(
        self, pulses: Iterable[PulseLike], sampling_rate: float, num_samples: int
    ):
        """Generates a waveform for DC flux pulses."""
        buffer = np.zeros(num_samples)
        stopwatch = 0

        for pulse in pulses:

            if pulse.kind == "delay":
                stopwatch += pulse.duration

            else:
                i_envelope = pulse.i(sampling_rate)
                num_pulse_samples = len(i_envelope)
                start = int(stopwatch * sampling_rate)
                end = start + num_pulse_samples
                buffer[start:end] = i_envelope
                stopwatch += pulse.duration

            return buffer

    def generate_iq_waveform(
        self,
        pulses: Iterable[PulseLike],
        sampling_rate: float,
        num_samples: int,
        frequency: float,
    ):
        """Generates an IQ waveform for the given pulses."""
        buffer = np.zeros((num_samples, 2))
        time_interval = 1 / sampling_rate
        stopwatch = 0
        vz_phase = 0

        for pulse in pulses:
            if pulse.kind == "virtualz":
                vz_phase += pulse.phase

            elif pulse.kind == "delay":
                stopwatch += pulse.duration

            else:
                if pulse.kind == "readout":
                    pulse = pulse.acquisition

                i_envelope, q_envelope = pulse.envelopes(sampling_rate)
                num_pulse_samples = len(i_envelope)
                start = int(stopwatch * sampling_rate)
                end = start + num_pulse_samples

                time_array = np.arange(start, end) * time_interval
                angles = (
                    2 * np.pi * frequency * time_array + pulse.relative_phase + vz_phase
                )
                buffer[start:end, 0] = i_envelope * np.cos(angles)
                buffer[start:end, 1] = q_envelope * np.sin(angles)
                stopwatch += pulse.duration
            return buffer

    def recursive_sweep(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        """Sample implementation of recursive sweeping.

        Modify if device/channel implements a certain sweeper type
        """

        parallel_sweeper = sweepers.pop(0)

        # Ensure that parallel sweepers have equal length
        sweeper_length = len(parallel_sweeper[0].values)
        for sweeper in parallel_sweeper:
            if len(sweeper.values) != sweeper_length:
                raise ValueError("Parallel sweepers have unequal length")

        accumulated_results = {}
        # Iterate across the sweeper and play on hardware
        for idx in range(sweeper_length):
            for sweeper in parallel_sweeper:
                value = sweeper.values[idx]

                if sweeper.parameter == Parameter.frequency:
                    for channel in sweeper.channels:
                        # This is OK to mutate, as the actual config is cached by the platform object
                        setattr(configs[channel], sweeper.parameter.name, value)

                elif (
                    sweeper.parameter == Parameter.amplitude
                    or sweeper.parameter == Parameter.duration
                ):
                    for pulse in sweeper.pulses:
                        setattr(pulse, sweeper.parameter.name, value)

                else:
                    raise ValueError(
                        "Sweeper parameter not supported", sweeper.parameter.name
                    )

            accumulated_results = merge_results(
                accumulated_results, self.play(configs, sequences, options, sweepers)
            )

        return accumulated_results

    @abstractmethod
    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        """Play a pulse sequence and retrieve feedback.

        If :class:`qibolab.sweeper.Sweeper` objects are passed as arguments, they are
        executed in real-time. If not possible, an error is raised.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """

        if len(sweepers) != 0:
            return self.recursive_sweep(configs, sequences, options, sweepers)

        for sequence in sequences:
            channel_waveform_map = self.generate_waveforms(sequence, configs)
            # Upload waveforms to instrument and play pulse sequence


def merge_results(result_a: dict[int, Result], result_b: dict[int, Result]):
    """Merge two dictionaries mapping acquisition ids to Result numpy arrays.

    If dict_b has a key (serial) that dict_a does not have, simply add it,
    otherwise concatenate the two arrays

    Args:
        dict_a (dict): dict mapping acquisition ids to Result numpy arrays.
        dict_b (dict): dict mapping acquisition ids to Result numpy arrays.
    Returns:
        A dict mapping acquisition ids to Result numpy arrays.
    """
    for key, value in result_b.items():
        if key in result_a:
            result_a[key] = np.concatenate(result_a[key], value)
        else:
            result_a[key] = value
    return result_a
