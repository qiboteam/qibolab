"""Pulse abstractions."""
import bisect
import itertools
import numpy as np
from qibo.config import raise_error


class PulseSequence:
    # TODO: Move this to a different file (temporarily here as placeholder)

    def __init__(self):
        super().__init__()
        self.qubit_pulses = []
        self.readout_pulses = []

    def add(self, pulse):
        if isinstance(pulse, ReadoutPulse):
            self.readout_pulses.append(pulse)
        else:
            self.qubit_pulses.append(pulse)

    def __len__(self):
        return len(self.qubit_pulses) + len(self.readout_pulses)

    def __iter__(self):
        return itertools.chain(self.qubit_pulses, self.readout_pulses)

    @property
    def start(self):
        if self.qubit_pulses:
            return self.qubit_pulses[0].start
        elif self.readout_pulse is not None:
            return self.readout_pulses[0].start
        else:
            raise_error(ValueError, "Cannot calculate start of an empy pulse sequence.")


class Pulse:
    """Describes a single pulse to be added to waveform array.

    Args:
        start (float): Start time of pulse in ns.
        duration (float): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [0 to 1].
        frequency (float): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        phase (float): TODO.
        shape: (PulseShape): Pulse shape, see :class:`qibolab.pulses.Rectangular`,
        :class:`qibolab.pulses.Gaussian` for more information.
        channel (int): Leftover from IcarusQ.
            May be useful to distinguish QRM and QCM pulses?
        offset_i (float): Optional pulse I offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        offset_q (float): Optional pulse Q offset (unitless).
            (amplitude + offset) should be between [0 and 1].
    """
    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel=0, offset_i=0, offset_q=0):
        # FIXME: Since the ``start`` value depends on the previous pulses we are
        # not sure if it should be a local property of the ``Pulse`` object
        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape  # PulseShape objects

        self.channel = channel
        self.offset_i = offset_i
        self.offset_q = offset_q

    def serial(self):
        return "P({}, {}, {}, {}, {}, {}, {})".format(self.channel, self.start, self.duration,
                                                      self.amplitude, self.frequency, self.phase, self.shape)

    ### IcarusQ specific method ###
    #def compile(self, waveform, sequence):
    #    i_start = bisect.bisect(sequence.time, self.start)
    #    #i_start = int((self.start / sequence.duration) * sequence.sample_size)
    #    i_duration = int((self.duration / sequence.duration) * sequence.sample_size)
    #    time = sequence.time[i_start:i_start + i_duration]
    #    envelope = self.shape.envelope(time, self.start, self.duration, self.amplitude)
    #    waveform[self.channel, i_start:i_start + i_duration] += (
    #        envelope * np.sin(2 * np.pi * self.frequency * time + self.phase))
    #    return waveform

    def compile(self):
        return self.shape.envelope(None, None, self.duration, self.amplitude)

    def __repr__(self):
        return self.serial()


class ReadoutPulse(Pulse):
    """ Describes a pair of IQ pulses for the readout

    Args:
        start (float): Start time of pulse in seconds.
        duration (float): Pulse duration in ns.
        amplitude (float): Pulse amplitude unitless [0 to 1].
        frequency (float): Pulse frequency in Hz.
        phase (float): Pulse phase offset for mixer sideband.
    """

    def __init__(self, start, duration, amplitude, frequency, phase, shape,
                 channel=0, offset_i=0, offset_q=0, delay_before_readout=0):
        super().__init__(start, duration, amplitude, frequency, phase, shape, channel, offset_i, offset_q)

    def serial(self):
        return ""

    ### IcarusQ specific method ###
    #def compile(self, waveform, sequence):
    #    i_start = bisect.bisect(sequence.time, self.start)
        #i_start = int((self.start / sequence.duration) * sequence.sample_size)
    #    i_duration = int((self.duration / sequence.duration) * sequence.sample_size)
    #    time = sequence.time[i_start:i_start + i_duration]
    #    waveform[self.channels[0], i_start:i_start + i_duration] += self.amplitude * np.cos(2 * np.pi * self.frequency * time + self.phases[0])
    #    waveform[self.channels[1], i_start:i_start + i_duration] -= self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phases[1])
    #    return waveform


class IQReadoutPulse(Pulse):
    # TODO: Remove this or think how to merge with ``ReadoutPulse``.
    # Currently keeping it for compatibility with IcarusQ as it breaks the import
    """ Describes a pair of IQ pulses for the readout

    Args:
        channels (int): Pair of FPGA channels to play pulses on.
        start (float): Start time of pulse in seconds.
        duration (float): Pulse duration in seconds.
        amplitude (float): Pulse amplitude in volts.
        frequency (float): Pulse frequency in Hz.
        phases (float): Pulse phase offset for mixer sideband.
    """

    def __init__(self, channels, start, duration, amplitude, frequency, phases):
        self.channels = channels
        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phases = phases

    def serial(self):
        return ""

    def compile(self, waveform, sequence):
        i_start = bisect.bisect(sequence.time, self.start)
        #i_start = int((self.start / sequence.duration) * sequence.sample_size)
        i_duration = int((self.duration / sequence.duration) * sequence.sample_size)
        time = sequence.time[i_start:i_start + i_duration]

        waveform[self.channels[0], i_start:i_start + i_duration] += self.amplitude * np.cos(2 * np.pi * self.frequency * time + self.phases[0])
        waveform[self.channels[1], i_start:i_start + i_duration] -= self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phases[1])

        return waveform


class MultifrequencyPulse(Pulse):
    """Describes multiple pulses to be added to waveform array.

    Used when multiple pulses are overlapping to avoid overwrite
    """
    def __init__(self, members):
        self.members = members

    def serial(self):
        return "M({})".format(", ".join([m.serial() for m in self.members]))

    def compile(self, waveform, sequence):
        for member in self.members:
            waveform += member.compile(waveform, sequence)
        return waveform


class FilePulse(Pulse):
    """Commands the FPGA to load a file as a waveform array in the specified channel
    """
    def __init__(self, channel, start, filename):
        self.channel = channel
        self.start = start
        self.filename = filename

    def serial(self):
        return "F({}, {}, {})".format(self.channel, self.start, self.filename)

    def compile(self, waveform, sequence):
        # `FilePulse` cannot be tested in CI because a file is not available
        i_start = int((self.start / sequence.duration) * sequence.sample_size)
        arr = np.genfromtxt(sequence.file_dir, delimiter=',')[:-1]
        waveform[self.channel, i_start:i_start + len(arr)] = arr
        return waveform
