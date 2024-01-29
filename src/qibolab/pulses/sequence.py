"""PulseSequence class."""

from collections import defaultdict

from .pulse import PulseType


class PulseSequence(list):
    """A collection of scheduled pulses.

    A quantum circuit can be translated into a set of scheduled pulses
    that implement the circuit gates. This class contains many
    supporting fuctions to facilitate the creation and manipulation of
    these collections of pulses. None of the methods of PulseSequence
    modify any of the properties of its pulses.
    """

    def __add__(self, other):
        """Return self+value."""
        return type(self)(super().__add__(other))

    def __mul__(self, other):
        """Return self*value."""
        return type(self)(super().__mul__(other))

    def __repr__(self):
        """Return repr(self)."""
        return f"{type(self).__name__}({super().__repr__()})"

    def copy(self):
        """Return a shallow copy of the sequence."""
        return type(self)(super().copy())

    @property
    def ro_pulses(self):
        """A new sequence containing only its readout pulses."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.type == PulseType.READOUT:
                new_pc.append(pulse)
        return new_pc

    @property
    def qd_pulses(self):
        """A new sequence containing only its qubit drive pulses."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.type == PulseType.DRIVE:
                new_pc.append(pulse)
        return new_pc

    @property
    def qf_pulses(self):
        """A new sequence containing only its qubit flux pulses."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.type == PulseType.FLUX:
                new_pc.append(pulse)
        return new_pc

    @property
    def cf_pulses(self):
        """A new sequence containing only its coupler flux pulses."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.type is PulseType.COUPLERFLUX:
                new_pc.append(pulse)
        return new_pc

    def get_channel_pulses(self, *channels):
        """Return a new sequence containing the pulses on some channels."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.channel in channels:
                new_pc.append(pulse)
        return new_pc

    def get_qubit_pulses(self, *qubits):
        """Return a new sequence containing the pulses on some qubits."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.type is not PulseType.COUPLERFLUX:
                if pulse.qubit in qubits:
                    new_pc.append(pulse)
        return new_pc

    def coupler_pulses(self, *couplers):
        """Return a new sequence containing the pulses on some couplers."""
        new_pc = PulseSequence()
        for pulse in self:
            if pulse.type is not PulseType.COUPLERFLUX:
                if pulse.qubit in couplers:
                    new_pc.append(pulse)
        return new_pc

    @property
    def finish(self) -> int:
        """The time when the last pulse of the sequence finishes."""
        channel_pulses = defaultdict(list)
        for pulse in self:
            channel_pulses[pulse.channel].append(pulse)
        return max(
            sum(p.duration for p in pulses) for pulses in channel_pulses.values()
        )

    @property
    def start(self) -> int:
        """The start time of the first pulse of the sequence."""
        t = self.finish
        for pulse in self:
            if pulse.start < t:
                t = pulse.start
        return t

    @property
    def duration(self) -> int:
        """Duration of the sequence calculated as its finish - start times."""
        return self.finish - self.start

    @property
    def channels(self) -> list:
        """List containing the channels used by the pulses in the sequence."""
        channels = []
        for pulse in self:
            if not pulse.channel in channels:
                channels.append(pulse.channel)
        channels.sort()
        return channels

    @property
    def qubits(self) -> list:
        """The qubits associated with the pulses in the sequence."""
        qubits = []
        for pulse in self:
            if not pulse.qubit in qubits:
                qubits.append(pulse.qubit)
        qubits.sort()
        return qubits

    def get_pulse_overlaps(self):  # -> dict((int,int): PulseSequence):
        """Return a dictionary of slices of time (tuples with start and finish
        times) where pulses overlap."""
        times = []
        for pulse in self:
            if not pulse.start in times:
                times.append(pulse.start)
            if not pulse.finish in times:
                times.append(pulse.finish)
        times.sort()

        overlaps = {}
        for n in range(len(times) - 1):
            overlaps[(times[n], times[n + 1])] = PulseSequence()
            for pulse in self:
                if (pulse.start <= times[n]) & (pulse.finish >= times[n + 1]):
                    overlaps[(times[n], times[n + 1])] += [pulse]
        return overlaps

    def separate_overlapping_pulses(self):  # -> dict((int,int): PulseSequence):
        """Separate a sequence of overlapping pulses into a list of non-
        overlapping sequences."""
        # This routine separates the pulses of a sequence into non-overlapping sets
        # but it does not check if the frequencies of the pulses within a set have the same frequency

        separated_pulses = []
        for new_pulse in self:
            stored = False
            for ps in separated_pulses:
                overlaps = False
                for existing_pulse in ps:
                    if (
                        new_pulse.start < existing_pulse.finish
                        and new_pulse.finish > existing_pulse.start
                    ):
                        overlaps = True
                        break
                if not overlaps:
                    ps.append(new_pulse)
                    stored = True
                    break
            if not stored:
                separated_pulses.append(PulseSequence([new_pulse]))
        return separated_pulses

    # TODO: Implement separate_different_frequency_pulses()

    @property
    def pulses_overlap(self) -> bool:
        """Whether any of the pulses in the sequence overlap."""
        overlap = False
        for pc in self.get_pulse_overlaps().values():
            if len(pc) > 1:
                overlap = True
                break
        return overlap
