"""PulseSequence class."""
import numpy as np


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
        t: int = 0
        for pulse in self:
            if pulse.finish > t:
                t = pulse.finish
        return t

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
        """The time when the last pulse of the sequence finishes."""
        channel_pulses = self.pulses_per_channel
        if len(channel_pulses) == 1:
            pulses = next(iter(channel_pulses.values()))
            return sum(pulse.duration for pulse in pulses)
        return max(
            (sequence.duration for sequence in channel_pulses.values()), default=0
        )

    @property
    def channels(self) -> list:
        """List containing the channels used by the pulses in the sequence."""
        channels = []
        for pulse in self:
            if pulse.channel not in channels:
                channels.append(pulse.channel)

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

    def plot(self, savefig_filename=None, sampling_rate=SAMPLING_RATE):
        """Plot the sequence of pulses.

        Args:
            savefig_filename (str): a file path. If provided the plot is save to a file.
        """
        if len(self) > 0:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec

            fig = plt.figure(figsize=(14, 2 * len(self)), dpi=200)
            gs = gridspec.GridSpec(ncols=1, nrows=len(self))
            vertical_lines = []
            for pulse in self:
                vertical_lines.append(pulse.start)
                vertical_lines.append(pulse.finish)

            n = -1
            for qubit in self.qubits:
                qubit_pulses = self.get_qubit_pulses(qubit)
                for channel in qubit_pulses.channels:
                    n += 1
                    channel_pulses = qubit_pulses.get_channel_pulses(channel)
                    ax = plt.subplot(gs[n])
                    ax.axis([0, self.finish, -1, 1])
                    for pulse in channel_pulses:
                        num_samples = len(
                            pulse.shape.modulated_waveform_i(sampling_rate)
                        )
                        time = pulse.start + np.arange(num_samples) / sampling_rate
                        ax.plot(
                            time,
                            pulse.shape.modulated_waveform_q(sampling_rate).data,
                            c="lightgrey",
                        )
                        ax.plot(
                            time,
                            pulse.shape.modulated_waveform_i(sampling_rate).data,
                            c=f"C{str(n)}",
                        )
                        ax.plot(
                            time,
                            pulse.shape.envelope_waveform_i(sampling_rate).data,
                            c=f"C{str(n)}",
                        )
                        ax.plot(
                            time,
                            -pulse.shape.envelope_waveform_i(sampling_rate).data,
                            c=f"C{str(n)}",
                        )
                        # TODO: if they overlap use different shades
                        ax.axhline(0, c="dimgrey")
                        ax.set_ylabel(f"qubit {qubit} \n channel {channel}")
                        for vl in vertical_lines:
                            ax.axvline(vl, c="slategrey", linestyle="--")
                        ax.axis([0, self.finish, -1, 1])
                        ax.grid(
                            visible=True,
                            which="both",
                            axis="both",
                            color="#CCCCCC",
                            linestyle="-",
                        )
            if savefig_filename:
                plt.savefig(savefig_filename)
            else:
                plt.show()
            plt.close()
