from qibolab.pulse.pulse import Pulse


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'ro', offset_i=0, offset_q=0, qubit=0):
        super().__init__(start, duration, amplitude, frequency, phase, shape, channel, type , offset_i, offset_q, qubit)

    @property
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.3f')}, {self.frequency}, {format(self.phase, '.3f')}, '{self.shape}', {self.channel}, '{self.type}')"
