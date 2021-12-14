import math
from qibo import gates
from qibo.config import raise_error


class U3(gates.U3):

    def pulses(self, platform, start, phase):
        # from calibration
        duration = plaform.pi_half_duration
        amplitude = platform.pi_half_amplitude
        frequency = 0 # what is the frequency?

        phase += self.phi - math.pi / 2
        yield pulses.Pulse(start, duration, amplitude, frequency, phase,
                           shape=Gaussian(duration / 5), channel="qcm")
        start += duration # + interval between pulses
        phase += math.pi - self.theta
        yield pulses.Pulse(start, duration, amplitude, frequency, phase,
                           shape=Gaussian(duration / 5), channel="qcm")
        start += duration # + interval
        phase += self.lam - math.pi / 2
