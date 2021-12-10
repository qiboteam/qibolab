import numpy as np
from abc import ABC, abstractmethod
from qibo.config import raise_error


class PulseShape(ABC):
    """Describes the pulse shape to be used
    """
    def __init__(self): # pragma: no cover
        self.name = ""

    @abstractmethod
    def envelope(self, time, start, duration, amplitude): # pragma: no cover
        raise_error(NotImplementedError)

    def __repr__(self):
        return "({})".format(self.name)


class Rectangular(PulseShape):
    """Rectangular/square pulse shape
    """
    def __init__(self):
        self.name = "rectangular"

    def envelope(self, time, start, duration, amplitude):
        """Constant amplitude envelope
        """
        #return amplitude
        # FIXME: This may have broken IcarusQ
        return amplitude * np.ones(int(duration))


class Gaussian(PulseShape):
    """Gaussian pulse shape"""

    def __init__(self, sigma):
        self.name = "gaussian"
        self.sigma = sigma

    def envelope(self, time, start, duration, amplitude):
        """Gaussian envelope centered with respect to the pulse:
        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
        """
        from scipy.signal import gaussian
        return amplitude * gaussian(int(duration), std=self.sigma)
        # FIXME: This may have broken IcarusQ
        #mu = start + duration / 2
        #return amplitude * np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)

    def __repr__(self):
        return "({}, {})".format(self.name, self.sigma)


class Drag(PulseShape):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape"""

    def __init__(self, sigma, beta):
        self.name = "drag"
        self.sigma = sigma
        self.beta = beta

    def envelope(self, time, start, duration, amplitude):
        """DRAG envelope centered with respect to the pulse:
        G + i\beta(-\frac{t-\mu}{\sigma^2})G
        where Gaussian G = A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
        """
        mu = start + duration / 2
        gaussian = amplitude * np.exp(-0.5 * (time - mu) ** 2 / self.sigma ** 2)
        return gaussian + 1j * self.beta * (-(time - mu) / self.sigma ** 2) * gaussian

    def __repr__(self):
        return "({}, {}, {})".format(self.name, self.sigma, self.beta)


class SWIPHT(PulseShape):
    """Speeding up Wave forms by Inducing Phase to Harmful Transitions pulse shape"""

    def __init__(self, g):
        self.name = "SWIPHT"
        self.g = g

    def envelope(self, time, start, duration, amplitude):

        ki_qq = self.g * np.pi
        t_g = 5.87 / (2 * abs(ki_qq))
        t = np.linspace(0, t_g, len(time))

        gamma = 138.9 * (t / t_g)**4 *(1 - t / t_g)**4 + np.pi / 4
        gamma_1st = 4 * 138.9 * (t / t_g)**3 * (1 - t / t_g)**3 * (1 / t_g - 2 * t / t_g**2)
        gamma_2nd = 4*138.9*(t / t_g)**2 * (1 - t / t_g)**2 * (14*(t / t_g**2)**2 - 14*(t / t_g**3) + 3 / t_g**2)
        omega = gamma_2nd / np.sqrt(ki_qq**2 - gamma_1st**2) - 2*np.sqrt(ki_qq**2 - gamma_1st**2) * 1 / np.tan(2 * gamma)
        omega = omega / max(omega)

        return omega * amplitude

    def __repr__(self):
        return "({}, {})".format(self.name, self.g)
