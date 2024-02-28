"""PulseShape class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

SAMPLING_RATE = 1
"""Default sampling rate in gigasamples per second (GSps).

Used for generating waveform envelopes if the instruments do not provide
a different value.
"""

Times = npt.NDArray[np.float64]
# TODO: they could be distinguished among them, and distinguished from generic float
# arrays, using the NewType pattern -> but this require some more effort to encforce
# types throughout the whole code base
Waveform = npt.NDArray[np.float64]
""""""
IqWaveform = npt.NDArray[np.float64]
""""""


def modulate(
    envelope: IqWaveform,
    freq: float,
    rate: float = SAMPLING_RATE,
    phase: float = 0.0,
) -> IqWaveform:
    """Modulate the envelope waveform with a carrier.

    `envelope` is a `(2, n)`-shaped array of I and Q (first dimension) envelope signals,
    as a function of time (second dimension), and `freq` the frequency of the carrier to
    modulate with (usually the IF) in GHz.
    `rate` is an optional sampling rate, in Gs/s, to sample the carrier.

    .. note::

        Only the combination `freq / rate` is actually relevant, but it is frequently
        convenient to specify one in GHz and the other in Gs/s. Thus the two arguments
        are provided for the simplicity of their interpretation.

    `phase` is an optional initial phase for the carrier.
    """
    samples = np.arange(envelope.shape[1])
    phases = (2 * np.pi * freq / rate) * samples + phase
    cos = np.cos(phases)
    sin = np.sin(phases)
    mod = np.array([[cos, -sin], [sin, cos]])

    # the normalization is related to `mod`, but only applied at the end for the sake of
    # performances
    return np.einsum("ijt,jt->it", mod, envelope) / np.sqrt(2)


def demodulate(
    modulated: IqWaveform,
    freq: float,
    rate: float = SAMPLING_RATE,
) -> IqWaveform:
    """Demodulate the acquired pulse.

    The role of the arguments is the same of the corresponding ones in :func:`modulate`,
    which is essentially the inverse of this function.
    """
    # in case the offsets have not been removed in hardware
    modulated = modulated - np.mean(modulated)

    samples = np.arange(modulated.shape[1])
    phases = (2 * np.pi * freq / rate) * samples
    cos = np.cos(phases)
    sin = np.sin(phases)
    demod = np.array([[cos, sin], [-sin, cos]])

    # the normalization is related to `demod`, but only applied at the end for the sake
    # of performances
    return np.sqrt(2) * np.einsum("ijt,jt->it", demod, modulated)


class Shape(ABC):
    """Pulse envelopes.

    Generates both i (in-phase) and q (quadrature) components.
    """

    @abstractmethod
    def i(self, times: Times) -> Waveform:
        """In-phase envelope."""

    @abstractmethod
    def q(self, times: Times) -> Waveform:
        """Quadrature envelope."""

    def envelopes(self, times: Times) -> IqWaveform:
        """Stacked i and q envelope waveforms of the pulse."""
        return np.array(self.i(times), self.q(times))


@dataclass(frozen=True)
class Rectangular(Shape):
    """Rectangular envelope."""

    amplitude: float

    def i(self, times: Times) -> Waveform:
        """Generate a rectangular envelope."""
        return self.amplitude * np.ones_like(times)

    def q(self, times: Times) -> Waveform:
        """Generate an identically null signal."""
        return np.zeros_like(times)


@dataclass(frozen=True)
class Exponential(Shape):
    r"""Exponential shape, i.e. square pulse with an exponential decay.

    .. math::

        A\frac{\exp\left(-\frac{x}{\text{upsilon}}\right) + g \exp\left(-\frac{x}{\text{tau}}\right)}{1 + g}
    """

    amplitude: float
    tau: float
    """The decay rate of the first exponential function."""
    upsilon: float
    """The decay rate of the second exponential function."""
    g: float = 0.1
    """Weight of the second exponential function."""

    def i(self, times: Times) -> Waveform:
        """Generate a combination of two exponential decays."""
        return (
            self.amplitude
            * (np.exp(-times / self.upsilon) + self.g * np.exp(-times / self.tau))
            / (1 + self.g)
        )

    def q(self, times: Times) -> Waveform:
        """Generate an identically null signal."""
        return np.zeros_like(times)


@dataclass(frozen=True)
class Gaussian(Shape):
    r"""Gaussian pulse shape.

    Args:
        rel_sigma (float):

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
    """

    amplitude: float
    mu: float
    sigma: float
    """Relative standard deviation.

    The pulse standard deviation will then be `sigma = duration /
    rel_sigma`.
    """

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian window."""
        return self.amplitude * np.exp(-(((times - self.mu) / self.sigma) ** 2) / 2)

    def q(self, times: Times) -> Waveform:
        """Generate an indentically null signal."""
        return np.zeros_like(times)


class Shapes(Enum):
    """Available pulse shapes."""

    RECTANGULAR = Rectangular
    EXPONENTIAL = Exponential
    GAUSSIAN = Gaussian


class GaussianSquare(Shape):
    r"""GaussianSquare pulse shape.

    Args:
        rel_sigma (float): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma
        width (float): Percentage of the pulse that is flat

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Rise] + Flat + A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Decay]
    """

    def __init__(self, rel_sigma: float, width: float):
        self.name = "GaussianSquare"
        self.pulse: "Pulse" = None
        self.rel_sigma: float = float(rel_sigma)
        self.width: float = float(width)

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        if super().__eq__(item):
            return self.rel_sigma == item.rel_sigma and self.width == item.width
        return False

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        def gaussian(t, rel_sigma, gaussian_samples):
            mu = (2 * gaussian_samples - 1) / 2
            sigma = (2 * gaussian_samples) / rel_sigma
            return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

        def fvec(t, gaussian_samples, rel_sigma, length=None):
            if length is None:
                length = t.shape[0]

            pulse = np.ones_like(t, dtype=float)
            rise = t < gaussian_samples
            fall = t > length - gaussian_samples - 1
            pulse[rise] = gaussian(t[rise], rel_sigma, gaussian_samples)
            pulse[fall] = gaussian(t[rise], rel_sigma, gaussian_samples)[::-1]
            return pulse

        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        gaussian_samples = num_samples * (1 - self.width) // 2
        t = np.arange(0, num_samples)

        pulse = fvec(t, gaussian_samples, rel_sigma=self.rel_sigma)

        return self.pulse.amplitude * pulse

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        return np.zeros(num_samples)

    def __repr__(self):
        return f"{self.name}({format(self.rel_sigma, '.6f').rstrip('0').rstrip('.')}, {format(self.width, '.6f').rstrip('0').rstrip('.')})"


class Drag(PulseShape):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape.

    Args:
        rel_sigma (float): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma
        beta (float): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma
    .. math::
    """

    def __init__(self, rel_sigma, beta):
        self.name = "Drag"
        self.pulse: "Pulse" = None
        self.rel_sigma = float(rel_sigma)
        self.beta = float(beta)

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        if super().__eq__(item):
            return self.rel_sigma == item.rel_sigma and self.beta == item.beta
        return False

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        x = np.arange(0, num_samples, 1)
        return self.pulse.amplitude * np.exp(
            -(1 / 2)
            * (
                ((x - (num_samples - 1) / 2) ** 2)
                / (((num_samples) / self.rel_sigma) ** 2)
            )
        )

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        x = np.arange(0, num_samples, 1)
        i = self.pulse.amplitude * np.exp(
            -(1 / 2)
            * (
                ((x - (num_samples - 1) / 2) ** 2)
                / (((num_samples) / self.rel_sigma) ** 2)
            )
        )
        return (
            self.beta
            * (-(x - (num_samples - 1) / 2) / ((num_samples / self.rel_sigma) ** 2))
            * i
            * sampling_rate
        )

    def __repr__(self):
        return f"{self.name}({format(self.rel_sigma, '.6f').rstrip('0').rstrip('.')}, {format(self.beta, '.6f').rstrip('0').rstrip('.')})"


class IIR(PulseShape):
    """IIR Filter using scipy.signal lfilter."""

    # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
    # p = [A, tau_iir]
    # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
    # p = [b0, b1, a0, a1]

    def __init__(self, b, a, target: PulseShape):
        self.name = "IIR"
        self.target: PulseShape = target
        self._pulse: "Pulse" = None
        self.a: np.ndarray = np.array(a)
        self.b: np.ndarray = np.array(b)
        # Check len(a) = len(b) = 2

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        if super().__eq__(item):
            return (
                self.target == item.target
                and (self.a == item.a).all()
                and (self.b == item.b).all()
            )
        return False

    @property
    def pulse(self):
        return self._pulse

    @pulse.setter
    def pulse(self, value):
        self._pulse = value
        self.target.pulse = value

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        self.a = self.a / self.a[0]
        gain = np.sum(self.b) / np.sum(self.a)
        if not gain == 0:
            self.b = self.b / gain
        data = lfilter(
            b=self.b,
            a=self.a,
            x=self.target.envelope_waveform_i(sampling_rate),
        )
        if not np.max(np.abs(data)) == 0:
            data = data / np.max(np.abs(data))
        return np.abs(self.pulse.amplitude) * data

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        self.a = self.a / self.a[0]
        gain = np.sum(self.b) / np.sum(self.a)
        if not gain == 0:
            self.b = self.b / gain
        data = lfilter(
            b=self.b,
            a=self.a,
            x=self.target.envelope_waveform_q(sampling_rate),
        )
        if not np.max(np.abs(data)) == 0:
            data = data / np.max(np.abs(data))
        return np.abs(self.pulse.amplitude) * data

    def __repr__(self):
        formatted_b = [round(b, 3) for b in self.b]
        formatted_a = [round(a, 3) for a in self.a]
        return f"{self.name}({formatted_b}, {formatted_a}, {self.target})"


class SNZ(PulseShape):
    """Sudden variant Net Zero.

    https://arxiv.org/abs/2008.07411
    (Supplementary materials: FIG. S1.)
    """

    def __init__(self, t_idling, b_amplitude=None):
        self.name = "SNZ"
        self.pulse: "Pulse" = None
        self.t_idling: float = t_idling
        self.b_amplitude = b_amplitude

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        if super().__eq__(item):
            return (
                self.t_idling == item.t_idling and self.b_amplitude == item.b_amplitude
            )
        return False

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""
        if self.t_idling > self.pulse.duration:
            raise ValueError(
                f"Cannot put idling time {self.t_idling} higher than duration {self.pulse.duration}."
            )
        if self.b_amplitude is None:
            self.b_amplitude = self.pulse.amplitude / 2
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        half_pulse_duration = (self.pulse.duration - self.t_idling) / 2
        half_flux_pulse_samples = int(
            np.rint(num_samples * half_pulse_duration / self.pulse.duration)
        )
        idling_samples = num_samples - 2 * half_flux_pulse_samples
        return np.concatenate(
            (
                self.pulse.amplitude * np.ones(half_flux_pulse_samples - 1),
                np.array([self.b_amplitude]),
                np.zeros(idling_samples),
                -np.array([self.b_amplitude]),
                -self.pulse.amplitude * np.ones(half_flux_pulse_samples - 1),
            )
        )

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))
        return np.zeros(num_samples)

    def __repr__(self):
        return f"{self.name}({self.t_idling})"


class eCap(PulseShape):
    r"""ECap pulse shape.

    Args:
        alpha (float):

    .. math::

        e_{\cap(t,\alpha)} &=& A[1 + \tanh(\alpha t/t_\theta)][1 + \tanh(\alpha (1 - t/t_\theta))]\\
        &\times& [1 + \tanh(\alpha/2)]^{-2}
    """

    def __init__(self, alpha: float):
        self.name = "eCap"
        self.pulse: "Pulse" = None
        self.alpha: float = float(alpha)

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        if super().__eq__(item):
            return self.alpha == item.alpha
        return False

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        num_samples = int(self.pulse.duration * sampling_rate)
        x = np.arange(0, num_samples, 1)
        return (
            self.pulse.amplitude
            * (1 + np.tanh(self.alpha * x / num_samples))
            * (1 + np.tanh(self.alpha * (1 - x / num_samples)))
            / (1 + np.tanh(self.alpha / 2)) ** 2
        )

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        num_samples = int(self.pulse.duration * sampling_rate)
        return np.zeros(num_samples)

    def __repr__(self):
        return f"{self.name}({format(self.alpha, '.6f').rstrip('0').rstrip('.')})"


class Custom(PulseShape):
    """Arbitrary shape."""

    def __init__(self, envelope_i, envelope_q=None):
        self.name = "Custom"
        self.pulse: "Pulse" = None
        self.envelope_i: np.ndarray = np.array(envelope_i)
        if envelope_q is not None:
            self.envelope_q: np.ndarray = np.array(envelope_q)
        else:
            self.envelope_q = self.envelope_i

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""
        if self.pulse.duration != len(self.envelope_i):
            raise ValueError("Length of envelope_i must be equal to pulse duration")
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))

        return self.envelope_i * self.pulse.amplitude

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""
        if self.pulse.duration != len(self.envelope_q):
            raise ValueError("Length of envelope_q must be equal to pulse duration")
        num_samples = int(np.rint(self.pulse.duration * sampling_rate))

        return self.envelope_q * self.pulse.amplitude

    def __repr__(self):
        return f"{self.name}({self.envelope_i[:3]}, ..., {self.envelope_q[:3]}, ...)"
