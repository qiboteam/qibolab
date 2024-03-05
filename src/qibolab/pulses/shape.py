"""Library of pulse shapes."""

from abc import ABC
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

__all__ = [
    "Times",
    "Waveform",
    "IqWaveform",
    "Envelope",
    "Envelopes",
]

Times = npt.NDArray[np.float64]
"""The time window of a pulse.

This should span the entire pulse interval, and contain one point per-desired sample.

.. note::

    It is not possible to deal with partial windows or arrays with a rank different from
    1, since some envelopes are defined in terms of the pulse duration and the
    individual samples themselves. Cf. :cls:`Snz`.
"""
# TODO: they could be distinguished among them, and distinguished from generic float
# arrays, using the NewType pattern -> but this require some more effort to encforce
# types throughout the whole code base
Waveform = npt.NDArray[np.float64]
""""""
IqWaveform = npt.NDArray[np.float64]
""""""


class Envelope(ABC):
    """Pulse envelopes.

    Generates both i (in-phase) and q (quadrature) components.
    """

    def i(self, times: Times) -> Waveform:
        """In-phase envelope."""
        return np.zeros_like(times)

    def q(self, times: Times) -> Waveform:
        """Quadrature envelope."""
        return np.zeros_like(times)

    def envelopes(self, times: Times) -> IqWaveform:
        """Stacked i and q envelope waveforms of the pulse."""
        return np.array(self.i(times), self.q(times))


@dataclass(frozen=True)
class Rectangular(Envelope):
    """Rectangular envelope."""

    amplitude: float

    def i(self, times: Times) -> Waveform:
        """Generate a rectangular envelope."""
        return self.amplitude * np.ones_like(times)


@dataclass(frozen=True)
class Exponential(Envelope):
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


def _gaussian(t, mu, sigma):
    """Gaussian function, normalized to be 1 at the max."""
    # TODO: if a centered Gaussian has to be used, and we agree that `Times` should
    # always be the full window, just use `scipy.signal.gaussian`, that is exactly this
    # function, autcomputing the mean from the number of points
    return np.exp(-(((t - mu) / sigma) ** 2) / 2)


@dataclass(frozen=True)
class Gaussian(Envelope):
    r"""Gaussian pulse shape.

    Args:
        rel_sigma (float):

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
    """

    amplitude: float
    mu: float
    """Gaussian mean."""
    sigma: float
    """Gaussian standard deviation."""

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian window."""
        return self.amplitude * _gaussian(times, self.mu, self.sigma)


@dataclass(frozen=True)
class GaussianSquare(Envelope):
    r"""GaussianSquare pulse shape.

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Rise] + Flat + A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Decay]
    """

    amplitude: float
    mu: float
    """Gaussian mean."""
    sigma: float
    """Gaussian standard deviation."""
    width: float
    """Length of the flat portion."""

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian envelope, with a flat central window."""

        pulse = np.ones_like(times)
        u, hw = self.mu, self.width / 2
        tails = (times < (u - hw)) | ((u + hw) < times)
        pulse[tails] = _gaussian(times[tails], self.mu, self.sigma)

        return self.amplitude * pulse


@dataclass(frozen=True)
class Drag(Envelope):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape.

    .. todo::

        - add expression
        - add reference
    """

    amplitude: float
    mu: float
    """Gaussian mean."""
    sigma: float
    """Gaussian standard deviation."""
    beta: float
    """.. todo::"""

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian envelope."""
        return self.amplitude * _gaussian(times, self.mu, self.sigma)

    def q(self, times: Times) -> Waveform:
        """Generate ...

        .. todo::
        """
        i = self.amplitude * _gaussian(times, self.mu, self.sigma)
        return self.beta * (-(times - self.mu) / (self.sigma**2)) * i


@dataclass(frozen=True)
class Iir(Envelope):
    """IIR Filter using scipy.signal lfilter."""

    # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
    # p = [A, tau_iir]
    # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
    # p = [b0, b1, a0, a1]

    amplitude: float
    a: npt.NDArray
    b: npt.NDArray
    target: Envelope

    def _data(self, target):
        a = self.a / self.a[0]
        gain = np.sum(self.b) / np.sum(a)
        b = self.b / gain if gain != 0 else self.b

        data = lfilter(b=b, a=a, x=target)
        if np.max(np.abs(data)) != 0:
            data = data / np.max(np.abs(data))
        return data

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        return self.amplitude * self._data(self.target.i(times))

    def q(self, times: Times) -> Waveform:
        """.. todo::"""
        return self.amplitude * self._data(self.target.q(times))


@dataclass(frozen=True)
class Snz(Envelope):
    """Sudden variant Net Zero.

    https://arxiv.org/abs/2008.07411
    (Supplementary materials: FIG. S1.)

    .. todo::

        - expression
    """

    amplitude: float
    width: float
    """Essentially, the pulse duration...

    .. todo::

        - reset to duration, if decided so
    """
    t_idling: float
    b_amplitude: float = 0.5
    """Relative B amplitude (wrt A)."""

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        # convert timings to samples
        half_pulse_duration = (self.width - self.t_idling) / 2
        aspan = np.sum(times < half_pulse_duration)
        idle = len(times) - 2 * (aspan + 1)

        pulse = np.ones_like(times)
        # the aspan + 1 sample is B (and so the aspan + 1 + idle + 1), indexes are 0-based
        pulse[aspan] = pulse[aspan + 1 + idle] = self.b_amplitude
        # set idle time to 0
        pulse[aspan + 1 : aspan + 1 + idle] = 0
        return self.amplitude * pulse


@dataclass(frozen=True)
class ECap(Envelope):
    r"""ECap pulse shape.

    .. todo::

        - add reference

    .. math::

        e_{\cap(t,\alpha)} &=& A[1 + \tanh(\alpha t/t_\theta)][1 + \tanh(\alpha (1 - t/t_\theta))]\\
        &\times& [1 + \tanh(\alpha/2)]^{-2}
    """

    amplitude: float
    alpha: float

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        x = times / len(times)
        return (
            self.amplitude
            * (1 + np.tanh(self.alpha * times))
            * (1 + np.tanh(self.alpha * (1 - x)))
            / (1 + np.tanh(self.alpha / 2)) ** 2
        )


@dataclass(frozen=True)
class Custom(Envelope):
    """Arbitrary shape.

    .. todo::

        - expand description
        - add attribute docstrings
    """

    amplitude: float
    custom_i: npt.NDArray
    custom_q: npt.NDArray

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        return self.amplitude * self.custom_i

    def envelope_waveform_q(self, times: Times) -> Waveform:
        """.. todo::"""
        return self.amplitude * self.custom_q


class Envelopes(Enum):
    """Available pulse shapes."""

    RECTANGULAR = Rectangular
    EXPONENTIAL = Exponential
    GAUSSIAN = Gaussian
    GAUSSIAN_SQUARE = GaussianSquare
    DRAG = Drag
    IIR = Iir
    SNZ = Snz
    ECAP = ECap
    CUSTOM = Custom
