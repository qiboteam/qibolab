"""Library of pulse shapes."""

from abc import ABC
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter
from scipy.signal.windows import gaussian

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


def _duration(times: Times) -> float:
    return times[-1] - times[0]


def _mean(times: Times) -> float:
    return _duration(times) / 2 + times[0]


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
        return np.array([self.i(times), self.q(times)])


@dataclass(frozen=True)
class Rectangular(Envelope):
    """Rectangular envelope."""

    def i(self, times: Times) -> Waveform:
        """Generate a rectangular envelope."""
        return np.ones_like(times)


@dataclass(frozen=True)
class Exponential(Envelope):
    r"""Exponential shape, i.e. square pulse with an exponential decay.

    .. math::

        A\frac{\exp\left(-\frac{x}{\text{upsilon}}\right) + g \exp\left(-\frac{x}{\text{tau}}\right)}{1 + g}
    """

    tau: float
    """The decay rate of the first exponential function."""
    upsilon: float
    """The decay rate of the second exponential function."""
    g: float = 0.1
    """Weight of the second exponential function."""

    def i(self, times: Times) -> Waveform:
        """Generate a combination of two exponential decays."""
        return (np.exp(-times / self.upsilon) + self.g * np.exp(-times / self.tau)) / (
            1 + self.g
        )


def _samples_sigma(rel_sigma: float, times: Times) -> float:
    """Convert standard deviation in samples.

    `rel_sigma` is assumed in units of the interval duration, and it is turned in units
    of samples, by counting the number of samples in the interval.
    """
    return rel_sigma * len(times)


@dataclass(frozen=True)
class Gaussian(Envelope):
    r"""Gaussian pulse shape.

    Args:
        rel_sigma (float):

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
    """

    rel_sigma: float
    """Relative Gaussian standard deviation.

    In units of the interval duration.
    """

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian window."""
        return gaussian(len(times), _samples_sigma(self.rel_sigma, times))


@dataclass(frozen=True)
class GaussianSquare(Envelope):
    r"""GaussianSquare pulse shape.

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Rise] + Flat + A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Decay]
    """

    rel_sigma: float
    """Relative Gaussian standard deviation.

    In units of the interval duration.
    """
    width: float
    """Length of the flat portion."""

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian envelope, with a flat central window."""

        pulse = np.ones_like(times)
        u, hw = _mean(times), self.width / 2
        tails = (times < (u - hw)) | ((u + hw) < times)
        pulse[tails] = gaussian(
            len(times[tails]), _samples_sigma(self.rel_sigma, times)
        )

        return pulse


@dataclass(frozen=True)
class Drag(Envelope):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse shape.

    .. todo::

        - add expression
        - add reference
    """

    rel_sigma: float
    """Relative Gaussian standard deviation.

    In units of the interval duration.
    """
    beta: float
    """.. todo::"""

    def i(self, times: Times) -> Waveform:
        """Generate a Gaussian envelope."""
        return gaussian(len(times), _samples_sigma(self.rel_sigma, times))

    def q(self, times: Times) -> Waveform:
        """Generate ...

        .. todo::
        """
        sigma = self.rel_sigma * _duration(times)
        return self.beta * (-(times - _mean(times)) / (sigma**2)) * self.i(times)


@dataclass(frozen=True)
class Iir(Envelope):
    """IIR Filter using scipy.signal lfilter.

    https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)::

        p = [A, tau_iir]
        p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
        p = [b0, b1, a0, a1]
    """

    a: npt.NDArray
    b: npt.NDArray
    target: Envelope

    def _data(self, target: npt.NDArray) -> npt.NDArray:
        a = self.a / self.a[0]
        gain = np.sum(self.b) / np.sum(a)
        b = self.b / gain if gain != 0 else self.b

        data = lfilter(b=b, a=a, x=target)
        if np.max(np.abs(data)) != 0:
            data /= np.max(np.abs(data))
        return data

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        return self._data(self.target.i(times))

    def q(self, times: Times) -> Waveform:
        """.. todo::"""
        return self._data(self.target.q(times))


@dataclass(frozen=True)
class Snz(Envelope):
    """Sudden variant Net Zero.

    https://arxiv.org/abs/2008.07411
    (Supplementary materials: FIG. S1.)

    .. todo::

        - expression
    """

    t_idling: float
    b_amplitude: float = 0.5
    """Relative B amplitude (wrt A)."""

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        # convert timings to samples
        half_pulse_duration = (_duration(times) - self.t_idling) / 2
        aspan = np.sum(times < half_pulse_duration)
        idle = len(times) - 2 * (aspan + 1)

        pulse = np.ones_like(times)
        # the aspan + 1 sample is B (and so the aspan + 1 + idle + 1), indexes are 0-based
        pulse[aspan] = pulse[aspan + 1 + idle] = self.b_amplitude
        # set idle time to 0
        pulse[aspan + 1 : aspan + 1 + idle] = 0
        return pulse


@dataclass(frozen=True)
class ECap(Envelope):
    r"""ECap pulse shape.

    .. todo::

        - add reference

    .. math::

        e_{\cap(t,\alpha)} &=& A[1 + \tanh(\alpha t/t_\theta)][1 + \tanh(\alpha (1 - t/t_\theta))]\\
        &\times& [1 + \tanh(\alpha/2)]^{-2}
    """

    alpha: float

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        x = times / len(times)
        return (
            (1 + np.tanh(self.alpha * times))
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

    custom_i: npt.NDArray
    custom_q: npt.NDArray

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        return self.custom_i

    def q(self, times: Times) -> Waveform:
        """.. todo::"""
        return self.custom_q


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
