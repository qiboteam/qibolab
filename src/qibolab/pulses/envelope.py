"""Library of pulse shapes."""

from abc import ABC
from dataclasses import dataclass
from functools import cached_property
from typing import Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.signal import lfilter
from scipy.signal.windows import gaussian

__all__ = [
    "Times",
    "Waveform",
    "IqWaveform",
    "BaseEnvelope",
    "Envelope",
    "Rectangular",
    "Exponential",
    "Gaussian",
    "GaussianSquare",
    "Drag",
    "Iir",
    "Snz",
    "ECap",
    "Custom",
]


# TODO: they could be distinguished among them, and distinguished from generic float
# arrays, using the NewType pattern -> but this require some more effort to encforce
# types throughout the whole code base
Waveform = npt.NDArray[np.float64]
"""Single component waveform, either I (in-phase) or Q (quadrature)."""
IqWaveform = npt.NDArray[np.float64]
"""Full shape, both I and Q components."""


@dataclass
class Times:
    """Time window of a pulse."""

    duration: float
    """Pulse duration."""
    samples: int
    """Number of requested samples."""
    # Here only the information consumed by the `Envelopes` is stored. How to go from
    # the sampling rate to the number of samples is callers' business, since nothing
    # else has to be known by this module.

    @property
    def mean(self) -> float:
        """Middle point of the temporal window."""
        return self.duration / 2

    @cached_property
    def window(self):
        """Individual timing of each sample."""
        return np.linspace(0, self.duration, self.samples)


class BaseEnvelope(ABC, BaseModel):
    """Pulse envelopes.

    Generates both i (in-phase) and q (quadrature) components.
    """

    def i(self, times: Times) -> Waveform:
        """In-phase envelope."""
        return np.zeros(times.samples)

    def q(self, times: Times) -> Waveform:
        """Quadrature envelope."""
        return np.zeros(times.samples)

    def envelopes(self, times: Times) -> IqWaveform:
        """Stacked i and q envelope waveforms of the pulse."""
        return np.array([self.i(times), self.q(times)])


class Rectangular(BaseEnvelope):
    """Rectangular envelope."""

    def i(self, times: Times) -> Waveform:
        """Generate a rectangular envelope."""
        return np.ones(times.samples)


class Exponential(BaseEnvelope):
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
        ts = times.window
        return (np.exp(-ts / self.upsilon) + self.g * np.exp(-ts / self.tau)) / (
            1 + self.g
        )


def _samples_sigma(rel_sigma: float, times: Times) -> float:
    """Convert standard deviation in samples.

    `rel_sigma` is assumed in units of the interval duration, and it is turned in units
    of samples, by counting the number of samples in the interval.
    """
    return rel_sigma * times.samples


class Gaussian(BaseEnvelope):
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
        return gaussian(times.samples, _samples_sigma(self.rel_sigma, times))


class GaussianSquare(BaseEnvelope):
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
        u, hw = times.mean, self.width / 2
        ts = times.window
        tails = (ts < (u - hw)) | ((u + hw) < ts)
        pulse[tails] = gaussian(len(ts[tails]), _samples_sigma(self.rel_sigma, times))

        return pulse


class Drag(BaseEnvelope):
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
        return gaussian(times.samples, _samples_sigma(self.rel_sigma, times))

    def q(self, times: Times) -> Waveform:
        """Generate ...

        .. todo::
        """
        sigma = self.rel_sigma * times.duration
        ts = times.window
        return self.beta * (-(ts - times.mean) / (sigma**2)) * self.i(times)


class Iir(BaseEnvelope):
    """IIR Filter using scipy.signal lfilter.

    https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)::

        p = [A, tau_iir]
        p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
        p = [b0, b1, a0, a1]
    """

    a: npt.NDArray
    b: npt.NDArray
    target: BaseEnvelope

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


class Snz(BaseEnvelope):
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
        half_pulse_duration = (times.duration - self.t_idling) / 2
        aspan = np.sum(times.window < half_pulse_duration)
        idle = times.samples - 2 * (aspan + 1)

        pulse = np.ones(times.samples)
        # the aspan + 1 sample is B (and so the aspan + 1 + idle + 1), indexes are 0-based
        pulse[aspan] = pulse[aspan + 1 + idle] = self.b_amplitude
        # set idle time to 0
        pulse[aspan + 1 : aspan + 1 + idle] = 0
        return pulse


class ECap(BaseEnvelope):
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
        x = times.window / times.samples
        return (
            (1 + np.tanh(self.alpha * times.window))
            * (1 + np.tanh(self.alpha * (1 - x)))
            / (1 + np.tanh(self.alpha / 2)) ** 2
        )


class Custom(BaseEnvelope):
    """Arbitrary shape.

    .. todo::

        - expand description
        - add attribute docstrings
    """

    i_: npt.NDArray
    q_: npt.NDArray

    def i(self, times: Times) -> Waveform:
        """.. todo::"""
        raise NotImplementedError

    def q(self, times: Times) -> Waveform:
        """.. todo::"""
        raise NotImplementedError


Envelope = Union[
    Rectangular,
    Exponential,
    Gaussian,
    GaussianSquare,
    Drag,
    Iir,
    Snz,
    ECap,
    Custom,
]
"""Available pulse shapes."""
