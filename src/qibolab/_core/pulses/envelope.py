"""Library of pulse envelopes."""

from abc import ABC
from typing import Annotated, Literal, Union

import numpy as np
import numpy.typing as npt
from pydantic import Field
from scipy.signal import lfilter
from scipy.signal.windows import gaussian

from ..serialize import Model, NdArray, eq

__all__ = [
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


class BaseEnvelope(ABC, Model):
    """Pulse envelopes.

    Generates both i (in-phase) and q (quadrature) components.
    """

    def i(self, samples: int) -> Waveform:
        """In-phase envelope."""
        return np.zeros(samples)

    def q(self, samples: int) -> Waveform:
        """Quadrature envelope."""
        return np.zeros(samples)

    def envelopes(self, samples: int) -> IqWaveform:
        """Stacked i and q envelope waveforms of the pulse."""
        return np.array([self.i(samples), self.q(samples)])


class Rectangular(BaseEnvelope):
    """Rectangular envelope."""

    kind: Literal["rectangular"] = "rectangular"

    def i(self, samples: int) -> Waveform:
        """Generate a rectangular envelope."""
        return np.ones(samples)


class Exponential(BaseEnvelope):
    r"""Exponential shape, i.e. square pulse with an exponential decay.

    .. math::

        \frac{\exp\left(-\frac{x}{\text{upsilon}}\right) + g \exp\left(-\frac{x}{\text{tau}}\right)}{1 + g}
    """

    kind: Literal["exponential"] = "exponential"

    tau: float
    """The decay rate of the first exponential function.

    In units of the interval duration.
    """
    upsilon: float
    """The decay rate of the second exponential function.

    In units of the interval duration.
    """
    g: float = 0.1
    """Weight of the second exponential function."""

    def i(self, samples: int) -> Waveform:
        """Generate a combination of two exponential decays."""
        x = np.arange(samples)
        upsilon = self.upsilon * samples
        tau = self.tau * samples
        return (np.exp(-x / upsilon) + self.g * np.exp(-x / tau)) / (1 + self.g)


def _samples_sigma(rel_sigma: float, samples: int) -> float:
    """Convert standard deviation in samples.

    `rel_sigma` is assumed in units of the interval duration, and it is turned in units
    of samples, by counting the number of samples in the interval.
    """
    return rel_sigma * samples


class Gaussian(BaseEnvelope):
    r"""Gaussian pulse shape.

    Args:
        rel_sigma (float):

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
    """

    kind: Literal["gaussian"] = "gaussian"

    rel_sigma: float
    """Relative Gaussian standard deviation.

    In units of the interval duration.
    """

    def i(self, samples: int) -> Waveform:
        """Generate a Gaussian window."""
        return gaussian(samples, _samples_sigma(self.rel_sigma, samples))


class GaussianSquare(BaseEnvelope):
    r"""Rectangular envelope with Gaussian rise and fall.

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Rise] + Flat + A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}[Decay]
    """

    kind: Literal["gaussian_square"] = "gaussian_square"

    rel_sigma: float
    """Relative Gaussian standard deviation.

    In units of the interval duration.
    """
    width: float
    """Length of the flat portion."""

    def i(self, samples: int) -> Waveform:
        """Generate a Gaussian envelope, with a flat central window."""

        pulse = np.ones(samples)
        u, hw = samples / 2, self.width / 2
        ts = np.arange(samples)
        tails = (ts < (u - hw)) | ((u + hw) < ts)
        pulse[tails] = gaussian(len(ts[tails]), _samples_sigma(self.rel_sigma, samples))

        return pulse


class Drag(BaseEnvelope):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse envelope.

    .. todo::

        - add expression
        - add reference
    """

    kind: Literal["drag"] = "drag"

    rel_sigma: float
    """Relative Gaussian standard deviation.

    In units of the interval duration.
    """
    beta: float
    """Beta.

    .. todo::

        Add docstring
    """

    def i(self, samples: int) -> Waveform:
        """Generate a Gaussian envelope."""
        return gaussian(samples, _samples_sigma(self.rel_sigma, samples))

    def q(self, samples: int) -> Waveform:
        """Generate ...

        .. todo::

            Add docstring
        """
        ts = np.arange(samples)
        mu = (samples - 1) / 2
        sigma = _samples_sigma(self.rel_sigma, samples)
        return self.beta * (-(ts - mu) / (sigma**2)) * self.i(samples)


class Iir(BaseEnvelope):
    """IIR Filter using scipy.signal lfilter.

    https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)::

        p = [A, tau_iir]
        p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
        p = [b0, b1, a0, a1]
    """

    kind: Literal["iir"] = "iir"

    a: NdArray
    b: NdArray
    target: BaseEnvelope

    def _data(self, target: npt.NDArray) -> npt.NDArray:
        a = self.a / self.a[0]
        gain = np.sum(self.b) / np.sum(a)
        b = self.b / gain if gain != 0 else self.b

        data = lfilter(b=b, a=a, x=target)
        if np.max(np.abs(data)) != 0:
            data /= np.max(np.abs(data))
        return data

    def i(self, samples: int) -> Waveform:
        """I.

        .. todo::

            Add docstring
        """
        return self._data(self.target.i(samples))

    def q(self, samples: int) -> Waveform:
        """Q.
        .. todo::

            Add docstring
        """
        return self._data(self.target.q(samples))

    def __eq__(self, other) -> bool:
        """Eq.

        .. todo::

            Add docstring
        """
        return eq(self, other)


class Snz(BaseEnvelope):
    """Sudden variant Net Zero.

    https://arxiv.org/abs/2008.07411
    (Supplementary materials: FIG. S1.)

    .. todo::

        - expression
    """

    kind: Literal["snz"] = "snz"

    t_idling: float
    """Fraction of interval where idling."""
    b_amplitude: float = 0.5
    """Relative B amplitude (wrt A)."""

    def i(self, samples: int) -> Waveform:
        """I.

        .. todo::

            Add docstring
        """
        # convert timings to samples
        half_pulse_duration = (1 - self.t_idling) * samples / 2
        aspan = np.sum(np.arange(samples) < half_pulse_duration)
        idle = samples - 2 * (aspan + 1)
        pulse = np.ones(samples)
        pulse[-aspan:] = -1
        # the aspan + 1 sample is B (and so the aspan + 1 + idle + 1), indexes are 0-based
        pulse[aspan] = self.b_amplitude
        pulse[aspan + 1 + idle] = -self.b_amplitude
        # set idle time to 0
        pulse[aspan + 1 : aspan + 1 + idle] = 0
        return pulse


class ECap(BaseEnvelope):
    r"""ECap pulse envelope.

    .. todo::

        - add reference

    .. math::

        e_{\cap(t,\alpha)} &=& A[1 + \tanh(\alpha t/t_\theta)][1 + \tanh(\alpha (1 - t/t_\theta))]\\
        &\times& [1 + \tanh(\alpha/2)]^{-2}
    """

    kind: Literal["ecap"] = "ecap"

    alpha: float
    """In units of the inverse interval duration."""

    def i(self, samples: int) -> Waveform:
        """I.

        .. todo::

            Add docstring
        """
        ss = np.arange(samples)
        x = ss / samples
        return (
            (1 + np.tanh(self.alpha * ss))
            * (1 + np.tanh(self.alpha * (1 - x)))
            / (1 + np.tanh(self.alpha / 2)) ** 2
        )


class Custom(BaseEnvelope):
    """Arbitrary envelope.

    .. todo::

        - expand description
        - add attribute docstrings
    """

    kind: Literal["custom"] = "custom"

    i_: NdArray
    q_: NdArray

    def i(self, samples: int) -> Waveform:
        """I.

        .. todo::

            Add docstring
        """
        if len(self.i_) != samples:
            raise ValueError

        return self.i_

    def q(self, samples: int) -> Waveform:
        """Q.

        .. todo::

            Add docstring
        """
        if len(self.q_) != samples:
            raise ValueError

        return self.q_

    def __eq__(self, other) -> bool:
        """Eq.

        .. todo::

            Add docstring
        """
        return eq(self, other)

    def __hash__(self):
        return hash(np.concatenate([self.i_, self.q_]).tobytes())


Envelope = Annotated[
    Union[
        Rectangular,
        Exponential,
        Gaussian,
        GaussianSquare,
        Drag,
        Iir,
        Snz,
        ECap,
        Custom,
    ],
    Field(discriminator="kind"),
]
"""Available pulse shapes."""
