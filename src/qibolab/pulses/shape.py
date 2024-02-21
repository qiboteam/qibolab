"""PulseShape class."""

import re
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from qibo.config import log
from scipy.signal import lfilter

SAMPLING_RATE = 1
"""Default sampling rate in gigasamples per second (GSps).

Used for generating waveform envelopes if the instruments do not provide
a different value.
"""

Waveform = npt.NDArray[np.float64]


class ShapeInitError(RuntimeError):
    """Error raised when a pulse has not been fully defined."""

    default_msg = "PulseShape attribute pulse must be initialised in order to be able to generate pulse waveforms"

    def __init__(self, msg=None, *args):
        if msg is None:
            msg = self.default_msg
        super().__init__(msg, *args)


class PulseShape(ABC):
    """Abstract class for pulse shapes.

    This object is responsible for generating envelope and modulated
    waveforms from a set of pulse parameters and its type. Generates
    both i (in-phase) and q (quadrature) components.
    """

    pulse = None
    """Pulse (Pulse): the pulse associated with it.

    Its parameters are used to generate pulse waveforms.
    """

    @abstractmethod
    def envelope_waveform_i(
        self, sampling_rate=SAMPLING_RATE
    ) -> Waveform:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def envelope_waveform_q(
        self, sampling_rate=SAMPLING_RATE
    ) -> Waveform:  # pragma: no cover
        raise NotImplementedError

    def envelope_waveforms(self, sampling_rate=SAMPLING_RATE):
        """A tuple with the i and q envelope waveforms of the pulse."""

        return (
            self.envelope_waveform_i(sampling_rate),
            self.envelope_waveform_q(sampling_rate),
        )

    def modulated_waveform_i(self, _if: int, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The waveform of the i component of the pulse, modulated with its
        frequency."""

        return self.modulated_waveforms(_if, sampling_rate)[0]

    def modulated_waveform_q(self, _if: int, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The waveform of the q component of the pulse, modulated with its
        frequency."""

        return self.modulated_waveforms(_if, sampling_rate)[1]

    def modulated_waveforms(self, _if: int, sampling_rate=SAMPLING_RATE):
        """A tuple with the i and q waveforms of the pulse, modulated with its
        frequency."""

        pulse = self.pulse
        if abs(_if) * 2 > sampling_rate:
            log.info(
                f"WARNING: The frequency of pulse {pulse.id} is higher than the nyqusit frequency ({int(sampling_rate // 2)}) for the device sampling rate: {int(sampling_rate)}"
            )
        num_samples = int(np.rint(pulse.duration * sampling_rate))
        time = np.arange(num_samples) / sampling_rate
        global_phase = pulse.global_phase
        cosalpha = np.cos(2 * np.pi * _if * time + global_phase + pulse.relative_phase)
        sinalpha = np.sin(2 * np.pi * _if * time + global_phase + pulse.relative_phase)

        mod_matrix = np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]]) / np.sqrt(
            2
        )

        (envelope_waveform_i, envelope_waveform_q) = self.envelope_waveforms(
            sampling_rate
        )
        result = []
        for n, t, ii, qq in zip(
            np.arange(num_samples),
            time,
            envelope_waveform_i,
            envelope_waveform_q,
        ):
            result.append(mod_matrix[:, :, n] @ np.array([ii, qq]))
        mod_signals = np.array(result)

        modulated_waveform_i = mod_signals[:, 0]
        modulated_waveform_q = mod_signals[:, 1]
        return (modulated_waveform_i, modulated_waveform_q)

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        return isinstance(item, type(self))

    @staticmethod
    def eval(value: str) -> "PulseShape":
        """Deserialize string representation.

        .. todo::

            To be replaced by proper serialization.
        """
        shape_name = re.findall(r"(\w+)", value)[0]
        if shape_name not in globals():
            raise ValueError(f"shape {value} not found")
        shape_parameters = re.findall(r"[\w+\d\.\d]+", value)[1:]
        # TODO: create multiple tests to prove regex working correctly
        return globals()[shape_name](*shape_parameters)


class Rectangular(PulseShape):
    """Rectangular pulse shape."""

    def __init__(self):
        self.name = "Rectangular"
        self.pulse: "Pulse" = None

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            return self.pulse.amplitude * np.ones(num_samples)
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            return np.zeros(num_samples)
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}()"


class Exponential(PulseShape):
    r"""Exponential pulse shape (Square pulse with an exponential decay).

    Args:
        tau (float): Parameter that controls the decay of the first exponential function
        upsilon (float): Parameter that controls the decay of the second exponential function
        g (float): Parameter that weights the second exponential function


    .. math::

        A\frac{\exp\left(-\frac{x}{\text{upsilon}}\right) + g \exp\left(-\frac{x}{\text{tau}}\right)}{1 + g}
    """

    def __init__(self, tau: float, upsilon: float, g: float = 0.1):
        self.name = "Exponential"
        self.pulse: "Pulse" = None
        self.tau: float = float(tau)
        self.upsilon: float = float(upsilon)
        self.g: float = float(g)

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            x = np.arange(0, num_samples, 1)
            return (
                self.pulse.amplitude
                * (
                    (np.ones(num_samples) * np.exp(-x / self.upsilon))
                    + self.g * np.exp(-x / self.tau)
                )
                / (1 + self.g)
            )

        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            return np.zeros(num_samples)
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({format(self.tau, '.3f').rstrip('0').rstrip('.')}, {format(self.upsilon, '.3f').rstrip('0').rstrip('.')}, {format(self.g, '.3f').rstrip('0').rstrip('.')})"


class Gaussian(PulseShape):
    r"""Gaussian pulse shape.

    Args:
        rel_sigma (float): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma

    .. math::

        A\exp^{-\frac{1}{2}\frac{(t-\mu)^2}{\sigma^2}}
    """

    def __init__(self, rel_sigma: float):
        self.name = "Gaussian"
        self.pulse: "Pulse" = None
        self.rel_sigma: float = float(rel_sigma)

    def __eq__(self, item) -> bool:
        """Overloads == operator."""
        if super().__eq__(item):
            return self.rel_sigma == item.rel_sigma
        return False

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            x = np.arange(0, num_samples, 1)
            return self.pulse.amplitude * np.exp(
                -(1 / 2)
                * (
                    ((x - (num_samples - 1) / 2) ** 2)
                    / (((num_samples) / self.rel_sigma) ** 2)
                )
            )
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            return np.zeros(num_samples)
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({format(self.rel_sigma, '.6f').rstrip('0').rstrip('.')})"


class GaussianSquare(PulseShape):
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

        if self.pulse:

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

        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            return np.zeros(num_samples)
        raise ShapeInitError

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

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            x = np.arange(0, num_samples, 1)
            return self.pulse.amplitude * np.exp(
                -(1 / 2)
                * (
                    ((x - (num_samples - 1) / 2) ** 2)
                    / (((num_samples) / self.rel_sigma) ** 2)
                )
            )
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
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
        raise ShapeInitError

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

        if self.pulse:
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
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
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
        raise ShapeInitError

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

        if self.pulse:
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
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            return np.zeros(num_samples)
        raise ShapeInitError

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
        if self.pulse:
            num_samples = int(self.pulse.duration * sampling_rate)
            x = np.arange(0, num_samples, 1)
            return (
                self.pulse.amplitude
                * (1 + np.tanh(self.alpha * x / num_samples))
                * (1 + np.tanh(self.alpha * (1 - x / num_samples)))
                / (1 + np.tanh(self.alpha / 2)) ** 2
            )
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration * sampling_rate)
            return np.zeros(num_samples)
        raise ShapeInitError

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

        if self.pulse:
            if self.pulse.duration != len(self.envelope_i):
                raise ValueError("Length of envelope_i must be equal to pulse duration")
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))

            return self.envelope_i * self.pulse.amplitude
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            if self.pulse.duration != len(self.envelope_q):
                raise ValueError("Length of envelope_q must be equal to pulse duration")
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))

            return self.envelope_q * self.pulse.amplitude
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({self.envelope_i[:3]}, ..., {self.envelope_q[:3]}, ...)"
