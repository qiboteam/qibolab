"""Pulse and PulseSequence classes."""
import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional

import numpy as np
from qibo.config import log
from scipy.signal import lfilter

SAMPLING_RATE = 1
"""Default sampling rate in gigasamples per second (GSps).

Used for generating waveform envelopes if the instruments do not provide
a different value.
"""


class PulseType(Enum):
    """An enumeration to distinguish different types of pulses.

    READOUT pulses triger acquisitions. DRIVE pulses are used to control
    qubit states. FLUX pulses are used to shift the frequency of flux
    tunable qubits and with it implement two-qubit gates.
    """

    READOUT = "ro"
    DRIVE = "qd"
    FLUX = "qf"
    COUPLERFLUX = "cf"


class Waveform:
    """A class to save pulse waveforms.

    A waveform is a list of samples, or discrete data points, used by the digital to analogue converters (DACs)
    to synthesise pulses.

    Attributes:
        data (np.ndarray): a numpy array containing the samples.
    """

    DECIMALS = 5

    def __init__(self, data):
        """Initialise the waveform with a of samples."""
        self.data: np.ndarray = np.array(data)

    def __len__(self):
        """Return the length of the waveform, the number of samples."""
        return len(self.data)

    def __hash__(self):
        """Hash the underlying data."""
        return hash(self.data.tobytes())

    def __eq__(self, other):
        """Compare two waveforms.

        Two waveforms are considered equal if their samples, rounded to
        `Waveform.DECIMALS` decimal places, are all equal.
        """
        return np.allclose(self.data, other.data)

    def plot(self, savefig_filename=None):
        """Plot the waveform.

        Args:
            savefig_filename (str): a file path. If provided the plot is save to a file.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5), dpi=200)
        plt.plot(self.data, c="C0", linestyle="dashed")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.grid(
            visible=True, which="both", axis="both", color="#888888", linestyle="-"
        )
        if savefig_filename:
            plt.savefig(savefig_filename)
        else:
            plt.show()
        plt.close()


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

    def envelope_waveforms(
        self, sampling_rate=SAMPLING_RATE
    ):  #  -> tuple[Waveform, Waveform]:  # pragma: no cover
        """A tuple with the i and q envelope waveforms of the pulse."""

        return (
            self.envelope_waveform_i(sampling_rate),
            self.envelope_waveform_q(sampling_rate),
        )

    def modulated_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The waveform of the i component of the pulse, modulated with its
        frequency."""

        return self.modulated_waveforms(sampling_rate)[0]

    def modulated_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The waveform of the q component of the pulse, modulated with its
        frequency."""

        return self.modulated_waveforms(sampling_rate)[1]

    def modulated_waveforms(self, sampling_rate=SAMPLING_RATE):
        """A tuple with the i and q waveforms of the pulse, modulated with its
        frequency."""

        pulse = self.pulse
        if abs(pulse._if) * 2 > sampling_rate:
            log.info(
                f"WARNING: The frequency of pulse {pulse.id} is higher than the nyqusit frequency ({int(sampling_rate // 2)}) for the device sampling rate: {int(sampling_rate)}"
            )
        num_samples = int(np.rint(pulse.duration * sampling_rate))
        time = np.arange(num_samples) / sampling_rate
        global_phase = pulse.global_phase
        cosalpha = np.cos(
            2 * np.pi * pulse._if * time + global_phase + pulse.relative_phase
        )
        sinalpha = np.sin(
            2 * np.pi * pulse._if * time + global_phase + pulse.relative_phase
        )

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
            envelope_waveform_i.data,
            envelope_waveform_q.data,
        ):
            result.append(mod_matrix[:, :, n] @ np.array([ii, qq]))
        mod_signals = np.array(result)

        modulated_waveform_i = Waveform(mod_signals[:, 0])
        modulated_waveform_q = Waveform(mod_signals[:, 1])
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
        self.pulse: Pulse = None

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            waveform = Waveform(self.pulse.amplitude * np.ones(num_samples))
            return waveform
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            waveform = Waveform(np.zeros(num_samples))
            return waveform
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
        self.pulse: Pulse = None
        self.tau: float = float(tau)
        self.upsilon: float = float(upsilon)
        self.g: float = float(g)

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            x = np.arange(0, num_samples, 1)
            waveform = Waveform(
                self.pulse.amplitude
                * (
                    (np.ones(num_samples) * np.exp(-x / self.upsilon))
                    + self.g * np.exp(-x / self.tau)
                )
                / (1 + self.g)
            )

            return waveform
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            waveform = Waveform(np.zeros(num_samples))
            return waveform
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
        self.pulse: Pulse = None
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
            waveform = Waveform(
                self.pulse.amplitude
                * np.exp(
                    -(1 / 2)
                    * (
                        ((x - (num_samples - 1) / 2) ** 2)
                        / (((num_samples) / self.rel_sigma) ** 2)
                    )
                )
            )
            return waveform
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            waveform = Waveform(np.zeros(num_samples))
            return waveform
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
        self.pulse: Pulse = None
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

            waveform = Waveform(self.pulse.amplitude * pulse)
            return waveform

        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            waveform = Waveform(np.zeros(num_samples))
            return waveform
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
        self.pulse: Pulse = None
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
            i = self.pulse.amplitude * np.exp(
                -(1 / 2)
                * (
                    ((x - (num_samples - 1) / 2) ** 2)
                    / (((num_samples) / self.rel_sigma) ** 2)
                )
            )
            waveform = Waveform(i)
            return waveform
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
            q = (
                self.beta
                * (-(x - (num_samples - 1) / 2) / ((num_samples / self.rel_sigma) ** 2))
                * i
                * sampling_rate
            )
            waveform = Waveform(q)
            return waveform
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
        self._pulse: Pulse = None
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
                x=self.target.envelope_waveform_i(sampling_rate).data,
            )
            if not np.max(np.abs(data)) == 0:
                data = data / np.max(np.abs(data))
            data = np.abs(self.pulse.amplitude) * data
            waveform = Waveform(data)
            return waveform
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
                x=self.target.envelope_waveform_q(sampling_rate).data,
            )
            if not np.max(np.abs(data)) == 0:
                data = data / np.max(np.abs(data))
            data = np.abs(self.pulse.amplitude) * data
            waveform = Waveform(data)
            return waveform
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
        self.pulse: Pulse = None
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
            waveform = Waveform(
                np.concatenate(
                    (
                        self.pulse.amplitude * np.ones(half_flux_pulse_samples - 1),
                        np.array([self.b_amplitude]),
                        np.zeros(idling_samples),
                        -np.array([self.b_amplitude]),
                        -self.pulse.amplitude * np.ones(half_flux_pulse_samples - 1),
                    )
                )
            )
            return waveform
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))
            waveform = Waveform(np.zeros(num_samples))
            return waveform
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
        self.pulse: Pulse = None
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
            waveform = Waveform(
                self.pulse.amplitude
                * (1 + np.tanh(self.alpha * x / num_samples))
                * (1 + np.tanh(self.alpha * (1 - x / num_samples)))
                / (1 + np.tanh(self.alpha / 2)) ** 2
            )
            return waveform
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration * sampling_rate)
            waveform = Waveform(np.zeros(num_samples))
            return waveform
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({format(self.alpha, '.6f').rstrip('0').rstrip('.')})"


class Custom(PulseShape):
    """Arbitrary shape."""

    def __init__(self, envelope_i, envelope_q=None):
        self.name = "Custom"
        self.pulse: Pulse = None
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

            waveform = Waveform(self.envelope_i * self.pulse.amplitude)
            return waveform
        raise ShapeInitError

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            if self.pulse.duration != len(self.envelope_q):
                raise ValueError("Length of envelope_q must be equal to pulse duration")
            num_samples = int(np.rint(self.pulse.duration * sampling_rate))

            waveform = Waveform(self.envelope_q * self.pulse.amplitude)
            return waveform
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({self.envelope_i[:3]}, ..., {self.envelope_q[:3]}, ...)"


@dataclass
class Pulse:
    """A class to represent a pulse to be sent to the QPU."""

    start: int
    """Start time of pulse in ns."""
    duration: int
    """Pulse duration in ns."""
    amplitude: float
    """Pulse digital amplitude (unitless).

    Pulse amplitudes are normalised between -1 and 1.
    """
    frequency: int
    """Pulse Intermediate Frequency in Hz.

    The value has to be in the range [10e6 to 300e6].
    """
    relative_phase: float
    """Relative phase of the pulse, in radians."""
    shape: PulseShape
    """Pulse shape, as a PulseShape object.

    See
    :py: mod:`qibolab.pulses` for list of available shapes.
    """
    channel: Optional[str] = None
    """Channel on which the pulse should be played.

    When a sequence of pulses is sent to the platform for execution,
    each pulse is sent to the instrument responsible for playing pulses
    the pulse channel. The connection of instruments with channels is
    defined in the platform runcard.
    """
    type: PulseType = PulseType.DRIVE
    """Pulse type, as an element of PulseType enumeration."""
    qubit: int = 0
    """Qubit or coupler addressed by the pulse."""
    _if: int = 0

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = PulseType(self.type)
        if isinstance(self.shape, str):
            self.shape = PulseShape.eval(self.shape)
        # TODO: drop the cyclic reference
        self.shape.pulse = self

    def __hash__(self):
        """Return hash(self).

        .. todo::

            this has to be replaced by turning :cls:`Pulse` into a _frozen_ dataclass
        """
        return hash(
            tuple(getattr(self, f.name) for f in fields(self) if f.name != "shape")
        )

    @property
    def finish(self) -> Optional[int]:
        """Time when the pulse is scheduled to finish."""
        if None in {self.start, self.duration}:
            return None
        return self.start + self.duration

    @property
    def global_phase(self):
        """Global phase of the pulse, in radians.

        This phase is calculated from the pulse start time and frequency
        as `2 * pi * frequency * start`.
        """

        # pulse start, duration and finish are in ns
        return 2 * np.pi * self.frequency * self.start / 1e9

    @property
    def phase(self) -> float:
        """Total phase of the pulse, in radians.

        The total phase is computed as the sum of the global and
        relative phases.
        """
        return self.global_phase + self.relative_phase

    @property
    def id(self) -> int:
        return id(self)

    def envelope_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        return self.shape.envelope_waveform_i(sampling_rate)

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        return self.shape.envelope_waveform_q(sampling_rate)

    def envelope_waveforms(
        self, sampling_rate=SAMPLING_RATE
    ):  #  -> tuple[Waveform, Waveform]:
        """A tuple with the i and q envelope waveforms of the pulse."""

        return (
            self.shape.envelope_waveform_i(sampling_rate),
            self.shape.envelope_waveform_q(sampling_rate),
        )

    def modulated_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The waveform of the i component of the pulse, modulated with its
        frequency."""

        return self.shape.modulated_waveform_i(sampling_rate)

    def modulated_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """The waveform of the q component of the pulse, modulated with its
        frequency."""

        return self.shape.modulated_waveform_q(sampling_rate)

    def modulated_waveforms(self, sampling_rate):  #  -> tuple[Waveform, Waveform]:
        """A tuple with the i and q waveforms of the pulse, modulated with its
        frequency."""

        return self.shape.modulated_waveforms(sampling_rate)

    def __hash__(self):
        """Hash the content.

        .. warning::

            unhashable attributes are not taken into account, so there will be more
            clashes than those usually expected with a regular hash

        .. todo::

            This method should be eventually dropped, and be provided automatically by
            freezing the dataclass (i.e. setting ``frozen=true`` in the decorator).
            However, at the moment is not possible nor desired, because it contains
            unhashable attributes and because some instances are mutated inside Qibolab.
        """
        return hash(
            tuple(
                getattr(self, f.name)
                for f in fields(self)
                if f.name not in ("type", "shape")
            )
        )

    def __add__(self, other):
        if isinstance(other, Pulse):
            return PulseSequence(self, other)
        if isinstance(other, PulseSequence):
            return PulseSequence(self, *other)
        raise TypeError(f"Expected Pulse or PulseSequence; got {type(other).__name__}")

    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f"Expected int; got {type(n).__name__}")
        if n < 0:
            raise TypeError(f"argument n should be >=0, got {n}")
        return PulseSequence(*([self.copy()] * n))

    def __rmul__(self, n):
        return self.__mul__(n)

    def copy(self):  # -> Pulse|ReadoutPulse|DrivePulse|FluxPulse:
        """Returns a new Pulse object with the same attributes."""

        if type(self) == ReadoutPulse:
            return ReadoutPulse(
                self.start,
                self.duration,
                self.amplitude,
                self.frequency,
                self.relative_phase,
                repr(self.shape),  # self.shape,
                self.channel,
                self.qubit,
            )
        elif type(self) == DrivePulse:
            return DrivePulse(
                self.start,
                self.duration,
                self.amplitude,
                self.frequency,
                self.relative_phase,
                repr(self.shape),  # self.shape,
                self.channel,
                self.qubit,
            )

        elif type(self) == FluxPulse:
            return FluxPulse(
                self.start,
                self.duration,
                self.amplitude,
                self.shape,
                self.channel,
                self.qubit,
            )
        else:
            return Pulse(
                self.start,
                self.duration,
                self.amplitude,
                self.frequency,
                self.relative_phase,
                repr(self.shape),  # self.shape,
                self.channel,
                self.type,
                self.qubit,
            )

    def shallow_copy(self):  # -> Pulse:
        return Pulse(
            self.start,
            self.duration,
            self.amplitude,
            self.frequency,
            self.relative_phase,
            self.shape,
            self.channel,
            self.type,
            self.qubit,
        )

    def is_equal_ignoring_start(self, item) -> bool:
        """Check if two pulses are equal ignoring start time."""
        return (
            self.duration == item.duration
            and self.amplitude == item.amplitude
            and self.frequency == item.frequency
            and self.relative_phase == item.relative_phase
            and self.shape == item.shape
            and self.channel == item.channel
            and self.type == item.type
            and self.qubit == item.qubit
        )

    def plot(self, savefig_filename=None, sampling_rate=SAMPLING_RATE):
        """Plots the pulse envelope and modulated waveforms.

        Args:
            savefig_filename (str): a file path. If provided the plot is save to a file.
        """

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        waveform_i = self.shape.envelope_waveform_i(sampling_rate)
        waveform_q = self.shape.envelope_waveform_q(sampling_rate)

        num_samples = len(waveform_i)
        time = self.start + np.arange(num_samples) / sampling_rate
        fig = plt.figure(figsize=(14, 5), dpi=200)
        gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(
            time,
            waveform_i.data,
            label="envelope i",
            c="C0",
            linestyle="dashed",
        )
        ax1.plot(
            time,
            waveform_q.data,
            label="envelope q",
            c="C1",
            linestyle="dashed",
        )
        ax1.plot(
            time,
            self.shape.modulated_waveform_i(sampling_rate).data,
            label="modulated i",
            c="C0",
        )
        ax1.plot(
            time,
            self.shape.modulated_waveform_q(sampling_rate).data,
            label="modulated q",
            c="C1",
        )
        ax1.plot(time, -waveform_i.data, c="silver", linestyle="dashed")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Amplitude")

        ax1.grid(
            visible=True, which="both", axis="both", color="#888888", linestyle="-"
        )
        ax1.axis([self.start, self.finish, -1, 1])
        ax1.legend()

        modulated_i = self.shape.modulated_waveform_i(sampling_rate).data
        modulated_q = self.shape.modulated_waveform_q(sampling_rate).data
        ax2 = plt.subplot(gs[1])
        ax2.plot(
            modulated_i,
            modulated_q,
            label="modulated",
            c="C3",
        )
        ax2.plot(
            waveform_i.data,
            waveform_q.data,
            label="envelope",
            c="C2",
        )
        ax2.plot(
            modulated_i[0],
            modulated_q[0],
            marker="o",
            markersize=5,
            label="start",
            c="lightcoral",
        )
        ax2.plot(
            modulated_i[-1],
            modulated_q[-1],
            marker="o",
            markersize=5,
            label="finish",
            c="darkred",
        )

        ax2.plot(
            np.cos(time * 2 * np.pi / self.duration),
            np.sin(time * 2 * np.pi / self.duration),
            c="silver",
            linestyle="dashed",
        )

        ax2.grid(
            visible=True, which="both", axis="both", color="#888888", linestyle="-"
        )
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        if savefig_filename:
            plt.savefig(savefig_filename)
        else:
            plt.show()
        plt.close()


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See
    :class: `qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(
        self,
        start,
        duration,
        amplitude,
        frequency,
        relative_phase,
        shape,
        channel=0,
        qubit=0,
    ):
        super().__init__(
            start,
            duration,
            amplitude,
            frequency,
            relative_phase,
            shape,
            channel,
            type=PulseType.READOUT,
            qubit=qubit,
        )

    @property
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.qubit})"

    @property
    def global_phase(self):
        # readout pulses should have zero global phase so that we can
        # calculate probabilities in the i-q plane
        return 0

    def copy(self):  # -> Pulse|ReadoutPulse|DrivePulse|FluxPulse:
        """Returns a new Pulse object with the same attributes."""

        return ReadoutPulse(
            self.start,
            self.duration,
            self.amplitude,
            self.frequency,
            self.relative_phase,
            copy.deepcopy(self.shape),  # self.shape,
            self.channel,
            self.qubit,
        )


class DrivePulse(Pulse):
    """Describes a qubit drive pulse.

    See
    :class: `qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(
        self,
        start,
        duration,
        amplitude,
        frequency,
        relative_phase,
        shape,
        channel=0,
        qubit=0,
    ):
        super().__init__(
            start,
            duration,
            amplitude,
            frequency,
            relative_phase,
            shape,
            channel,
            type=PulseType.DRIVE,
            qubit=qubit,
        )

    @property
    def serial(self):
        return f"DrivePulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.qubit})"


class FluxPulse(Pulse):
    """Describes a qubit flux pulse.

    Flux pulses have frequency and relative_phase equal to 0. Their i
    and q components are equal. See
    :class: `qibolab.pulses.Pulse` for argument desciption.
    """

    PULSE_TYPE = PulseType.FLUX

    def __init__(self, start, duration, amplitude, shape, channel=0, qubit=0):
        super().__init__(
            start,
            duration,
            amplitude,
            0,
            0,
            shape,
            channel,
            type=self.PULSE_TYPE,
            qubit=qubit,
        )

    def envelope_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        """Flux pulses only have i component."""
        return self.shape.envelope_waveform_i(sampling_rate)

    def modulated_waveform_i(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        return self.shape.envelope_waveform_i(sampling_rate)

    def modulated_waveform_q(self, sampling_rate=SAMPLING_RATE) -> Waveform:
        return self.shape.envelope_waveform_i(sampling_rate)

    @property
    def serial(self):
        return f"{self.__class__.__name__}({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.qubit})"


class CouplerFluxPulse(FluxPulse):
    """Describes a coupler flux pulse.

    See
    :class: `qibolab.pulses.FluxPulse` for argument desciption.
    """

    PULSE_TYPE = PulseType.COUPLERFLUX


class PulseConstructor(Enum):
    """An enumeration to map each ``PulseType`` to the proper pulse
    constructor."""

    READOUT = ReadoutPulse
    DRIVE = DrivePulse
    FLUX = FluxPulse


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
            if not isinstance(pulse, CouplerFluxPulse):
                if pulse.qubit in qubits:
                    new_pc.append(pulse)
        return new_pc

    def coupler_pulses(self, *couplers):
        """Return a new sequence containing the pulses on some couplers."""
        new_pc = PulseSequence()
        for pulse in self:
            if isinstance(pulse, CouplerFluxPulse):
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
        """Duration of the sequence calculated as its finish - start times."""
        return self.finish - self.start

    @property
    def channels(self) -> list:
        """List containing the channels used by the pulses in the sequence."""
        channels = []
        for pulse in self:
            if not pulse.channel in channels:
                channels.append(pulse.channel)
        channels.sort()
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
