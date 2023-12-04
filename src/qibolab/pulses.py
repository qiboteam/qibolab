"""Pulse and PulseSequence classes."""
import copy
import re
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from qibo.config import log
from scipy.signal import lfilter

from qibolab.symbolic import intSymbolicExpression as se_int


class PulseType(Enum):
    """An enumeration to distinguish different types of pulses.

    READOUT pulses triger acquisitions.
    DRIVE pulses are used to control qubit states.
    FLUX pulses are used to shift the frequency of flux tunable qubits and with it implement two-qubit gates.
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
        serial (str): a string that can be used as a lable to identify the waveform. It is not automatically
            generated, it must be set by the user.
    """

    DECIMALS = 5

    def __init__(self, data):
        """Initialises the waveform with a of samples."""

        self.data: np.ndarray = np.array(data)
        self.serial: str = ""

    def __len__(self):
        """Returns the length of the waveform, the number of samples."""

        return len(self.data)

    def __eq__(self, other):
        """Compares two waveforms.

        Two waveforms are considered equal if their samples, rounded to `Waveform.DECIMALS` decimal places,
        are all equal.
        """

        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """Returns a hash of the array of data, after rounding each sample to `Waveform.DECIMALS` decimal places."""

        return hash(str(np.around(self.data, Waveform.DECIMALS) + 0))

    def __repr__(self):
        """Returns the waveform serial as its string representation."""

        return self.serial

    def plot(self, savefig_filename=None):
        """Plots the waveform.

        Args:
            savefig_filename (str): a file path. If provided the plot is save to a file.
        """

        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 5), dpi=200)
        plt.plot(self.data, c="C0", linestyle="dashed")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
        plt.suptitle(self.serial)
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

    A PulseShape object is responsible for generating envelope and modulated waveforms from a set
    of pulse parameters, its type and a predefined SAMPLING_RATE. PulsShape generates both i (in-phase)
    and q (quadrature) components.
    """

    SAMPLING_RATE = 1e9  # 1GSaPS
    """SAMPLING_RATE (int): sampling rate in samples per second (SaPS)"""
    pulse = None
    """pulse (Pulse): the pulse associated with it. Its parameters are used to generate pulse waveforms."""

    @property
    @abstractmethod
    def envelope_waveform_i(self) -> Waveform:  # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def envelope_waveform_q(self) -> Waveform:  # pragma: no cover
        raise NotImplementedError

    @property
    def envelope_waveforms(self):  #  -> tuple[Waveform, Waveform]:  # pragma: no cover
        """A tuple with the i and q envelope waveforms of the pulse."""

        return (self.envelope_waveform_i, self.envelope_waveform_q)

    @property
    def modulated_waveform_i(self) -> Waveform:
        """The waveform of the i component of the pulse, modulated with its frequency."""

        return self.modulated_waveforms[0]

    @property
    def modulated_waveform_q(self) -> Waveform:
        """The waveform of the q component of the pulse, modulated with its frequency."""

        return self.modulated_waveforms[1]

    @property
    def modulated_waveforms(self):
        """A tuple with the i and q waveforms of the pulse, modulated with its frequency."""

        if not self.pulse:
            raise ShapeInitError

        pulse = self.pulse
        if abs(pulse._if) * 2 > PulseShape.SAMPLING_RATE:
            log.info(
                f"WARNING: The frequency of pulse {pulse.serial} is higher than the nyqusit frequency ({int(PulseShape.SAMPLING_RATE // 2)}) for the device sampling rate: {int(PulseShape.SAMPLING_RATE)}"
            )
        num_samples = int(np.rint(pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
        time = np.arange(num_samples) / PulseShape.SAMPLING_RATE
        global_phase = pulse.global_phase
        cosalpha = np.cos(2 * np.pi * pulse._if * time + global_phase + pulse.relative_phase)
        sinalpha = np.sin(2 * np.pi * pulse._if * time + global_phase + pulse.relative_phase)

        mod_matrix = np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]]) / np.sqrt(2)

        (envelope_waveform_i, envelope_waveform_q) = self.envelope_waveforms
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
        modulated_waveform_i.serial = f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
        modulated_waveform_q = Waveform(mod_signals[:, 1])
        modulated_waveform_q.serial = f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse._if, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
        return (modulated_waveform_i, modulated_waveform_q)

    def __eq__(self, item) -> bool:
        """Overloads == operator"""
        return isinstance(item, type(self))


class Rectangular(PulseShape):
    """Rectangular pulse shape."""

    def __init__(self):
        self.name = "Rectangular"
        self.pulse: Pulse = None

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            waveform = Waveform(self.pulse.amplitude * np.ones(num_samples))
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
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

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            x = np.arange(0, num_samples, 1)
            waveform = Waveform(
                self.pulse.amplitude
                * ((np.ones(num_samples) * np.exp(-x / self.upsilon)) + self.g * np.exp(-x / self.tau))
                / (1 + self.g)
            )

            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
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
        """Overloads == operator"""
        if super().__eq__(item):
            return self.rel_sigma == item.rel_sigma
        return False

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            x = np.arange(0, num_samples, 1)
            waveform = Waveform(
                self.pulse.amplitude
                * np.exp(-(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / self.rel_sigma) ** 2)))
            )
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({format(self.rel_sigma, '.6f').rstrip('0').rstrip('.')})"


class Drag(PulseShape):
    """
    Derivative Removal by Adiabatic Gate (DRAG) pulse shape.

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
        """Overloads == operator"""
        if super().__eq__(item):
            return self.rel_sigma == item.rel_sigma and self.beta == item.beta
        return False

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            x = np.arange(0, num_samples, 1)
            i = self.pulse.amplitude * np.exp(
                -(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / self.rel_sigma) ** 2))
            )
            waveform = Waveform(i)
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            x = np.arange(0, num_samples, 1)
            i = self.pulse.amplitude * np.exp(
                -(1 / 2) * (((x - (num_samples - 1) / 2) ** 2) / (((num_samples) / self.rel_sigma) ** 2))
            )
            q = (
                self.beta
                * (-(x - (num_samples - 1) / 2) / ((num_samples / self.rel_sigma) ** 2))
                * i
                * PulseShape.SAMPLING_RATE
                / 1e9
            )
            waveform = Waveform(q)
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
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
        """Overloads == operator"""
        if super().__eq__(item):
            return self.target == item.target and (self.a == item.a).all() and (self.b == item.b).all()
        return False

    @property
    def pulse(self):
        return self._pulse

    @pulse.setter
    def pulse(self, value):
        self._pulse = value
        self.target.pulse = value

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            self.a = self.a / self.a[0]
            gain = np.sum(self.b) / np.sum(self.a)
            if not gain == 0:
                self.b = self.b / gain
            data = lfilter(b=self.b, a=self.a, x=self.target.envelope_waveform_i.data)
            if not np.max(np.abs(data)) == 0:
                data = data / np.max(np.abs(data))
            data = np.abs(self.pulse.amplitude) * data
            waveform = Waveform(data)
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            self.a = self.a / self.a[0]
            gain = np.sum(self.b) / np.sum(self.a)
            if not gain == 0:
                self.b = self.b / gain
            data = lfilter(b=self.b, a=self.a, x=self.target.envelope_waveform_q.data)
            if not np.max(np.abs(data)) == 0:
                data = data / np.max(np.abs(data))
            data = np.abs(self.pulse.amplitude) * data
            waveform = Waveform(data)
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    def __repr__(self):
        formatted_b = [round(b, 3) for b in self.b]
        formatted_a = [round(a, 3) for a in self.a]
        return f"{self.name}({formatted_b}, {formatted_a}, {self.target})"


class SNZ(PulseShape):
    """
    Sudden variant Net Zero.
    https://arxiv.org/abs/2008.07411

    """

    def __init__(self, t_idling):
        self.name = "SNZ"
        self.pulse: Pulse = None
        self.t_idling: float = t_idling

    def __eq__(self, item) -> bool:
        """Overloads == operator"""
        if super().__eq__(item):
            return self.t_idling == item.t_idling
        return False

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            if self.t_idling > self.pulse.duration:
                raise ValueError(f"Cannot put idling time {self.t_idling} higher than duration {self.pulse.duration}.")
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            half_pulse_duration = (self.pulse.duration - self.t_idling) / 2
            half_flux_pulse_samples = int(np.rint(num_samples * half_pulse_duration / self.pulse.duration))
            idling_samples = num_samples - 2 * half_flux_pulse_samples
            waveform = Waveform(
                np.concatenate(
                    (
                        self.pulse.amplitude * np.ones(half_flux_pulse_samples),
                        np.zeros(idling_samples),
                        -1 * self.pulse.amplitude * np.ones(half_flux_pulse_samples),
                    )
                )
            )
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({self.t_idling})"


class eCap(PulseShape):
    r"""eCap pulse shape.

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
        """Overloads == operator"""
        if super().__eq__(item):
            return self.alpha == item.alpha
        return False

    @property
    def envelope_waveform_i(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0, num_samples, 1)
            waveform = Waveform(
                self.pulse.amplitude
                * (1 + np.tanh(self.alpha * x / num_samples))
                * (1 + np.tanh(self.alpha * (1 - x / num_samples)))
                / (1 + np.tanh(self.alpha / 2)) ** 2
            )
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
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

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        if self.pulse:
            if self.pulse.duration != len(self.envelope_i):
                raise ValueError("Length of envelope_i must be equal to pulse duration")
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))

            waveform = Waveform(self.envelope_i * self.pulse.amplitude)
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        if self.pulse:
            if self.pulse.duration != len(self.envelope_q):
                raise ValueError("Length of envelope_q must be equal to pulse duration")
            num_samples = int(np.rint(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE))

            waveform = Waveform(self.envelope_q * self.pulse.amplitude)
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        raise ShapeInitError

    def __repr__(self):
        return f"{self.name}({self.envelope_i[:3]}, ..., {self.envelope_q[:3]}, ...)"


class Pulse:
    """A class to represent a pulse to be sent to the QPU.

    Args:
        start (int | intSymbolicExpression): Start time of pulse in ns.
        duration (int | intSymbolicExpression): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [-1 to 1].
        frequency (int): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        relative_phase (float): To be added.
        shape: (PulseShape | str): {'Rectangular()', 'Gaussian(rel_sigma)', 'DRAG(rel_sigma, beta)'} Pulse shape.
            See :py:mod:`qibolab.pulses` for list of available shapes.
        channel (int | str): the channel on which the pulse should be synthesised.
        type (PulseType | str): {'ro', 'qd', 'qf'} type of pulse {ReadOut, Qubit Drive, Qubit Flux}
        qubit (int): qubit or coupler associated with the pulse

    Example:
        .. code-block:: python

            from qibolab.pulses import Pulse, Gaussian

            # define Gaussian drive pulse
            drive_pulse = Pulse(
                start=0,
                duration=60,
                amplitude=0.3,
                frequency=-200_000_000,
                relative_phase=0.0,
                shape=Gaussian(5),
                channel=1,
                type=PulseType.DRIVE,
                qubit=0,
            )

            # define Rectangular readout pulse
            readout_pulse = Pulse(
                start=intSymbolicExpression(60),
                duration=2000,
                amplitude=0.3,
                frequency=20_000_000,
                relative_phase=0.0,
                shape=Rectangular(),
                channel=2,
                type=PulseType.READOUT,
                qubit=0,
            )
    """

    count: int = 0

    def __init__(
        self,
        start,
        duration,
        amplitude,
        frequency,
        relative_phase,
        shape,
        channel=0,
        type=PulseType.DRIVE,
        qubit=0,
    ):
        # def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
        #                    channel: int | str, type: PulseType | str  = PulseType.DRIVE, qubit: int | str = 0):

        self._start: se_int = None
        self._duration: se_int = None
        self._finish: se_int = None
        self._amplitude: float = None
        self._frequency: int = None
        self._relative_phase: float = None
        self._shape: PulseShape = None
        self._channel = None
        # self._channel: int | str = None
        self._type: PulseType = None
        self._qubit = None
        # self._qubit: int | str = None
        self._id: int = Pulse.count
        Pulse.count += 1

        self.start = start
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.relative_phase = relative_phase
        self.shape = shape
        self.channel = channel
        self.type = type
        self.qubit = qubit

        self._if = 0

    def __del__(self):
        del self._start
        del self._duration
        del self._finish
        # del self._shape TODO activate when returning a deep copy of shape or when making a deep copy of shape in init

    @property
    def start(self) -> int:
        """Returns the time when the pulse is scheduled to be played, in ns."""
        if isinstance(self._start, se_int):
            return self._start.value
        return self._start

    @start.setter
    def start(self, value):
        """Sets the time when the pulse is scheduled to be played.

        Args:
            value (se_int | int | np.integer): the time in ns.
        """

        if not isinstance(value, (se_int, int, np.integer, float)):
            raise TypeError(f"start argument type should be intSymbolicExpression or int, got {type(value).__name__}")
        if not value >= 0:
            raise ValueError(f"start argument must be >= 0, got {value}")

        if isinstance(value, se_int):
            self._start = se_int(value.symbol)["_p" + str(self._id) + "_start"]

        elif isinstance(self._start, se_int):
            if isinstance(value, np.integer):
                self._start.value = int(value)
            elif isinstance(value, int):
                self._start.value = value
        else:
            if isinstance(value, np.integer):
                self._start = int(value)
            else:
                self._start = value

        if not self._duration is None:
            if (
                isinstance(self._start, se_int)
                or isinstance(self._duration, se_int)
                or isinstance(self._finish, se_int)
            ):
                self._finish = se_int(self._start + self._duration)["_p" + str(self._id) + "_finish"]
            else:
                self._finish = self._start + self._duration

    @property
    def duration(self) -> int:
        """Returns the duration of the pulse, in ns."""
        if isinstance(self._duration, se_int):
            return self._duration.value
        return self._duration

    @duration.setter
    def duration(self, value):
        """Sets the duration of the pulse.

        Args:
            value (se_int | int | np.integer): the time in ns.
        """

        if not isinstance(value, (se_int, int, np.integer, float)):
            raise TypeError(
                f"duration argument type should be float, intSymbolicExpression or int, got {type(value).__name__}"
            )
        if not value >= 0:
            raise ValueError(f"duration argument must be >= 0, got {value}")
        if isinstance(value, se_int):
            self._duration = se_int(value.symbol)["_p" + str(self._id) + "_duration"]

        elif isinstance(self._duration, se_int):
            if isinstance(value, np.integer):
                self._duration.value = int(value)
            elif isinstance(value, int):
                self._duration.value = value
        else:
            if isinstance(value, np.integer):
                self._duration = int(value)
            else:
                self._duration = value

        if not self._start is None:
            if (
                isinstance(self._start, se_int)
                or isinstance(self._duration, se_int)
                or isinstance(self._finish, se_int)
            ):
                self._finish = se_int(self._start + self._duration)["_p" + str(self._id) + "_finish"]
            else:
                self._finish = self._start + self._duration

    @property
    def finish(self) -> int:
        """Returns the time when the pulse is scheduled to finish.

        Calculated as pulse.start - pulse finish.
        """
        if isinstance(self._finish, se_int):
            return self._finish.value
        return self._finish

    @property
    def se_start(self) -> se_int:
        """Returns a symbolic expression for the pulse start."""

        if not isinstance(self._start, se_int):
            self._start = se_int(self._start)["_p" + str(self._id) + "_start"]
        return self._start

    @property
    def se_duration(self) -> se_int:
        """Returns a symbolic expression for the pulse duration."""

        if not isinstance(self._duration, se_int):
            self._duration = se_int(self._duration)["_p" + str(self._id) + "_duration"]
        return self._duration

    @property
    def se_finish(self) -> se_int:
        """Returns a symbolic expression for the pulse finish."""

        if not isinstance(self._finish, se_int):
            self._finish = se_int(self._finish)["_p" + str(self._id) + "_finish"]
        return self._finish

    @property
    def amplitude(self) -> float:
        """Returns the amplitude of the pulse.

        Pulse amplitudes are normalised between -1 and 1.
        """

        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        """Sets the amplitude of the pulse.

        Args:
            value (int | float | np.floating): a unitless value between -1 and 1.
        """

        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, (float, np.floating)):
            raise TypeError(f"amplitude argument type should be float, got {type(value).__name__}")
        if not ((value >= -1) & (value <= 1)):
            raise ValueError(f"amplitude argument must be >= -1 & <= 1, got {value}")
        if isinstance(value, np.floating):
            self._amplitude = float(value)
        elif isinstance(value, float):
            self._amplitude = value

    @property
    def frequency(self) -> int:
        """Returns the frequency of the pulse, in Hz."""

        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """Sets the frequency of the pulse.

        Args:
            value (int | float | np.integer | np.floating): the frequency in Hz.
        """

        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"frequency argument type should be int, got {type(value).__name__}")
        if isinstance(value, (float, np.integer, np.floating)):
            self._frequency = int(value)
        elif isinstance(value, int):
            self._frequency = value

    @property
    def global_phase(self):
        """Returns the global phase of the pulse, in radians.

        This phase is calculated from the pulse start time and frequency as `2 * pi * frequency * start`.
        """

        # pulse start, duration and finish are in ns
        return 2 * np.pi * self._frequency * self.start / 1e9

    @property
    def relative_phase(self) -> float:
        """Returns the relative phase of the pulse, in radians."""

        return self._relative_phase

    @relative_phase.setter
    def relative_phase(self, value):
        """Sets a relative phase for the pulse.

        Args:
            value (int | float | np.integer | np.floating): the relative phase in radians.
        """

        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"relative_phase argument type should be int or float, got {type(value).__name__}")
        if isinstance(value, (int, np.integer, np.floating)):
            self._relative_phase = float(value)
        elif isinstance(value, float):
            self._relative_phase = value

    @property
    def phase(self) -> float:
        """Returns the total phase of the pulse, in radians.

        The total phase is computed as the sum of the global and relative phases.
        """
        return 2 * np.pi * self._frequency * self.start / 1e9 + self._relative_phase

    @property
    def shape(self) -> PulseShape:
        """Returns the shape of the pulse, as a PulseShape object."""

        return self._shape

    @shape.setter
    def shape(self, value):
        """Sets the shape of the pulse.

        Args:
            value (PulseShape | str): a string representing the pulse shape and its main parameters, or a PulseShape object.
        """

        if not isinstance(value, (PulseShape, str)):
            raise TypeError(f"shape argument type should be PulseShape or str, got {type(value).__name__}")
        if isinstance(value, PulseShape):
            self._shape = value
        elif isinstance(value, str):
            shape_name = re.findall(r"(\w+)", value)[0]
            if shape_name not in globals():
                raise ValueError(f"shape {value} not found")
            shape_parameters = re.findall(r"[\w+\d\.\d]+", value)[1:]
            # TODO: create multiple tests to prove regex working correctly
            self._shape = globals()[shape_name](*shape_parameters)

        # link the pulse attribute of the PulseShape object to the pulse.
        self._shape.pulse = self

    @property
    def channel(self):
        """Returns the channel on which the pulse should be played.

        When a sequence of pulses is sent to the platform for execution, each pulse is sent to the instrument
        responsible for playing pulses the pulse channel. The connection of instruments with channels is defined
        in the platform runcard.
        """

        # def channel(self) -> int | str:
        return self._channel

    @channel.setter
    def channel(self, value):
        """Sets the channel on which the pulse should be played.

        Args:
            value (int | str): an integer or a string used to identify the channel.
        """

        if not isinstance(value, (int, str)):
            raise TypeError(f"channel argument type should be int or str, got {type(value).__name__}")
        self._channel = value

    @property
    def type(self) -> PulseType:
        """Returns the pulse type, as an element of PulseType enumeration."""

        return self._type

    @type.setter
    def type(self, value):
        """Sets the type of the pulse.

        Args:
            value (PulseType | str): the type of pulse as an element of PulseType enumeration or as a two-letter string.
        """

        if isinstance(value, PulseType):
            self._type = value
        elif isinstance(value, str):
            self._type = PulseType(value)
        else:
            raise TypeError(f"type argument should be PulseType or str, got {type(value).__name__}")

    @property
    def qubit(self):
        """Returns the qubit addressed by the pulse."""

        # def qubit(self) -> int | str:
        return self._qubit

    @qubit.setter
    def qubit(self, value):
        """Sets the qubit addressed by the pulse.

        Args:
            value (int | str): an integer or a string used to identify the qubit.
        """

        if not isinstance(value, (int, str)):
            raise TypeError(f"qubit argument type should be int or str, got {type(value).__name__}")
        self._qubit = value

    @property
    def serial(self) -> str:
        """Returns a string representation of the pulse."""

        return f"Pulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.type}, {self.qubit})"

    @property
    def id(self) -> int:
        return id(self)

    @property
    def envelope_waveform_i(self) -> Waveform:
        """The envelope waveform of the i component of the pulse."""

        return self._shape.envelope_waveform_i

    @property
    def envelope_waveform_q(self) -> Waveform:
        """The envelope waveform of the q component of the pulse."""

        return self._shape.envelope_waveform_q

    @property
    def envelope_waveforms(self):  #  -> tuple[Waveform, Waveform]:
        """A tuple with the i and q envelope waveforms of the pulse."""

        return (self._shape.envelope_waveform_i, self._shape.envelope_waveform_q)

    @property
    def modulated_waveform_i(self) -> Waveform:
        """The waveform of the i component of the pulse, modulated with its frequency."""

        return self._shape.modulated_waveform_i

    @property
    def modulated_waveform_q(self) -> Waveform:
        """The waveform of the q component of the pulse, modulated with its frequency."""

        return self._shape.modulated_waveform_q

    @property
    def modulated_waveforms(self):  #  -> tuple[Waveform, Waveform]:
        """A tuple with the i and q waveforms of the pulse, modulated with its frequency."""

        return self._shape.modulated_waveforms

    def __repr__(self):
        return self.serial

    def __hash__(self):
        return hash(self.serial)

    def __eq__(self, other):
        if isinstance(other, Pulse):
            return self.serial == other.serial
        return False

    def __add__(self, other):
        if isinstance(other, Pulse):
            return PulseSequence(self, other)
        if isinstance(other, PulseSequence):
            return PulseSequence(self, *other.pulses)
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
                repr(self._shape),  # self._shape,
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
                repr(self._shape),  # self._shape,
                self.channel,
                self.qubit,
            )

        elif type(self) == FluxPulse:
            return FluxPulse(
                self.start,
                self.duration,
                self.amplitude,
                self._shape,
                self.channel,
                self.qubit,
            )
        else:
            # return eval(self.serial)
            return Pulse(
                self.start,
                self.duration,
                self.amplitude,
                self.frequency,
                self.relative_phase,
                repr(self._shape),  # self._shape,
                self.channel,
                self.type,
                self.qubit,
            )

    def shallow_copy(self):  # -> Pulse:
        return Pulse(
            self._start,
            self._duration,
            self._amplitude,
            self._frequency,
            self._relative_phase,
            self._shape,
            self._channel,
            self._type,
            self._qubit,
        )

    def is_equal_ignoring_start(self, item) -> bool:
        """Check if two pulses are equal ignoring start time"""
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

    def plot(self, savefig_filename=None):
        """Plots the pulse envelope and modulated waveforms.

        Args:
            savefig_filename (str): a file path. If provided the plot is save to a file.
        """

        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        num_samples = len(self.shape.envelope_waveform_i)
        time = self.start + np.arange(num_samples) / PulseShape.SAMPLING_RATE * 1e9
        fig = plt.figure(figsize=(14, 5), dpi=200)
        gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(
            time,
            self.shape.envelope_waveform_i.data,
            label="envelope i",
            c="C0",
            linestyle="dashed",
        )
        ax1.plot(
            time,
            self.shape.envelope_waveform_q.data,
            label="envelope q",
            c="C1",
            linestyle="dashed",
        )
        ax1.plot(time, self.shape.modulated_waveform_i.data, label="modulated i", c="C0")
        ax1.plot(time, self.shape.modulated_waveform_q.data, label="modulated q", c="C1")
        ax1.plot(time, -self.shape.envelope_waveform_i.data, c="silver", linestyle="dashed")
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Amplitude")

        ax1.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
        ax1.axis([self.start, self.finish, -1, 1])
        ax1.legend()

        ax2 = plt.subplot(gs[1])
        ax2.plot(
            self.shape.modulated_waveform_i.data,
            self.shape.modulated_waveform_q.data,
            label="modulated",
            c="C3",
        )
        ax2.plot(
            self.shape.envelope_waveform_i.data,
            self.shape.envelope_waveform_q.data,
            label="envelope",
            c="C2",
        )
        ax2.plot(
            self.shape.modulated_waveform_i.data[0],
            self.shape.modulated_waveform_q.data[0],
            marker="o",
            markersize=5,
            label="start",
            c="lightcoral",
        )
        ax2.plot(
            self.shape.modulated_waveform_i.data[-1],
            self.shape.modulated_waveform_q.data[-1],
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

        ax2.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        plt.suptitle(self.serial)
        if savefig_filename:
            plt.savefig(savefig_filename)
        else:
            plt.show()
        plt.close()


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
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
        # def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
        #                    channel: int | str, qubit: int | str = 0):
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
            copy.deepcopy(self._shape),  # self._shape,
            self.channel,
            self.qubit,
        )


class DrivePulse(Pulse):
    """Describes a qubit drive pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
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
        # def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
        #                    channel: int | str, qubit: int | str = 0):
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

    Flux pulses have frequency and relative_phase equal to 0. Their i and q components are equal.
    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """

    PULSE_TYPE = PulseType.FLUX

    def __init__(self, start, duration, amplitude, shape, channel=0, qubit=0):
        # def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
        #                    channel: int | str, qubit: int | str = 0):
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

    @property
    def envelope_waveform_i(self) -> Waveform:
        return self._shape.envelope_waveform_i

    @property
    def envelope_waveform_q(self) -> Waveform:
        return self._shape.envelope_waveform_i

    @property
    def envelope_waveforms(self):  #  -> tuple[Waveform, Waveform]:
        return (self._shape.envelope_waveform_i, self._shape.envelope_waveform_i)

    @property
    def modulated_waveform_i(self) -> Waveform:
        return self._shape.envelope_waveform_i

    @property
    def modulated_waveform_q(self) -> Waveform:
        return self._shape.envelope_waveform_i

    @property
    def modulated_waveforms(self):  #  -> tuple[Waveform, Waveform]:
        return (self._shape.envelope_waveform_i, self._shape.envelope_waveform_i)

    @property
    def serial(self):
        return f"{self.__class__.__name__}({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.qubit})"


class CouplerFluxPulse(FluxPulse):
    """Describes a coupler flux pulse.
    See :class:`qibolab.pulses.FluxPulse` for argument desciption.
    """

    PULSE_TYPE = PulseType.COUPLERFLUX


class SplitPulse(Pulse):
    """A supporting class to represent sections or slices of a pulse."""

    # TODO: Since this class is only required by qblox drivers, move to qblox.py
    def __init__(self, pulse: Pulse, window_start: int = None, window_finish: int = None):
        super().__init__(
            pulse.start,
            pulse.duration,
            pulse.amplitude,
            pulse.frequency,
            pulse.relative_phase,
            eval(str(pulse.shape)),
            pulse.channel,
            type=pulse.type,
            qubit=pulse.qubit,
        )
        self._window_start: int = pulse.start
        self._window_finish: int = pulse.finish
        if not window_start:
            window_start = pulse.start
        if not window_finish:
            window_finish = pulse.finish
        self.window_start = window_start
        self.window_finish = window_finish

    @property
    def window_start(self):
        return self._window_start

    @window_start.setter
    def window_start(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"window_start argument type should be int, got {type(value).__name__}")
        if value < self.start:
            raise ValueError("window_start should be >= pulse start ({self._start}), got {value}")
        self._window_start = value

    @property
    def window_finish(self):
        return self._window_finish

    @window_finish.setter
    def window_finish(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"window_start argument type should be int, got {type(value).__name__}")
        if value > self.finish:
            raise ValueError("window_finish should be <= pulse finish ({self._finish}), got {value}")
        self._window_finish = value

    @property
    def window_duration(self):
        return self._window_finish - self._window_start

    @property
    def serial(self):
        return f"SplitPulse({self.window_start}, {self.window_duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.qubit})"

    @property
    def envelope_waveform_i(self) -> Waveform:
        waveform = Waveform(
            self._shape.envelope_waveform_i.data[self._window_start - self.start : self._window_finish - self.start]
        )
        waveform.serial = (
            self._shape.envelope_waveform_i.serial
            + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        )
        return waveform

    @property
    def envelope_waveform_q(self) -> Waveform:
        waveform = Waveform(
            self._shape.modulated_waveform_i.data[self._window_start - self.start : self._window_finish - self.start]
        )
        waveform.serial = (
            self._shape.modulated_waveform_i.serial
            + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        )
        return waveform

    @property
    def envelope_waveforms(self):  #  -> tuple[Waveform, Waveform]:
        return (self.envelope_waveform_i, self.envelope_waveform_q)

    @property
    def modulated_waveform_i(self) -> Waveform:
        waveform = Waveform(
            self._shape.envelope_waveform_q.data[self._window_start - self.start : self._window_finish - self.start]
        )
        waveform.serial = (
            self._shape.envelope_waveform_q.serial
            + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        )
        return waveform

    @property
    def modulated_waveform_q(self) -> Waveform:
        waveform = Waveform(
            self._shape.modulated_waveform_q.data[self._window_start - self.start : self._window_finish - self.start]
        )
        waveform.serial = (
            self._shape.modulated_waveform_q.serial
            + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        )
        return waveform

    @property
    def modulated_waveforms(self):  #  -> tuple[Waveform, Waveform]:
        return (self.modulated_waveform_i, self.modulated_waveform_q)

    def plot(self, savefig_filename=None):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        idx = slice(self._window_start - self.start, self._window_finish - self.start)
        num_samples = len(self.shape.envelope_waveform_i.data[idx])
        time = self.window_start + np.arange(num_samples) / PulseShape.SAMPLING_RATE * 1e9

        fig = plt.figure(figsize=(14, 5), dpi=200)
        gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(
            time,
            self.shape.envelope_waveform_i.data[idx],
            label="envelope i",
            c="C0",
            linestyle="dashed",
        )
        ax1.plot(
            time,
            self.shape.envelope_waveform_q.data[idx],
            label="envelope q",
            c="C1",
            linestyle="dashed",
        )
        ax1.plot(
            time,
            self.shape.modulated_waveform_i.data[idx],
            label="modulated i",
            c="C0",
        )
        ax1.plot(
            time,
            self.shape.modulated_waveform_q.data[idx],
            label="modulated q",
            c="C1",
        )
        ax1.plot(
            time,
            -self.shape.envelope_waveform_i.data[idx],
            c="silver",
            linestyle="dashed",
        )
        ax1.set_xlabel("Time [ns]")
        ax1.set_ylabel("Amplitude")

        ax1.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
        ax1.axis([self.window_start, self._window_finish, -1, 1])
        ax1.legend()

        ax2 = plt.subplot(gs[1])
        ax2.plot(
            self.shape.modulated_waveform_i.data[idx],
            self.shape.modulated_waveform_q.data[idx],
            label="modulated",
            c="C3",
        )
        ax2.plot(
            self.shape.envelope_waveform_i.data[idx],
            self.shape.envelope_waveform_q.data[idx],
            label="envelope",
            c="C2",
        )
        ax2.plot(
            np.cos(time * 2 * np.pi / self.window_duration),
            np.sin(time * 2 * np.pi / self.window_duration),
            c="silver",
            linestyle="dashed",
        )

        ax2.grid(visible=True, which="both", axis="both", color="#888888", linestyle="-")
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        if savefig_filename:
            plt.savefig(savefig_filename)
        else:
            plt.show()
        plt.close()


class PulseConstructor(Enum):
    """An enumeration to map each ``PulseType`` to the proper pulse constructor."""

    READOUT = ReadoutPulse
    DRIVE = DrivePulse
    FLUX = FluxPulse


class PulseSequence:
    """A collection of scheduled pulses.

    A quantum circuit can be translated into a set of scheduled pulses that implement the circuit gates.
    This class contains many supporting fuctions to facilitate the creation and manipulation of
    these collections of pulses.
    None of the methods of PulseSequence modify any of the properties of its pulses.
    """

    def __init__(self, *pulses):
        self.pulses = []  #: list[Pulse] = []
        """pulses (list): a list containing the pulses, ordered by their channel and start times."""
        self.add(*pulses)

    def __len__(self):
        return len(self.pulses)

    def __iter__(self):
        return iter(self.pulses)

    def __getitem__(self, index):
        return self.pulses[index]

    def __setitem__(self, index, value):
        self.pulses[index] = value

    def __delitem__(self, index):
        del self.pulses[index]

    def __contains__(self, pulse):
        return pulse in self.pulses

    def __repr__(self):
        return self.serial

    @property
    def serial(self):
        """Returns a string representation of the pulse sequence."""

        return "PulseSequence\n" + "\n".join(f"{pulse.serial}" for pulse in self.pulses)

    def __eq__(self, other):
        if not isinstance(other, PulseSequence):
            raise TypeError(f"Expected PulseSequence; got {type(other).__name__}")
        return self.serial == other.serial

    def __ne__(self, other):
        if not isinstance(other, PulseSequence):
            raise TypeError(f"Expected PulseSequence; got {type(other).__name__}")
        return self.serial != other.serial

    def __hash__(self):
        return hash(self.serial)

    def __add__(self, other):
        if isinstance(other, PulseSequence):
            return PulseSequence(*self.pulses, *other.pulses)
        if isinstance(other, Pulse):
            return PulseSequence(*self.pulses, other)
        raise TypeError(f"Expected PulseSequence or Pulse; got {type(other).__name__}")

    def __radd__(self, other):
        if isinstance(other, PulseSequence):
            return PulseSequence(*other.pulses, *self.pulses)
        if isinstance(other, Pulse):
            return PulseSequence(other, *self.pulses)
        raise TypeError(f"Expected PulseSequence or Pulse; got {type(other).__name__}")

    def __iadd__(self, other):
        if isinstance(other, PulseSequence):
            self.add(*other.pulses)
        elif isinstance(other, Pulse):
            self.add(other)
        else:
            raise TypeError(f"Expected PulseSequence or Pulse; got {type(other).__name__}")
        return self

    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f"Expected int; got {type(n).__name__}")
        if n < 0:
            raise TypeError(f"argument n should be >=0, got {n}")
        return PulseSequence(*(self.pulses * n))

    def __rmul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f"Expected int; got {type(n).__name__}")
        if n < 0:
            raise TypeError(f"argument n should be >=0, got {n}")
        return PulseSequence(*(self.pulses * n))

    def __imul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f"Expected int; got {type(n).__name__}")
        if n < 1:
            raise TypeError(f"argument n should be >=1, got {n}")
        original_set = self.shallow_copy()
        for x in range(n - 1):
            self.add(*original_set.pulses)
        return self

    @property
    def count(self):
        """Returns the number of pulses in the sequence."""

        return len(self.pulses)

    def add(self, *items):
        """Adds pulses to the sequence and sorts them by channel and start time."""

        for item in items:
            if isinstance(item, Pulse):
                pulse = item
                self.pulses.append(pulse)
            elif isinstance(item, PulseSequence):
                ps = item
                for pulse in ps.pulses:
                    self.pulses.append(pulse)
        self.pulses.sort(key=lambda item: (item.start, item.channel))

    def index(self, pulse):
        """Returns the index of a pulse in the sequence."""

        return self.pulses.index(pulse)

    def pop(self, index=-1):
        """Returns the pulse with the index provided and removes it from the sequence."""

        return self.pulses.pop(index)

    def remove(self, pulse):
        """Removes a pulse from the sequence."""

        while pulse in self.pulses:
            self.pulses.remove(pulse)

    def clear(self):
        """Removes all pulses from the sequence."""

        self.pulses.clear()

    def shallow_copy(self):
        """Returns a shallow copy of the sequence.

        It returns a new PulseSequence object with references to the same Pulse objects.
        """

        return PulseSequence(*self.pulses)

    def copy(self):
        """Returns a deep copy of the sequence.

        It returns a new PulseSequence with replicates of each of the pulses contained in the original sequence.
        """

        return PulseSequence(*[pulse.copy() for pulse in self.pulses])

    @property
    def ro_pulses(self):
        """Returns a new PulseSequence containing only its readout pulses."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type == PulseType.READOUT:
                new_pc.add(pulse)
        return new_pc

    @property
    def qd_pulses(self):
        """Returns a new PulseSequence containing only its qubit drive pulses."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type == PulseType.DRIVE:
                new_pc.add(pulse)
        return new_pc

    @property
    def qf_pulses(self):
        """Returns a new PulseSequence containing only its qubit flux pulses."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type == PulseType.FLUX:
                new_pc.add(pulse)
        return new_pc

    @property
    def cf_pulses(self):
        """Returns a new PulseSequence containing only its coupler flux pulses."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type is PulseType.COUPLERFLUX:
                new_pc.add(pulse)
        return new_pc

    def get_channel_pulses(self, *channels):
        """Returns a new PulseSequence containing only the pulses on a specific set of channels."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.channel in channels:
                new_pc.add(pulse)
        return new_pc

    def get_qubit_pulses(self, *qubits):
        """Returns a new PulseSequence containing only the pulses on a specific set of qubits."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if not isinstance(pulse, CouplerFluxPulse):
                if pulse.qubit in qubits:
                    new_pc.add(pulse)
        return new_pc

    def coupler_pulses(self, *couplers):
        """Returns a new PulseSequence containing only the pulses on a specific set of couplers."""

        new_pc = PulseSequence()
        for pulse in self.pulses:
            if isinstance(pulse, CouplerFluxPulse):
                if pulse.qubit in couplers:
                    new_pc.add(pulse)
        return new_pc

    @property
    def is_empty(self):
        """Returns True if the sequence does not contain any pulses."""

        return len(self.pulses) == 0

    @property
    def finish(self) -> int:
        """Returns the time when the last pulse of the sequence finishes."""

        t: int = 0
        for pulse in self.pulses:
            if pulse.finish > t:
                t = pulse.finish
        return t

    @property
    def start(self) -> int:
        """Returns the start time of the first pulse of the sequence."""

        t = self.finish
        for pulse in self.pulses:
            if pulse.start < t:
                t = pulse.start
        return t

    @property
    def duration(self) -> int:
        """Returns duration of the sequence calculated as its finish - start times."""

        return self.finish - self.start

    @property
    def channels(self) -> list:
        """Returns list containing the channels used by the pulses in the sequence."""

        channels = []
        for pulse in self.pulses:
            if not pulse.channel in channels:
                channels.append(pulse.channel)
        channels.sort()
        return channels

    @property
    def qubits(self) -> list:
        """Returns list containing the qubits associated with the pulses in the sequence."""

        qubits = []
        for pulse in self.pulses:
            if not pulse.qubit in qubits:
                qubits.append(pulse.qubit)
        qubits.sort()
        return qubits

    def get_pulse_overlaps(self):  # -> dict((int,int): PulseSequence):
        """Returns a dictionary of slices of time (tuples with start and finish times) where pulses overlap."""

        times = []
        for pulse in self.pulses:
            if not pulse.start in times:
                times.append(pulse.start)
            if not pulse.finish in times:
                times.append(pulse.finish)
        times.sort()

        overlaps = {}
        for n in range(len(times) - 1):
            overlaps[(times[n], times[n + 1])] = PulseSequence()
            for pulse in self.pulses:
                if (pulse.start <= times[n]) & (pulse.finish >= times[n + 1]):
                    overlaps[(times[n], times[n + 1])] += pulse
        return overlaps

    def separate_overlapping_pulses(self):  # -> dict((int,int): PulseSequence):
        """Separates a sequence of overlapping pulses into a list of non-overlapping sequences."""

        # This routine separates the pulses of a sequence into non-overlapping sets
        # but it does not check if the frequencies of the pulses within a set have the same frequency

        separated_pulses = []
        for new_pulse in self.pulses:
            stored = False
            for ps in separated_pulses:
                overlaps = False
                for existing_pulse in ps:
                    if new_pulse.start < existing_pulse.finish and new_pulse.finish > existing_pulse.start:
                        overlaps = True
                        break
                if not overlaps:
                    ps.add(new_pulse)
                    stored = True
                    break
            if not stored:
                separated_pulses.append(PulseSequence(new_pulse))
        return separated_pulses

    # TODO: Implement separate_different_frequency_pulses()

    @property
    def pulses_overlap(self) -> bool:
        """Returns True if any of the pulses in the sequence overlap."""

        overlap = False
        for pc in self.get_pulse_overlaps().values():
            if pc.count > 1:
                overlap = True
        return overlap

    def plot(self, savefig_filename=None):
        """Plots the sequence of pulses.

        Args:
            savefig_filename (str): a file path. If provided the plot is save to a file.
        """

        if not self.is_empty:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec

            fig = plt.figure(figsize=(14, 2 * self.count), dpi=200)
            gs = gridspec.GridSpec(ncols=1, nrows=self.count)
            vertical_lines = []
            for pulse in self.pulses:
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
                        if isinstance(pulse, SplitPulse):
                            idx = slice(pulse.window_start - pulse.start, pulse.window_finish - pulse.start)
                            num_samples = len(pulse.shape.modulated_waveform_i.data[idx])
                            time = pulse.window_start + np.arange(num_samples) / PulseShape.SAMPLING_RATE * 1e9
                            ax.plot(time, pulse.shape.modulated_waveform_q.data[idx], c="lightgrey")
                            ax.plot(
                                time,
                                pulse.shape.modulated_waveform_i.data[idx],
                                c=f"C{str(n)}",
                            )
                            ax.plot(
                                time,
                                pulse.shape.envelope_waveform_i.data[idx],
                                c=f"C{str(n)}",
                            )
                            ax.plot(
                                time,
                                -pulse.shape.envelope_waveform_i.data[idx],
                                c=f"C{str(n)}",
                            )
                        else:
                            num_samples = len(pulse.shape.modulated_waveform_i)
                            time = pulse.start + np.arange(num_samples) / PulseShape.SAMPLING_RATE * 1e9
                            ax.plot(
                                time,
                                pulse.shape.modulated_waveform_q.data,
                                c="lightgrey",
                            )
                            ax.plot(
                                time,
                                pulse.shape.modulated_waveform_i.data,
                                c=f"C{str(n)}",
                            )
                            ax.plot(
                                time,
                                pulse.shape.envelope_waveform_i.data,
                                c=f"C{str(n)}",
                            )
                            ax.plot(
                                time,
                                -pulse.shape.envelope_waveform_i.data,
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
