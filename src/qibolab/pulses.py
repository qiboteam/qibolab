"""Pulse abstractions."""
import numpy as np
import re
from abc import ABC, abstractmethod
from qibo.config import raise_error


class Pulse:
    """Describes a single pulse to be added to waveform array.

    Args:
        start (float): Start time of pulse in ns.
        duration (float): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [0 to 1].
        frequency (float): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        phase (float): To be added.
        shape: (str): {'Rectangular', 'Gaussian(rel_sigma)', 'DRAG(rel_sigma, beta)'} Pulse shape.
            See :py:mod:`qibolab.pulses_shapes` for list of available shapes.
        channel (int/str): Specifies the device that will execute this pulse.
        type (str): {'ro', 'qd', 'qf'} type of pulse {ReadOut, Qubit Drive, Qubit Flux}
        offset_i (float): Optional pulse I offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        offset_q (float): Optional pulse Q offset (unitless).
            (amplitude + offset) should be between [0 and 1].
        qubit (int): qubit associated with the pulse

    Example:
        .. code-block:: python

            from qibolab.pulses import Pulse
            from qibolab.pulse_shapes import Gaussian

            # define pulse with Gaussian shape
            pulse = Pulse(start=0,
                          duration=60,
                          amplitude=0.3,
                          frequency=200000000.0,
                          phase=0,
                          shape=Gaussian(5),
                          channel=1,
                          type='qd')
    """
    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'qd', offset_i=0, offset_q=0, qubit=0):
        self.start = start # absolut pulse start time (does not depend on other pulses of the sequence)
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.shape = shape
        self.channel = channel
        self.offset_i = offset_i
        self.offset_q = offset_q
        self.qubit = qubit
        self.type = type

        shape_name = re.findall('(\w+)', shape)[0]
        shape_parameters = re.findall('(\w+)', shape)[1:]
        self.shape_object = globals()[shape_name](self, *shape_parameters) # eval(f"{shape_name}(self, {shape_parameters})")

    @property
    def serial(self):
        return f"Pulse({self.start}, {self.duration}, {format(self.amplitude, '.3f')}, {self.frequency}, {format(self.phase, '.3f')}, '{self.shape}', {self.channel}, '{self.type}')"

    @property
    def envelope_i(self):
        return  self.shape_object.envelope_i

    @property
    def envelope_q(self):
        return  self.shape_object.envelope_q

    def __repr__(self):
        return self.serial


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """

    def __init__(self, start, duration, amplitude, frequency, phase, shape, channel, type = 'ro', offset_i=0, offset_q=0, qubit=0):
        super().__init__(start, duration, amplitude, frequency, phase, shape, channel, type , offset_i, offset_q, qubit)

    @property
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.3f')}, {self.frequency}, {format(self.phase, '.3f')}, '{self.shape}', {self.channel}, '{self.type}')"




class PulseShape(ABC):
    """Abstract class for pulse shapes"""

    @property
    @abstractmethod
    def envelope_i(self): # pragma: no cover
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def envelope_q(self): # pragma: no cover
        raise_error(NotImplementedError)


class Rectangular(PulseShape):
    """
    Rectangular pulse shape.

    Args:
        pulse (Pulse): pulse associated with the shape
    """
    def __init__(self, pulse):
        self.name = "Rectangular"
        self.pulse = pulse

    @property
    def envelope_i(self):
        return self.pulse.amplitude * np.ones(int(self.pulse.duration))

    @property
    def envelope_q(self):
        return np.zeros(int(self.pulse.duration))

    def __repr__(self):
        return f"{self.name}()"

class Gaussian(PulseShape):
    """
    Gaussian pulse shape.

    Args:
        pulse (Pulse): pulse associated with the shape
        rel_sigma (int): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma

    .. math::

        A\exp^{-\\frac{1}{2}\\frac{(t-\mu)^2}{\sigma^2}}
    """

    def __init__(self, pulse, rel_sigma):
        self.name = "Gaussian"
        self.pulse = pulse
        self.rel_sigma = float(rel_sigma)

    @property
    def envelope_i(self):
        x = np.arange(0,self.pulse.duration,1)
        return self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        # same as: self.pulse.amplitude * gaussian(int(self.pulse.duration), std=int(self.pulse.duration/self.rel_sigma))

    @property
    def envelope_q(self):
        return np.zeros(int(self.pulse.duration))

    def __repr__(self):
        return f"{self.name}({self.rel_sigma})"


class Drag(PulseShape):
    """
    Derivative Removal by Adiabatic Gate (DRAG) pulse shape.

    Args:
        pulse (Pulse): pulse associated with the shape
        rel_sigma (int): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma

    .. math::


    """

    def __init__(self, pulse, rel_sigma, beta):
        self.name = "Drag"
        self.pulse = pulse
        self.rel_sigma = float(rel_sigma)
        self.beta = float(beta)

    @property
    def envelope_i(self):
        x = np.arange(0,self.pulse.duration,1)
        i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        return i

    @property
    def envelope_q(self):
        x = np.arange(0,self.pulse.duration,1)
        i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(self.pulse.duration-1)/2)**2)/(((self.pulse.duration)/self.rel_sigma)**2)))
        q = self.beta * (-(x-(self.pulse.duration-1)/2)/((self.pulse.duration/self.rel_sigma)**2)) * i
        return q

    def __repr__(self):
        return f"{self.name}({self.rel_sigma}, {self.beta})"


class PulseSequence:
    """List of pulses.

    Holds a separate list for each instrument.
    """

    def __init__(self):
        super().__init__()
        self.ro_pulses = []
        self.qd_pulses = []
        self.qf_pulses = []
        self.pulses = []
        self.time = 0
        self.phase = 0

    def __len__(self):
        return len(self.pulses)

    @property
    def serial(self):
        """Serial form of the whole sequence using the serial of each pulse."""
        return ", ".join(pulse.serial for pulse in self.pulses)

    def add(self, pulse):
        """Add a pulse to the sequence.

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to add.

        Example:
            .. code-block:: python

                from qibolab.pulses import PulseSequence, Pulse, ReadoutPulse, Rectangular, Gaussian, Drag
                # define two arbitrary pulses
                pulse1 = Pulse( start=0,
                                duration=60,
                                amplitude=0.3,
                                frequency=200_000_000.0,
                                phase=0,
                                shape=Gaussian(5),
                                channel=1,
                                type='qd')
                pulse2 = Pulse( start=70,
                                duration=2000,
                                amplitude=0.5,
                                frequency=20_000_000.0,
                                phase=0,
                                shape=Rectangular(),
                                channel=2,
                                type='ro')

                # define the pulse sequence
                sequence = PulseSequence()

                # add pulses to the pulse sequence
                sequence.add(pulse1)
                sequence.add(pulse2)
        """
        if pulse.type == "ro":
            self.ro_pulses.append(pulse)
        elif pulse.type == "qd":
            self.qd_pulses.append(pulse)
        elif pulse.type == "qf":
            self.qf_pulses.append(pulse)

        self.pulses.append(pulse)

    def add_u3(self, platform, theta, phi, lam, qubit=0):
        """Add pulses that implement a U3 gate.

        Args:
            theta, phi, lam (float): Parameters of the U3 gate.
        """
        # apply RZ(lam)
        self.phase += lam
        # Fetch pi/2 pulse from calibration
        RX90_pulse_1= platform.RX90_pulse(qubit, self.time, self.phase)
        # apply RX(pi/2)
        self.add(RX90_pulse_1)
        self.time += RX90_pulse_1.duration
        # apply RZ(theta)
        self.phase += theta
        # Fetch pi/2 pulse from calibration
        RX90_pulse_2= platform.RX90_pulse(qubit, self.time, self.phase - np.pi)
        # apply RX(-pi/2)
        self.add(RX90_pulse_2)
        self.time += RX90_pulse_2.duration
        # apply RZ(phi)
        self.phase += phi

    def add_measurement(self, platform, qubit=0):
        """Add measurement pulse."""
        MZ_pulse = platform.MZ_pulse(qubit, self.time, self.phase)
        self.add(MZ_pulse)
        self.time += MZ_pulse.duration
