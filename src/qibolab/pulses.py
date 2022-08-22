"""Pulse and PulseSequence classes."""
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from qibolab.symbolic import intSymbolicExpression as se_int
from qibolab.symbolic import floatSymbolicExpression as se_float


class PulseType(Enum):
    READOUT = 'ro'
    DRIVE = 'qd'
    FLUX = 'qf'


class Waveform:
    DECIMALS = 5
    def __init__(self, data:np.ndarray):
        self.data = data
        self.serial: str = ""

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(str(np.around(self.data, Waveform.DECIMALS) + 0))

    def __repr__(self):
        return self.serial

    def plot(self): 
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(14, 5), dpi=120)
        plt.plot(self.data, c='C0', linestyle='dashed')
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        plt.suptitle(self.serial)
        plt.show()


class PulseShape(ABC):
    """Abstract class for pulse shapes"""

    SAMPLING_RATE = 1e9 # 1GSaPS

    @property
    @abstractmethod
    def envelope_waveform_i(self) -> Waveform: # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def envelope_waveform_q(self) -> Waveform: # pragma: no cover
        raise NotImplementedError

    @property
    def envelope_waveforms(self) -> tuple[Waveform, Waveform]: # pragma: no cover
        return (self.envelope_waveform_i, self.envelope_waveform_q)
        
    @property
    def modulated_waveform_i(self) -> Waveform:
        return self.modulated_waveforms[0]
        
    @property
    def modulated_waveform_q(self) -> Waveform:
        return self.modulated_waveforms[1]

    @property
    def modulated_waveforms(self):
        if not self.pulse:
                raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse waveforms")
        
        pulse = self.pulse
        num_samples = int(pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
        time = np.arange(num_samples) / PulseShape.SAMPLING_RATE
        global_phase = 2 * np.pi * pulse.frequency * pulse.start / 1e9 # pulse start, duration and finish are in ns 
        cosalpha = np.cos(2 * np.pi * pulse.frequency * time + global_phase + pulse.relative_phase)
        sinalpha = np.sin(2 * np.pi * pulse.frequency * time + global_phase + pulse.relative_phase)

        mod_matrix = np.array([[ cosalpha, -sinalpha], 
                                [sinalpha, cosalpha]])

        (envelope_waveform_i, envelope_waveform_q) = self.envelope_waveforms
        result = []
        for n, t, ii, qq in zip(np.arange(num_samples), time, envelope_waveform_i.data, envelope_waveform_q.data):
            result.append(mod_matrix[:, :, n] @ np.array([ii, qq]))
        mod_signals = np.array(result)

        modulated_waveform_i = Waveform(mod_signals[:, 0])
        modulated_waveform_i.serial = f"Modulated_Waveform_I(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
        modulated_waveform_q = Waveform(mod_signals[:, 1])
        modulated_waveform_q.serial = f"Modulated_Waveform_Q(num_samples = {num_samples}, amplitude = {format(pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {str(pulse.shape)}, frequency = {format(pulse.frequency, '_')}, phase = {format(global_phase + pulse.relative_phase, '.6f').rstrip('0').rstrip('.')})"
        return (modulated_waveform_i, modulated_waveform_q)


class Rectangular(PulseShape):
    """
    Rectangular pulse shape.

    """
    def __init__(self):
        self.name = "Rectangular"
        self.pulse: Pulse = None

    @property
    def envelope_waveform_i(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            waveform = Waveform(self.pulse.amplitude * np.ones(num_samples))
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform            
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse waveforms")

    @property
    def envelope_waveform_q(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse waveforms")

    def __repr__(self):
        return f"{self.name}()"


class Gaussian(PulseShape):
    """
    Gaussian pulse shape.

    Args:
        rel_sigma (float): relative sigma so that the pulse standard deviation (sigma) = duration / rel_sigma

    .. math::

        A\exp^{-\\frac{1}{2}\\frac{(t-\mu)^2}{\sigma^2}}
    """

    def __init__(self, rel_sigma: float):
        self.name = "Gaussian"
        self.pulse: Pulse = None
        self.rel_sigma: float = float(rel_sigma)

    @property
    def envelope_waveform_i(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0,num_samples,1)
            waveform = Waveform(self.pulse.amplitude * np.exp(-(1/2)*(((x-(num_samples-1)/2)**2)/(((num_samples)/self.rel_sigma)**2))))
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

    @property
    def envelope_waveform_q(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            waveform = Waveform(np.zeros(num_samples))
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

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

    @property
    def envelope_waveform_i(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0,num_samples,1)
            i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(num_samples-1)/2)**2)/(((num_samples)/self.rel_sigma)**2)))
            waveform = Waveform(i)
            waveform.serial = f"Envelope_Waveform_I(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

    @property
    def envelope_waveform_q(self) -> Waveform:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0,num_samples,1)
            i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(num_samples-1)/2)**2)/(((num_samples)/self.rel_sigma)**2)))
            q = self.beta * (-(x-(num_samples-1)/2)/((num_samples/self.rel_sigma)**2)) * i * PulseShape.SAMPLING_RATE / 1e9
            waveform = Waveform(q)
            waveform.serial = f"Envelope_Waveform_Q(num_samples = {num_samples}, amplitude = {format(self.pulse.amplitude, '.6f').rstrip('0').rstrip('.')}, shape = {repr(self)})"
            return waveform
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

    def __repr__(self):
        return f"{self.name}({format(self.rel_sigma, '.6f').rstrip('0').rstrip('.')}, {format(self.beta, '.6f').rstrip('0').rstrip('.')})"


class Pulse:
    """A class to represent a pulse to be sent to the QPU.

    Args:
        start (int | intSymbolicExpression): Start time of pulse in ns.
        duration (int | intSymbolicExpression): Pulse duration in ns.
        amplitude (float): Pulse digital amplitude (unitless) [0 to 1].
        frequency (int): Pulse Intermediate Frequency in Hz [10e6 to 300e6].
        relative_phase (float): To be added.
        shape: (PulseShape | str): {'Rectangular()', 'Gaussian(rel_sigma)', 'DRAG(rel_sigma, beta)'} Pulse shape.
            See :py:mod:`qibolab.pulses` for list of available shapes.
        channel (int | str): the channel on which the pulse should be synthesised.
        type (PulseType | str): {'ro', 'qd', 'qf'} type of pulse {ReadOut, Qubit Drive, Qubit Flux}
        qubit (int): qubit associated with the pulse

    Example:
        .. code-block:: python

            from qibolab.pulses import Pulse, Gaussian

            # define Gaussian drive pulse
            drive_pulse = Pulse(start=0,
                                duration=60,
                                amplitude=0.3,
                                frequency=-200_000_000,
                                relative_phase=0.0,
                                shape=Gaussian(5),
                                channel=1,
                                type=PulseType.DRIVE)

            # define Rectangular readout pulse
            readout_pulse = Pulse(start=intSymbolicExpression(60),
                                  duration=2000,
                                  amplitude=0.3,
                                  frequency=20_000_000,
                                  relative_phase=0.0,
                                  shape=Rectangular(),
                                  channel=2,
                                  type=PulseType.READOUT)
    """
    def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, type: PulseType | str  = PulseType.DRIVE, qubit: int | str = 0):

        self._start:se_int = None
        self._duration: se_int = None
        self._finish: se_int = None
        self._amplitude: float = None
        self._frequency: int = None
        self._relative_phase: float = None
        self._shape: PulseShape = None
        self._channel: int | str = None
        self._type: PulseType  = None
        self._qubit: int | str = None
        
        self.start = start 
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.relative_phase = relative_phase
        self.shape = shape
        self.channel = channel
        self.type = type
        self.qubit = qubit

    def __del__(self):
        del self._start
        del self._duration
        del self._finish
        # del self._shape TODO activate when returning a deep copy of shape or when making a deep copy of shape in init

    @property
    def start(self) -> int:
        return self._start.value
    
    @start.setter
    def start(self, value):
        if not isinstance(value, (se_int, int)):
            raise TypeError(f"start argument type should be intSymbolicExpression or int, got {type(value).__name__}")
        elif not value >= 0:
            raise ValueError(f"start argument must be >= 0, got {value}")
        if isinstance(value, se_int):
            #self._start = value
            #self._start = intSymbolicExpression(value)
            self._start = se_int(value.symbol)
        elif isinstance(value, int):
            self._start = se_int(value)

    @property
    def duration(self) -> int:
        return self._duration.value

    @duration.setter
    def duration(self, value):
        if not isinstance(value, (se_int, int)):
            raise TypeError(f"duration argument type should be intSymbolicExpression or int, got {type(value).__name__}")
        elif not value > 0:
            raise ValueError(f"duration argument must be >= 0, got {value}")
        if isinstance(value, se_int):
            self._duration = se_int(value.symbol)
        elif isinstance(value, int):
            self._duration = se_int(value)
        self._finish = self._start + self._duration

    @property
    def finish(self) -> int:
        return self._finish.value

    @property
    def se_start(self) -> se_int:
        return self._start

    @property
    def se_duration(self) -> se_int:
        return self._duration

    @property
    def se_finish(self) -> se_int:
        return self._finish

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(f"amplitude argument type should be float, got {type(value).__name__}")
        elif not ((value >= 0) & (value <= 1)):
            raise ValueError(f"amplitude argument must be >= 0 & <= 1, got {value}")
        self._amplitude = value

    @property
    def frequency(self)-> int:
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if not isinstance(value, (int|float)):
            raise TypeError(f"frequency argument type should be float, got {type(value).__name__}")
        elif isinstance(value, float):
            value = int(value)
        self._frequency = value

    @property
    def relative_phase(self) -> float:
        return self._relative_phase

    @relative_phase.setter
    def relative_phase(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(f"relative_phase argument type should be float, got {type(value).__name__}")
        self._relative_phase = value

    @property
    def shape(self) -> PulseShape:
        return self._shape

    @shape.setter
    def shape(self, value):
        if not isinstance(value, (PulseShape, str)):
            raise TypeError(f"shape argument type should be PulseShape or str, got {type(value).__name__}")
        if isinstance(value, PulseShape):
            self._shape = value
        elif isinstance(value, str):
            import re
            shape_name = re.findall('(\w+)', value)[0]
            if shape_name not in globals():
                raise ValueError(f"shape {value} not found")
            shape_parameters = re.findall(r'[\w+\d\.\d]+', value)[1:] 
            #TODO: create multiple tests to prove regex working correctly
            self._shape = globals()[shape_name](*shape_parameters)
        self._shape.pulse = self

    @property
    def channel(self) -> int | str:
        return self._channel

    @channel.setter
    def channel(self, value):
        if not isinstance(value, (int, str)):
            raise TypeError(f"channel argument type should be int or str, got {type(value).__name__}")
        self._channel = value

    @property
    def type(self) -> PulseType:
        return self._type

    @type.setter
    def type(self, value):
        if isinstance(value, PulseType):
            self._type = value
        elif isinstance(value, str):
            self._type = PulseType(value)
        else:
            raise TypeError(f"type argument should be PulseType or str, got {type(value).__name__}")

    @property
    def qubit(self) -> int | str:
        return self._qubit

    @qubit.setter
    def qubit(self, value):
        if not isinstance(value, (int, str)):
            raise TypeError(f"qubit argument type should be int or str, got {type(value).__name__}")
        self._qubit = value

    @property
    def serial(self) -> str:
        return f"Pulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel}, {self.type})"

    @property
    def envelope_waveform_i(self) -> Waveform:
        return self._shape.envelope_waveform_i

    @property
    def envelope_waveform_q(self) -> Waveform:
        return self._shape.envelope_waveform_q

    @property
    def envelope_waveforms(self) -> tuple[Waveform, Waveform]:
        return  (self._shape.envelope_waveform_i, self._shape.envelope_waveform_q)

    @property
    def modulated_waveform_i(self) -> Waveform:
        return self._shape.modulated_waveform_i

    @property
    def modulated_waveform_q(self) -> Waveform:
        return self._shape.modulated_waveform_q

    @property
    def modulated_waveforms(self) -> tuple[Waveform, Waveform]:
        return self._shape.modulated_waveforms

    def __repr__(self):
        return self.serial

    def __hash__(self):
        return hash(self.serial)

    def __eq__(self, other):
        if isinstance(other, Pulse):
            return self.serial == other.serial
        else:
            return False

    def __add__(self, other):
        if isinstance(other, Pulse):
            return PulseSequence(self, other)
        elif isinstance(other, PulseSequence):
            return PulseSequence(self, * other.pulses)
        else:
            raise TypeError(f'Expected Pulse or PulseSequence; got {type(other).__name__}')
            
    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 0:
            raise TypeError(f'argument n should be >=0, got {n}')
        return PulseSequence(* ([self.shallow_copy()] * n))

    def __rmul__(self, n):
        return self.__mul__(n)

    def deep_copy(self): # -> Pulse|ReadoutPulse|DrivePulse|FluxPulse:
        # return eval(self.serial)
        return Pulse(self.start, self.duration, self.amplitude, self.frequency, self.relative_phase, repr(self._shape), self.channel, self.type, self.qubit)

    def shallow_copy(self): # -> Pulse:
        return Pulse(self._start, self._duration, self._amplitude, self._frequency, self._relative_phase, self._shape, self._channel, self._type, self._qubit)
 
    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import numpy as np
        num_samples = int(self.duration / 1e9 * PulseShape.SAMPLING_RATE)
        time = self.start + np.arange(num_samples) / PulseShape.SAMPLING_RATE * 1e9
        fig = plt.figure(figsize=(14, 5), dpi=120)
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(time, self.shape.envelope_waveform_i.data, label='envelope i', c='C0', linestyle='dashed')
        ax1.plot(time, self.shape.envelope_waveform_q.data, label='envelope q', c='C1', linestyle='dashed')
        ax1.plot(time, self.shape.modulated_waveform_i.data, label='modulated i', c='C0')
        ax1.plot(time, self.shape.modulated_waveform_q.data, label='modulated q', c='C1')
        ax1.plot(time, -self.shape.envelope_waveform_i.data, c='silver', linestyle='dashed')
        ax1.set_xlabel('Time [ns]')
        ax1.set_ylabel('Amplitude')

        ax1.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax1.axis([self.start, self.finish, -1, 1])
        ax1.legend()

        ax2 = plt.subplot(gs[1])
        ax2.plot(self.shape.modulated_waveform_i.data, self.shape.modulated_waveform_q.data, label='modulated', c='C3')
        ax2.plot(self.shape.envelope_waveform_i.data, self.shape.envelope_waveform_q.data, label='envelope', c='C2')
        ax2.plot(self.shape.modulated_waveform_i.data[0], self.shape.modulated_waveform_q.data[0], marker="o", markersize=5, label='start', c='lightcoral')
        ax2.plot(self.shape.modulated_waveform_i.data[-1], self.shape.modulated_waveform_q.data[-1], marker="o", markersize=5, label='finish', c='darkred')

        ax2.plot(np.cos(time * 2 * np.pi / self.duration), np.sin(time * 2 * np.pi / self.duration), c='silver', linestyle='dashed')

        ax2.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        plt.suptitle(self.serial)
        plt.show()
        return


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """
    def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, qubit: int | str = 0):
        super().__init__(start, duration, amplitude, frequency, relative_phase, shape, channel, type =  PulseType.READOUT, qubit = qubit)

    @property
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"


class DrivePulse(Pulse):
    """Describes a qubit drive pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """
    def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, qubit: int | str = 0):
        super().__init__(start, duration, amplitude, frequency, relative_phase, shape, channel, type =  PulseType.DRIVE, qubit = qubit)

    @property
    def serial(self):
        return f"DrivePulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"


class FluxPulse(Pulse):
    """Describes a qubit drive pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """
    def __init__(self, start:int | se_int, duration:int | se_int, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, qubit: int | str = 0):
        super().__init__(start, duration, amplitude, frequency, relative_phase, shape, channel, type =  PulseType.FLUX, qubit = qubit)

    @property
    def serial(self):
        return f"FluxPulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"


class SplitPulse(Pulse):
    
    def __init__(self, pulse: Pulse, window_start:int = None, window_finish:int = None):
        super().__init__(pulse.start, pulse.duration, pulse.amplitude, pulse.frequency, pulse.relative_phase, eval(str(pulse.shape)), pulse.channel, type =  pulse.type, qubit = pulse.qubit)
        self._window_start:int = pulse.start
        self._window_finish:int = pulse.finish
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
        return f"SequencerPulse({self.window_start}, {self.window_duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"



    @property
    def envelope_waveform_i(self) -> Waveform:
        waveform = Waveform(self._shape.envelope_waveform_i.data[self._window_start - self.start : self._window_finish - self.start])
        waveform.serial = self._shape.envelope_waveform_i.serial + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        return  waveform

    @property
    def envelope_waveform_q(self) -> Waveform:
        waveform = Waveform(self._shape.modulated_waveform_i.data[self._window_start - self.start : self._window_finish - self.start])
        waveform.serial = self._shape.modulated_waveform_i.serial + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        return  waveform

    @property
    def envelope_waveforms(self) -> tuple[Waveform, Waveform]:
        return  (self.envelope_waveform_i, self.envelope_waveform_q)

    @property
    def modulated_waveform_i(self) -> Waveform:
        waveform = Waveform(self._shape.envelope_waveform_q.data[self._window_start - self.start : self._window_finish - self.start])
        waveform.serial = self._shape.envelope_waveform_q.serial + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        return  waveform

    @property
    def modulated_waveform_q(self) -> Waveform:
        waveform = Waveform(self._shape.modulated_waveform_q.data[self._window_start - self.start : self._window_finish - self.start])
        waveform.serial = self._shape.modulated_waveform_q.serial + f"[{self._window_start - self.start} : {self._window_finish - self.start}]"
        return  waveform

    @property
    def modulated_waveforms(self) -> tuple[Waveform, Waveform]:
        return  (self.modulated_waveform_i, self.modulated_waveform_q)

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import numpy as np

        time = self.window_start + np.arange(int(self.window_duration / 1e9 * PulseShape.SAMPLING_RATE)) / PulseShape.SAMPLING_RATE * 1e9
        
        fig = plt.figure(figsize=(14, 5), dpi=120)
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(time, self.shape.envelope_waveform_i.data[self._window_start - self.start : self._window_finish - self.start], label='envelope i', c='C0', linestyle='dashed')
        ax1.plot(time, self.shape.envelope_waveform_q.data[self._window_start - self.start : self._window_finish - self.start], label='envelope q', c='C1', linestyle='dashed')
        ax1.plot(time, self.shape.modulated_waveform_i.data[self._window_start - self.start : self._window_finish - self.start], label='modulated i', c='C0')
        ax1.plot(time, self.shape.modulated_waveform_q.data[self._window_start - self.start : self._window_finish - self.start], label='modulated q', c='C1')
        ax1.plot(time, -self.shape.envelope_waveform_i.data[self._window_start - self.start : self._window_finish - self.start], c='silver', linestyle='dashed')
        ax1.set_xlabel('Time [ns]')
        ax1.set_ylabel('Amplitude')

        ax1.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax1.axis([self.window_start, self._window_finish, -1, 1])
        ax1.legend()

        ax2 = plt.subplot(gs[1])
        ax2.plot(self.shape.modulated_waveform_i.data[self._window_start - self.start : self._window_finish - self.start], self.shape.modulated_waveform_q.data[self._window_start - self.start : self._window_finish - self.start], label='modulated', c='C3')
        ax2.plot(self.shape.envelope_waveform_i.data[self._window_start - self.start : self._window_finish - self.start], self.shape.envelope_waveform_q.data[self._window_start - self.start : self._window_finish - self.start], label='envelope', c='C2')
        ax2.plot(np.cos(time * 2 * np.pi / self.window_duration), np.sin(time * 2 * np.pi / self.window_duration), c='silver', linestyle='dashed')

        ax2.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        plt.show()
        return


class PulseSequence(): 
    def __init__(self, *pulses):
        self.pulses: list[Pulse] = []
        self.add(*pulses)

    def __len__(self):
        return len(self.pulses)

    def __iter__(self):
        return iter(self.pulses)
    
    def __getitem__(self, index):
        return self.pulses[index]

    def __setitem__ (self, index, value):
        self.pulses[index] = value

    def __delitem__(self, index):
        del self.pulses[index]

    def __contains__(self, pulse):
        return pulse in self.pulses
    
    def __repr__(self):
        return self.serial

    @property
    def serial(self):
        return 'PulseSequence\n' + '\n'.join(f'{pulse.serial}' for pulse in self.pulses)

    def __eq__(self, other):
        if not isinstance(other, PulseSequence):
            raise TypeError(f'Expected PulseSequence; got {type(other).__name__}')    
        return (self.serial == other.serial)

    def __ne__(self, other):
        if not isinstance(other, PulseSequence):
            raise TypeError(f'Expected PulseSequence; got {type(other).__name__}')    
        return (self.serial != other.serial)

    def __hash__(self):
        return hash(self.serial)

    def __add__(self, other):
        if isinstance(other, PulseSequence):
            return PulseSequence(* self.pulses, * other.pulses)
        elif isinstance(other, Pulse):
            return PulseSequence(* self.pulses, other)
        else:
            raise TypeError(f'Expected PulseSequence or Pulse; got {type(other).__name__}')

    def __radd__(self, other):
        if isinstance(other, PulseSequence):
            return PulseSequence(* other.pulses, * self.pulses)
        elif isinstance(other, Pulse):
            return PulseSequence(other, * self.pulses)
        else:
            raise TypeError(f'Expected PulseSequence or Pulse; got {type(other).__name__}')
    
    def __iadd__(self, other):
        if isinstance(other, PulseSequence):
            self.add(* other.pulses)
        elif isinstance(other, Pulse):
            self.add(other)
        else:
            raise TypeError(f'Expected PulseSequence or Pulse; got {type(other).__name__}')
        return self
        
    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 0:
            raise TypeError(f'argument n should be >=0, got {n}')
        return PulseSequence(* (self.pulses * n))
        
    def __rmul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 0:
            raise TypeError(f'argument n should be >=0, got {n}')
        return PulseSequence(* (self.pulses * n))

    def __imul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 1:
            raise TypeError(f'argument n should be >=1, got {n}')
        original_set = self.shallow_copy()
        for x in range(n - 1):
            self.add(* original_set.pulses)
        return self

    @property
    def count(self):
        return len(self.pulses)

    def add(self, *pulses):
        for pulse in pulses:
            self.pulses.append(pulse)
        self.pulses.sort(key=lambda item: (item.channel, item.start))
        
    def append_at_end_of_channel(self, *pulses):
        for pulse in pulses:
            pulse.start = self.pulses.get_channel_pulses(pulse.channel).finish
            self.add(pulse)

    def append_at_end_of_sequence(self, *pulses):
        for pulse in pulses:
            pulse.start = self.pulses.finish
            self.add(pulse)

    def index(self, pulse):
        return self.pulses.index(pulse)

    def pop(self, index = -1):
        return self.pulses.pop(index)

    def remove(self, pulse):
        while pulse in self.pulses:
            self.pulses.remove(pulse)

    def clear(self):
        self.pulses.clear()

    def shallow_copy(self):
        return PulseSequence(* self.pulses)

    def deep_copy(self):
        return PulseSequence(* [pulse.deep_copy() for pulse in self.pulses])

    @property
    def ro_pulses(self):
        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type == PulseType.READOUT:
                new_pc.add(pulse)
        return new_pc

    @property
    def qd_pulses(self):
        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type == PulseType.DRIVE:
                new_pc.add(pulse)
        return new_pc

    @property
    def qf_pulses(self):
        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.type == PulseType.FLUX:
                new_pc.add(pulse)
        return new_pc


    def get_channel_pulses(self, * channels):
        new_pc = PulseSequence()
        for pulse in self.pulses:
            if pulse.channel in channels:
                new_pc.add(pulse)
        return new_pc

    @property
    def is_empty(self):
        return len(self.pulses) == 0

    @property
    def finish(self) -> int:
        t: int = 0
        for pulse in self.pulses:
            if pulse.start + pulse.finish > t:
                t = pulse.start + pulse.finish
        return t

    @property
    def start(self) -> int:
        t = self.finish
        for pulse in self.pulses:
            if pulse.start < t:
                t = pulse.start
        return t

    @property
    def duration(self) -> int:
        return self.finish - self.start

    @property
    def channels(self) -> list:
        channels = []
        for pulse in self.pulses:
            if not pulse.channel in channels:
                channels.append(pulse.channel)
        channels.sort()
        return channels

    def get_pulse_overlaps(self): # -> dict((int,int): PulseSequence):
        times = []
        for pulse in self.pulses:
            if not pulse.start in times:
                times.append(pulse.start)
            if not pulse.finish in times:
                times.append(pulse.finish)
        times.sort()

        overlaps = {}
        for n in range(len(times)-1):
            overlaps[(times[n], times[n+1])] = PulseSequence()
            for pulse in self.pulses:
                if (pulse.start <= times[n]) & (pulse.finish >= times[n+1]):
                    overlaps[(times[n], times[n+1])] += pulse
        return overlaps

    def separate_overlapping_pulses(self): # -> dict((int,int): PulseSequence):
        separated_pulses = []
        for new_pulse in self.pulses:
            stored = False
            for ps in separated_pulses:
                overlaps = False
                for existing_pulse in ps:
                    if (new_pulse.start >= existing_pulse.start and new_pulse.start <= existing_pulse.finish
                        ) or (
                        existing_pulse.start >= new_pulse.start and existing_pulse.start <= new_pulse.finish):
                        overlaps = True
                        break
                if not overlaps:
                    ps.add(new_pulse)
                    stored = True
            if not stored:
                separated_pulses.append(PulseSequence(new_pulse))
        return separated_pulses

    @property
    def pulses_overlap(self) -> bool:
        overlap = False
        for pc in self.get_pulse_overlaps().values():
            if pc.count > 1:
                overlap = True
        return overlap

    def plot(self):
        if not self.is_empty:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            import numpy as np
            fig = plt.figure(figsize=(14, 2 * self.count), dpi=120)
            gs = gridspec.GridSpec(ncols=1, nrows=self.count)
            vertical_lines = []
            for pulse in self.pulses:
                vertical_lines.append(pulse.start)
                vertical_lines.append(pulse.finish)

            for n, channel in enumerate(self.channels):
                channel_pulses = self.get_channel_pulses(channel)
                ax = plt.subplot(gs[n])
                ax.axis([0, self.finish, -1, 1])
                for pulse in channel_pulses:
                    if isinstance(pulse, SplitPulse):
                        time = pulse.window_start + np.arange(pulse.window_duration)
                        ax.plot(time, pulse.shape.modulated_waveform_q.data[pulse.window_start - pulse.start : pulse.window_finish - pulse.start], c='lightgrey')
                        ax.plot(time, pulse.shape.modulated_waveform_i.data[pulse.window_start - pulse.start : pulse.window_finish - pulse.start], c=f'C{str(n)}')
                        ax.plot(time, pulse.shape.envelope_waveform_i.data[pulse.window_start - pulse.start : pulse.window_finish - pulse.start], c=f'C{str(n)}')
                        ax.plot(time, -pulse.shape.envelope_waveform_i.data[pulse.window_start - pulse.start : pulse.window_finish - pulse.start], c=f'C{str(n)}')
                    else:
                        time = pulse.start + np.arange(pulse.duration)
                        ax.plot(time, pulse.shape.modulated_waveform_q.data, c='lightgrey')
                        ax.plot(time, pulse.shape.modulated_waveform_i.data, c=f'C{str(n)}')
                        ax.plot(time, pulse.shape.envelope_waveform_i.data, c=f'C{str(n)}')
                        ax.plot(time, -pulse.shape.envelope_waveform_i.data, c=f'C{str(n)}')
                    # TODO: if they overlap use different shades
                    ax.axhline(0, c='dimgrey')
                    ax.set_ylabel(f'channel {channel}')
                    for vl in vertical_lines:
                        ax.axvline(vl, c='slategrey', linestyle = '--')
                    ax.axis([0, self.finish, -1, 1])
                    ax.grid(b=True, which='both', axis='both', color='#CCCCCC', linestyle='-')
            plt.show()