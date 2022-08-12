"""Pulse and PulseCollection classes."""
from tkinter import N
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from traitlets import Bool


class PulseType(Enum):
    READOUT = 'ro'
    DRIVE = 'qd'
    FLUX = 'qf'

# class PulseShape(Enum):


class TimeVariable:
    count: int = 0
    instances: dict = {}

    @classmethod
    def clear_instances(cls):
        cls.instances.clear()
        cls.count = 0

    class CircularReferenceError(Exception):
        pass

    def __init__(self, expression = 0, name:str = ''): # (self, expression:str|int|TimeVariable = 0, name:str = ''):
        self._name: str = ''
        self._formula:str = ''

        if name == '':
            while True:
                name = '_tv' + str(TimeVariable.count)
                TimeVariable.count += 1
                if name not in TimeVariable.instances.keys():
                    break

        self.formula = expression
        self.name = name

# TODO 
# t0, t1 = TimeVariable(0, 't0', 't1') # t0 = 0, t1 = 0
# t0, t1 = TimeVariable([0, 5], ['t0', 't1']) # t0 = 0, t1 = 5
# or even better with a dictionary
# tv_dict = {}
# tv_dict = TimeVariable({t0: 0, t1: 5}) # t0 = 0, t1 = 5


    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name:str):
        if not isinstance(name, str):
            raise TypeError(f"name argument type should be str, got {type(name).__name__}")
        if name in TimeVariable.instances.keys():
            pass # Allows overwriting
            # raise KeyError(f"name should be unique, there is already a TimeVariable with name {name}: {TimeVariable.instances[name]}")
        if self._name == '':
            # Creation
            TimeVariable.instances[name] = self
        else:
            # Renaming
            TimeVariable.instances[name] = self                     # Add a new reference with the new name
            if not  self._name == name:
                del TimeVariable.instances[self._name]              # Remove the previous reference
            import re
            for tv in TimeVariable.instances.values():              # Update all TimeVariable formulas of the name change
                match_string = rf'\b{re.escape(self._name)}\b'
                replacement = name
                tv.formula = re.sub(match_string, replacement, tv.formula)
        self._name = name
        # test for CircularReferenceError
        self.evaluate(self._formula, self._name)


    def __getitem__(self, name):
        self.name = name
        return self

    @property
    def formula(self):
        return self._formula
    
    @formula.setter
    def formula(self, expression): # (self, expression:str|int|TimeVariable):
        if isinstance(expression, str):
            self.evaluate(expression)
            self._formula = expression
        elif isinstance(expression, int):
            self._formula = str(expression)
        elif isinstance(expression, TimeVariable):
            self._formula = expression._formula
            #self.name = expression._name #(TODO find a solution so that intermediate operations are not stored in the instances)
        else:
            raise TypeError(f"expression argument type should be int or TimeVariable, got {type(expression).__name__}")

    @property
    def value(self) -> int:
        return self.evaluate(self._formula, self._name)

    @value.setter
    def value(self, value:int):
        if isinstance(value, int):
            self._formula = str(value) 
        else:
            raise TypeError(f"value argument type should be int, got {type(value).__name__}")


    @property
    def is_constant(self) -> bool:
        try:
            if str(int(self._formula)) == self._formula:
                return True
            else:
                return False
        except:
            return False

    def __repr__(self):
        try:
            response =  f"{self._name}: {self._formula} = {self.value}"
        except TimeVariable.CircularReferenceError:
            response =  f"{self._name}: {self._formula} = CircularReferenceError"
        return response

    def evaluate(self, expression:str, * previous_evaluations) -> int:
        import re
        for name in TimeVariable.instances.keys():
            if name in expression:
                if name in previous_evaluations:
                    raise TimeVariable.CircularReferenceError(f"Circular Reference evaluating {expression}, variable {name} found in {previous_evaluations}")
                match_string = rf'\b{re.escape(name)}\b'
                replacement = str(self.evaluate(TimeVariable.instances[name]._formula, * previous_evaluations, TimeVariable.instances[name]._name))
                expression = re.sub(match_string, replacement, expression)
        try:
            result = eval(expression)
        except: 
            raise ValueError(f"The evaluation of the expression: {expression} returned an error")

        if not isinstance(result, int):
            raise TypeError(f"The evaluation of the expression: {expression} does not return an integer")
        return result

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __str__(self):
        return str(self.value)

    def __lt__(self, other):
        if isinstance(other, TimeVariable):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        else:
            raise TypeError(f"Comparison operators expect TimeVariable or int arguments, got {type(other).__name__}")
            
    def __gt__(self, other):
        if isinstance(other, TimeVariable):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        else:
            raise TypeError(f"Comparison operators expect TimeVariable or int arguments, got {type(other).__name__}")

    def __le__(self, other):
        if isinstance(other, TimeVariable):
            return self.value <= other.value
        elif isinstance(other, int):
            return self.value <= other
        else:
            raise TypeError(f"Comparison operators expect TimeVariable or int arguments, got {type(other).__name__}")

    def __ge__(self, other):
        if isinstance(other, TimeVariable):
            return self.value >= other.value
        elif isinstance(other, int):
            return self.value >= other
        else:
            raise TypeError(f"Comparison operators expect TimeVariable or int arguments, got {type(other).__name__}")

    def __eq__(self, other):
        if isinstance(other, TimeVariable):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            raise TypeError(f"Comparison operators expect TimeVariable or int arguments, got {type(other).__name__}")

    def __ne__(self, other):
        if isinstance(other, TimeVariable):
            return self.value != other.value
        elif isinstance(other, int):
            return self.value != other
        else:
            raise TypeError(f"Comparison operators expect TimeVariable or int arguments, got {type(other).__name__}")

    def __add__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({self.name} + {other.name})")
        elif isinstance(other, int):
            return TimeVariable(f"({self.name} + {str(other)})")

    def __radd__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({other.formula} + {self.name})")
        elif isinstance(other, int):
            return TimeVariable(f"({str(other)} + {self.name})")

    def __sub__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({self.name} - {other.formula})")
        elif isinstance(other, int):
            return TimeVariable(f"({self.name} - {str(other)})")
            
    def __rsub__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({other.formula} - {self.name})")
        elif isinstance(other, int):
            return TimeVariable(f"({str(other)} - {self.name})")

    def __mul__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({self.name} * {other.formula})")
        elif isinstance(other, int):
            return TimeVariable(f"({self.name} * {str(other)})")
            
    def __rmul__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({other.formula} * {self.name})")
        elif isinstance(other, int):
            return TimeVariable(f"({str(other)} * {self.name})")

    def __floordiv__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({self.name} // {other.formula})")
        elif isinstance(other, int):
            return TimeVariable(f"({self.name} // {str(other)})")
            
    def __rfloordiv__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({other.formula} // {self.name})")
        elif isinstance(other, int):
            return TimeVariable(f"({str(other)} // {self.name})")

    def __mod__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({self.name} % {other.formula})")
        elif isinstance(other, int):
            return TimeVariable(f"({self.name} % {str(other)})")
            
    def __rmod__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            return TimeVariable(f"({other.formula} % {self.name})")
        elif isinstance(other, int):
            return TimeVariable(f"({str(other)} % {self.name})")

    def __iadd__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            self.formula =  f"({self.formula} + {other.name})"
            return self
        elif isinstance(other, int):
            if self.is_constant:
                self.formula = str(int(self.formula) + other)
            else:
                self.formula = f"({self.formula} + {str(other)})"
            return self

    def __isub__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            self.formula =  f"({self.formula} - {other.name})"
            return self
        elif isinstance(other, int):
            if self.is_constant:
                self.formula = str(int(self.formula) - other)
            else:
                self.formula = f"({self.formula} - {str(other)})"
            return self

    def __imul__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            self.formula =  f"({self.formula} * {other.name})"
            return self
        elif isinstance(other, int):
            if self.is_constant:
                self.formula = str(int(self.formula) * other)
            else:
                self.formula = f"({self.formula} * {str(other)})"
            return self

    def __ifloordiv__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            self.formula =  f"({self.formula} // {other.name})"
            return self
        elif isinstance(other, int):
            if self.is_constant:
                self.formula = str(int(self.formula) // other)
            else:
                self.formula = f"({self.formula} // {str(other)})"
            return self

    def __imod__(self, other): # -> TimeVariable:
        if isinstance(other, TimeVariable):
            self.formula =  f"({self.formula} % {other.name})"
            return self
        elif isinstance(other, int):
            if self.is_constant:
                self.formula = str(int(self.formula) % other)
            else:
                self.formula = f"({self.formula} % {str(other)})"
            return self

    def __neg__(self): # -> TimeVariable:
        return TimeVariable(f"-{self.name}")

    def __hash__(self):
        return hash(self._name)


class PulseShape(ABC):
    """Abstract class for pulse shapes"""

    SAMPLING_RATE = 1e9 # 1GSaPS

    @property
    @abstractmethod
    def envelope_waveform_i(self) -> np.ndarray: # pragma: no cover
        raise NotImplementedError

    @property
    @abstractmethod
    def envelope_waveform_q(self) -> np.ndarray: # pragma: no cover
        raise NotImplementedError

    def _modulate_waveforms(self):
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

        result = []
        for n, t, ii, qq in zip(np.arange(num_samples), time, self.envelope_waveform_i, self.envelope_waveform_q):
            result.append(mod_matrix[:, :, n] @ np.array([ii, qq]))
        mod_signals = np.array(result)

        self._modulated_waveform_i = mod_signals[:, 0]
        self._modulated_waveform_q = mod_signals[:, 1]
        
    @property
    def modulated_waveform_i(self) -> np.ndarray:
        self._modulate_waveforms()
        return self._modulated_waveform_i
        
    @property
    def modulated_waveform_q(self) -> np.ndarray:
        self._modulate_waveforms()
        return self._modulated_waveform_q


class Rectangular(PulseShape):
    """
    Rectangular pulse shape.

    """
    def __init__(self):
        self.name = "Rectangular"
        self.pulse: Pulse = None

    @property
    def envelope_waveform_i(self) -> np.ndarray:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            return self.pulse.amplitude * np.ones(num_samples)
            
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse waveforms")

    @property
    def envelope_waveform_q(self) -> np.ndarray:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            return np.zeros(num_samples)
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
    def envelope_waveform_i(self) -> np.ndarray:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0,num_samples,1)
            return self.pulse.amplitude * np.exp(-(1/2)*(((x-(num_samples-1)/2)**2)/(((num_samples)/self.rel_sigma)**2)))
            # same as: self.pulse.amplitude * gaussian(num_samples, std=int(num_samples/self.rel_sigma))
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

    @property
    def envelope_waveform_q(self) -> np.ndarray:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            return np.zeros(num_samples)
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
    def envelope_waveform_i(self) -> np.ndarray:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0,num_samples,1)
            i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(num_samples-1)/2)**2)/(((num_samples)/self.rel_sigma)**2)))
            return i
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

    @property
    def envelope_waveform_q(self) -> np.ndarray:
        if self.pulse:
            num_samples = int(self.pulse.duration / 1e9 * PulseShape.SAMPLING_RATE)
            x = np.arange(0,num_samples,1)
            i = self.pulse.amplitude * np.exp(-(1/2)*(((x-(num_samples-1)/2)**2)/(((num_samples)/self.rel_sigma)**2)))
            q = self.beta * (-(x-(num_samples-1)/2)/((num_samples/self.rel_sigma)**2)) * i * PulseShape.SAMPLING_RATE / 1e9
            return q
        else:
            raise Exception("PulseShape attribute pulse must be initialised in order to be able to generate pulse envelopes")

    def __repr__(self):
        return f"{self.name}({format(self.rel_sigma, '.6f').rstrip('0').rstrip('.')}, {format(self.beta, '.6f').rstrip('0').rstrip('.')})"


class Pulse:
    """A class to represent a pulse to be sent to the QPU.

    Args:
        start (int | TimeVariable): Start time of pulse in ns.
        duration (int | TimeVariable): Pulse duration in ns.
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
            readout_pulse = Pulse(start=TimeVariable(60),
                                  duration=2000,
                                  amplitude=0.3,
                                  frequency=20_000_000,
                                  relative_phase=0.0,
                                  shape=Rectangular(),
                                  channel=2,
                                  type=PulseType.READOUT)
    """
    def __init__(self, start:int | TimeVariable, duration:int | TimeVariable, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, type: PulseType | str  = PulseType.DRIVE, qubit: int | str = 0):

        self._start:TimeVariable = None
        self._duration: TimeVariable = None
        self._finish: TimeVariable = None
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
        if not isinstance(value, (TimeVariable, int)):
            raise TypeError(f"start argument type should be TimeVariable or int, got {type(value).__name__}")
        elif not value >= 0:
            raise ValueError(f"start argument must be >= 0, got {value}")
        if isinstance(value, TimeVariable):
            #self._start = value
            #self._start = TimeVariable(value)
            self._start = TimeVariable(value.name)
        elif isinstance(value, int):
            self._start = TimeVariable(value)

    @property
    def duration(self) -> int:
        return self._duration.value

    @duration.setter
    def duration(self, value):
        if not isinstance(value, (TimeVariable, int)):
            raise TypeError(f"duration argument type should be TimeVariable or int, got {type(value).__name__}")
        elif not value > 0:
            raise ValueError(f"duration argument must be >= 0, got {value}")
        if isinstance(value, TimeVariable):
            self._duration = TimeVariable(value.name)
        elif isinstance(value, int):
            self._duration = TimeVariable(value)
        self._finish = self._start + self._duration

    @property
    def finish(self) -> int:
        return self._finish.value

    @property
    def tv_start(self) -> TimeVariable:
        return self._start

    @property
    def tv_duration(self) -> TimeVariable:
        return self._duration

    @property
    def tv_finish(self) -> TimeVariable:
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
    def envelope_waveform_i(self) -> np.ndarray:
        return  self._shape.envelope_waveform_i

    @property
    def envelope_waveform_q(self) -> np.ndarray:
        return  self._shape.envelope_waveform_q

    @property
    def modulated_waveform_i(self) -> np.ndarray:
        return  self._shape.modulated_waveform_i

    @property
    def modulated_waveform_q(self) -> np.ndarray:
        return  self._shape.modulated_waveform_q

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
        if not isinstance(other, Pulse):
            raise TypeError(f'Expected int; got {type(other).__name__}')
        return PulseCollection(self, other)
            
    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 0:
            raise TypeError(f'argument n should be >=0, got {n}')
        return PulseCollection(* ([self.shallow_copy()] * n))

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

        time = self.start + np.arange(int(self.duration / 1e9 * PulseShape.SAMPLING_RATE)) / PulseShape.SAMPLING_RATE * 1e9
        fig = plt.figure(figsize=(14, 5), dpi=120)
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(time, self.envelope_waveform_i, label='envelope i', c='C0', linestyle='dashed')
        ax1.plot(time, self.envelope_waveform_q, label='envelope q', c='C1', linestyle='dashed')
        ax1.plot(time, self.modulated_waveform_i, label='modulated i', c='C0')
        ax1.plot(time, self.modulated_waveform_q, label='modulated q', c='C1')
        ax1.plot(time, -self.envelope_waveform_i, c='silver', linestyle='dashed')
        ax1.set_xlabel('Time [ns]')
        ax1.set_ylabel('Amplitude')

        ax1.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax1.axis([self.start, self.finish, -1, 1])
        ax1.legend()

        ax2 = plt.subplot(gs[1])
        ax2.plot(self.modulated_waveform_i, self.modulated_waveform_q, label='modulated', c='C3')
        ax2.plot(self.envelope_waveform_i, self.envelope_waveform_q, label='envelope', c='C2')
        ax2.plot(self.modulated_waveform_i[0], self.modulated_waveform_q[0], marker="o", markersize=5, label='start', c='lightcoral')
        ax2.plot(self.modulated_waveform_i[-1], self.modulated_waveform_q[-1], marker="o", markersize=5, label='finish', c='darkred')

        ax2.plot(np.cos(time * 2 * np.pi / self.duration), np.sin(time * 2 * np.pi / self.duration), c='silver', linestyle='dashed')

        ax2.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        plt.suptitle(self.serial)
        plt.show()
        return fig


class ReadoutPulse(Pulse):
    """Describes a readout pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """
    def __init__(self, start:int | TimeVariable, duration:int | TimeVariable, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, qubit: int | str = 0):
        super().__init__(start, duration, amplitude, frequency, relative_phase, shape, channel, type =  PulseType.READOUT, qubit = qubit)

    @property
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"


class DrivePulse(Pulse):
    """Describes a qubit drive pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """
    def __init__(self, start:int | TimeVariable, duration:int | TimeVariable, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
                       channel: int | str, qubit: int | str = 0):
        super().__init__(start, duration, amplitude, frequency, relative_phase, shape, channel, type =  PulseType.DRIVE, qubit = qubit)

    @property
    def serial(self):
        return f"DrivePulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"


class FluxPulse(Pulse):
    """Describes a qubit drive pulse.

    See :class:`qibolab.pulses.Pulse` for argument desciption.
    """
    def __init__(self, start:int | TimeVariable, duration:int | TimeVariable, amplitude:float, frequency:int, relative_phase:float, shape: PulseShape | str,
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
    def serial(self):
        return f"ReadoutPulse({self.start}, {self.duration}, {format(self.amplitude, '.6f').rstrip('0').rstrip('.')}, {format(self.frequency, '_')}, {format(self.relative_phase, '.6f').rstrip('0').rstrip('.')}, {self.shape}, {self.channel})"

    @property
    def window_duration(self):
        return self._window_finish - self._window_start

    @property
    def envelope_waveform_i(self):
        return  self._shape.envelope_waveform_i[self._window_start - self.start : self._window_finish - self.start]

    @property
    def envelope_waveform_q(self):
        return  self._shape.envelope_waveform_q[self._window_start - self.start : self._window_finish - self.start]

    @property
    def modulated_waveform_i(self):
        return  self._shape.modulated_waveform_i[self._window_start - self.start : self._window_finish - self.start]

    @property
    def modulated_waveform_q(self):
        return  self._shape.modulated_waveform_q[self._window_start - self.start : self._window_finish - self.start]

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import numpy as np

        time = self.window_start + np.arange(int(self.window_duration / 1e9 * PulseShape.SAMPLING_RATE)) / PulseShape.SAMPLING_RATE * 1e9
        
        fig = plt.figure(figsize=(14, 5), dpi=120)
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(time, self.envelope_waveform_i, label='envelope i', c='C0', linestyle='dashed')
        ax1.plot(time, self.envelope_waveform_q, label='envelope q', c='C1', linestyle='dashed')
        ax1.plot(time, self.modulated_waveform_i, label='modulated i', c='C0')
        ax1.plot(time, self.modulated_waveform_q, label='modulated q', c='C1')
        ax1.plot(time, -self.envelope_waveform_i, c='silver', linestyle='dashed')
        ax1.set_xlabel('Time [ns]')
        ax1.set_ylabel('Amplitude')

        ax1.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax1.axis([self.window_start, self._window_finish, -1, 1])
        ax1.legend()

        ax2 = plt.subplot(gs[1])
        ax2.plot(self.modulated_waveform_i, self.modulated_waveform_q, label='modulated', c='C3')
        ax2.plot(self.envelope_waveform_i, self.envelope_waveform_q, label='envelope', c='C2')
        ax2.plot(np.cos(time * 2 * np.pi / self.window_duration), np.sin(time * 2 * np.pi / self.window_duration), c='silver', linestyle='dashed')

        ax2.grid(b=True, which='both', axis='both', color='#888888', linestyle='-')
        ax2.legend()
        # ax2.axis([ -1, 1, -1, 1])
        ax2.axis("equal")
        plt.show()
        return fig


class PulseCollection(): 
    def __init__(self, *pulses):
        self.pulses: list[Pulse] = []
        self.append(*pulses)

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

    def __reversed__(self):
        for pulse in self.pulses[::-1]:
            yield pulse
    
    def __repr__(self):
        return '\n'.join(f'{pulse.serial}' for pulse in self.pulses)

    def __add__(self, other):
        if isinstance(other, PulseCollection):
            return PulseCollection(* self.pulses, * other.pulses)
        elif isinstance(other, Pulse):
            return PulseCollection(* self.pulses, other)
        else:
            raise TypeError(f'Expected PulseCollection or Pulse; got {type(other).__name__}')

    def __radd__(self, other):
        if isinstance(other, PulseCollection):
            return PulseCollection(* other.pulses, * self.pulses)
        elif isinstance(other, Pulse):
            return PulseCollection(other, * self.pulses)
        else:
            raise TypeError(f'Expected PulseCollection or Pulse; got {type(other).__name__}')
    
    def __iadd__(self, other):
        if isinstance(other, PulseCollection):
            self.append(* other.pulses)
        elif isinstance(other, Pulse):
            self.append(other)
        else:
            raise TypeError(f'Expected PulseCollection or Pulse; got {type(other).__name__}')
        return self
        
    def __mul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 0:
            raise TypeError(f'argument n should be >=0, got {n}')
        return PulseCollection(* (self.pulses * n))
        
    def __rmul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 0:
            raise TypeError(f'argument n should be >=0, got {n}')
        return PulseCollection(* (self.pulses * n))

    def __imul__(self, n):
        if not isinstance(n, int):
            raise TypeError(f'Expected int; got {type(n).__name__}')
        elif n < 1:
            raise TypeError(f'argument n should be >=1, got {n}')
        original_set = self.shallow_copy()
        for x in range(n - 1):
            self.append(* original_set.pulses)
        return self

    @property
    def count(self):
        return len(self.pulses)

    def append(self, *pulses):
        for pulse in pulses:
            self.pulses.append(pulse)

    def index(self, pulse):
        return self.pulses.index(pulse)
        
    def insert(self, index, pulse):
        self.pulses.insert(index, pulse)

    def pop(self, index = -1):
        return self.pulses.pop(index)

    def remove(self, pulse):
        while pulse in self.pulses:
            self.pulses.remove(pulse)

    def reverse(self):
        self.pulses.reverse()
        return self

    def sort(self):
        self.pulses.sort(key=lambda item: (item.channel, item.start))
        return self

    def clear(self):
        self.pulses.clear()

    def shallow_copy(self):
        return PulseCollection(* self.pulses)

    def deep_copy(self):
        return PulseCollection(* [pulse.deep_copy() for pulse in self.pulses])

    @property
    def ro_pulses(self):
        new_pc = PulseCollection()
        for pulse in self.pulses:
            if pulse.type == PulseType.READOUT:
                new_pc.append(pulse)
        return new_pc

    @property
    def qd_pulses(self):
        new_pc = PulseCollection()
        for pulse in self.pulses:
            if pulse.type == PulseType.DRIVE:
                new_pc.append(pulse)
        return new_pc

    @property
    def qf_pulses(self):
        new_pc = PulseCollection()
        for pulse in self.pulses:
            if pulse.type == PulseType.FLUX:
                new_pc.append(pulse)
        return new_pc


    def get_channel_pulses(self, channel):
        new_pc = PulseCollection()
        for pulse in self.pulses:
            if pulse.channel == channel:
                new_pc.append(pulse)
        return new_pc

    @property
    def is_empty(self):
        return len(self.pulses) == 0

    @property
    def finish(self) -> int:
        t: int = 0
        for pulse in self.pulses:
            if pulse.finish > t:
                t = pulse.finish
        return t

    @property
    def start(self) -> int:
        t = self.finish
        for pulse in self.pulses:
            if pulse.start < t:
                t = pulse.start
        return t

    @property
    def channels(self) -> list:
        channels = []
        for pulse in self.pulses:
            if not pulse.channel in channels:
                channels.append(pulse.channel)
        channels.sort()
        return channels

    def get_pulse_overlaps(self): # -> dict((int,int): PulseCollection):
        times = []
        for pulse in self.pulses:
            if not pulse.start in times:
                times.append(pulse.start)
            if not pulse.finish in times:
                times.append(pulse.finish)
        times.sort()

        overlaps = {}
        for n in range(len(times)-1):
            overlaps[(times[n], times[n+1])] = PulseCollection()
            for pulse in self.pulses:
                if (pulse.start <= times[n]) & (pulse.finish >= times[n+1]):
                    overlaps[(times[n], times[n+1])] += pulse
        return overlaps

    @property
    def pulses_overlap(self) -> bool:
        overlap = False
        for pc in self.get_pulse_overlaps().values():
            if pc.count > 1:
                overlap = True
        return overlap

    def plot(self):
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
                else:
                    time = pulse.start + np.arange(pulse.duration)
                ax.plot(time, pulse.modulated_waveform_i, c=f'C{str(n)}')
                ax.plot(time, pulse.envelope_waveform_i, c=f'C{str(n)}')
                ax.plot(time, -pulse.envelope_waveform_i, c=f'C{str(n)}')
                # TODO: if they overlap use different shades
                ax.axhline(0, c='dimgrey')
                ax.set_ylabel(f'channel {channel}')
                for vl in vertical_lines:
                    ax.axvline(vl, c='slategrey', linestyle = '--')
                ax.axis([0, self.finish, -1, 1])
                ax.grid(b=True, which='both', axis='both', color='#CCCCCC', linestyle='-')
        plt.show()
        return fig


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
                                relative_phase=0,
                                shape=Gaussian(5),
                                channel=1,
                                type='qd')
                pulse2 = Pulse( start=70,
                                duration=2000,
                                amplitude=0.5,
                                frequency=20_000_000.0,
                                relative_phase=0,
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
