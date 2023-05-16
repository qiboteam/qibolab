from dataclasses import InitVar, dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt


class IntegratedResults:
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.INTEGRATION and AveragingMode.SINGLESHOT
    """

    def __init__(self, data: np.ndarray):
        self.voltage: npt.NDArray[np.complex128] = data

    @cached_property
    def voltage_i(self):
        """Signal magnitude in volts."""
        return self.voltage.real

    @cached_property
    def voltage_q(self):
        """Signal magnitude in volts."""
        return self.voltage.imag

    @cached_property
    def magnitude(self):
        """Signal magnitude in volts."""
        return np.sqrt(self.voltage_i**2 + self.voltage_q**2)

    @cached_property
    def phase(self):
        """Signal phase in radians."""
        return np.angle(self.voltage_i + 1.0j * self.voltage_q)

    # #TODO: Check is adding as wanted, we may need some imput on what data is being added
    # def __add__(self, data, axis):  # __add__(self, data:IntegratedResults) -> IntegratedResults
    #     axis = 0
    #     voltage = np.append(self.voltage, data.voltage, axis=axis)
    #     return IntegratedResults(voltage)

    @property
    def serialize(self):
        """Serialize as a dictionary."""
        serialized_dict = {
            "MSR[V]": self.magnitude.flatten(),
            "i[V]": self.voltage_i.flatten(),
            "q[V]": self.voltage_q.flatten(),
            "phase[rad]": self.phase.flatten(),
        }
        return serialized_dict

    @property
    def average(self):
        """Perform average over i and q"""
        average_data = np.mean(self.voltage, axis=0)
        std_data = np.std(self.voltage, axis=0, ddof=1) / np.sqrt(self.voltage.shape[0])
        return AveragedIntegratedResults(average_data, std_data)


class AveragedIntegratedResults(IntegratedResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.INTEGRATION and AveragingMode.CYCLIC
    or the averages of ``IntegratedResults``
    """

    def __init__(self, data: np.ndarray, std: np.ndarray = np.array([])):
        super().__init__(data)
        self.std: Optional[npt.NDArray[np.float64]] = std


class RawWaveformResults(IntegratedResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.RAW and AveragingMode.SINGLESHOT
    may also be used to store the integration weights ?
    """


class AveragedRawWaveformResults(AveragedIntegratedResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.RAW and AveragingMode.CYCLIC
    or the averages of ``RawWaveformResults``
    """


# FIXME: If probabilities are out of range the error is displeyed weirdly
class StateResults:
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.SINGLESHOT
    """

    def __init__(self, data: np.ndarray):
        self.states: npt.NDArray[np.uint32] = data

    def probability(self, state=0):
        """Returns the statistical frequency of the specified state (0 or 1)."""
        return abs(1 - state - np.mean(self.states, axis=0))

    @cached_property
    def state_0_probability(self):
        """Returns the 0 state statistical frequency."""
        return self.probability(0)

    @cached_property
    def state_1_probability(self):
        """Returns the 1 state statistical frequency."""
        return self.probability(1)

    # #TODO: Check is adding as wanted, we may need some imput on what data is being added
    # def __add__(self, data):  # __add__(self, data:StateResults) -> StateResults
    #     states = np.append(self.states, data.states, axis=0)
    #     return StateResults(states)

    def serialize(self):
        """Serialize as a dictionary."""
        serialized_dict = {
            "state_0": self.state_0_probability,
        }
        return serialized_dict

    @property
    def average(self):
        """Perform states average"""
        average = self.state_1_probability
        std = np.std(self.states, axis=0, ddof=1) / np.sqrt(self.states.shape[0])
        return AveragedStateResults(average, std=std)


# FIXME: Here I take the states from StateResult that are typed to be ints but those are not what would you do ?
class AveragedStateResults(StateResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.CYCLIC
    or the averages of ``StateResults``
    """

    def __init__(self, states: np.ndarray, std: np.ndarray = np.array([])):
        super().__init__(states)
        self.std: Optional[npt.NDArray[np.float64]] = std
