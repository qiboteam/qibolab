from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

iq_typing = np.dtype([("i", np.float64), ("q", np.float64)])


@dataclass
class IntegratedResults:
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.INTEGRATION and AveragingMode.SINGLESHOT
    """

    def __init__(self, i: np.ndarray, q: np.ndarray, nshots=None):
        self.voltage: npt.NDArray[iq_typing] = (
            np.recarray((i.shape[0] // nshots, nshots), dtype=iq_typing)
            if nshots
            else np.recarray(i.shape, dtype=iq_typing)
        )
        self.voltage["i"] = i.reshape(i.shape[0] // nshots, nshots) if nshots else i
        try:
            self.voltage["q"] = q.reshape(q.shape[0] // nshots, nshots) if nshots else q
        except ValueError:
            raise ValueError(f"Given i has length {len(i)} which is different than q length {len(q)}.")

    @cached_property
    def magnitude(self):
        """Signal magnitude in volts."""
        return np.sqrt(self.voltage.i**2 + self.voltage.q**2)

    @cached_property
    def phase(self):
        """Signal phase in radians."""
        return np.angle(self.voltage.i + 1.0j * self.voltage.q)

    # We are asumming results from the same experiment so same number of nshots
    def __add__(self, data):  # __add__(self, data:IntegratedResults) -> IntegratedResults
        axis = 0
        i = np.append(self.voltage.i, data.voltage.i, axis=axis)
        q = np.append(self.voltage.q, data.voltage.q, axis=axis)
        return IntegratedResults(i, q)

    def serialize(self):
        """Serialize as a dictionary."""
        serialized_dict = {
            "magnitude[V]": self.magnitude,
            "i[V]": self.voltage.i,
            "q[V]": self.voltage.q,
            "phase[rad]": self.phase,
        }
        return serialized_dict

    @property
    def average(self):
        """Perform average over i and q"""
        average_i, average_q = np.array([]), np.array([])
        std_i, std_q = np.array([]), np.array([])
        for is_, qs_ in zip(self.voltage.i, self.voltage.q):
            average_i, average_q = np.append(average_i, np.mean(is_)), np.append(average_q, np.mean(qs_))
            std_i, std_q = np.append(std_i, np.std(is_)), np.append(std_q, np.std(qs_))
        return AveragedIntegratedResults(average_i, average_q, std_i=std_i, std_q=std_q)


# FIXME: Here I take the states from IntegratedResult that are typed to be ints but those are not what would you do ?
class AveragedIntegratedResults(IntegratedResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.INTEGRATION and AveragingMode.CYCLIC
    or the averages of ``IntegratedResults``
    """

    def __init__(self, i: np.ndarray, q: np.ndarray, nshots=None, std_i=None, std_q=None):
        IntegratedResults.__init__(self, i, q, nshots)
        self.std: Optional[npt.NDArray[np.float64]] = np.recarray(i.shape, dtype=iq_typing)
        self.std["i"] = std_i
        self.std["q"] = std_q


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
@dataclass
class StateResults:
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.SINGLESHOT
    """

    def __init__(self, states: np.ndarray = np.array([]), nshots=None):
        self.states: Optional[npt.NDArray[np.uint32]] = (
            states.reshape(states.shape[0] // nshots, nshots) if nshots else states
        )

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        if not np.all((values >= 0) & (values <= 1)):
            raise ValueError("Probability wrong")
        self._states = values

    def probability(self, state=0):
        """Returns the statistical frequency of the specified state (0 or 1)."""
        probability = np.array([])
        state = 1 - state
        for st in self.states:
            probability = np.append(probability, abs(state - np.mean(st)))
        return probability

    @cached_property
    def state_0_probability(self):
        """Returns the 0 state statistical frequency."""
        return self.probability(0)

    @cached_property
    def state_1_probability(self):
        """Returns the 1 state statistical frequency."""
        return self.probability(1)

    # We are asumming results from the same experiment so same number of nshots
    def __add__(self, data):  # __add__(self, data:StateResults) -> StateResults
        states = np.append(self.states, data.states, axis=0)
        return StateResults(states)

    def serialize(self):
        """Serialize as a dictionary."""
        serialized_dict = {
            "state_0": self.state_0_probability,
        }
        return serialized_dict

    @property
    def average(self):
        """Perform states average"""
        average = np.array([])
        std = np.array([])
        for st in self.states:
            average = np.append(average, np.mean(st))
            std = np.append(std, np.std(st))
        return AveragedStateResults(average, std=std)


class AveragedStateResults(StateResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.CYCLIC
    or the averages of ``StateResults``
    """

    def __init__(self, states: np.ndarray = np.array([]), nshots=None, std=None):
        StateResults.__init__(self, states, nshots)
        self.std: Optional[npt.NDArray[np.float64]] = std
