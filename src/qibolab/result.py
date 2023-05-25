from dataclasses import dataclass
from functools import cached_property, lru_cache
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

    def __add__(self, data):
        return self.__class__(np.append(self.voltage, data.voltage))

    @property
    def voltage_i(self):
        """Signal component i in volts."""
        return self.voltage.real

    @property
    def voltage_q(self):
        """Signal component q in volts."""
        return self.voltage.imag

    @cached_property
    def magnitude(self):
        """Signal magnitude in volts."""
        return np.sqrt(self.voltage_i**2 + self.voltage_q**2)

    @cached_property
    def phase(self):
        """Signal phase in radians."""
        return np.angle(self.voltage_i + 1.0j * self.voltage_q)

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

    def __add__(self, data):
        new_res = super().__add__(data)
        new_res.std = np.append(self.std, data.std)
        return new_res


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


class SampleResults:
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.SINGLESHOT
    """

    def __init__(self, data: np.ndarray):
        self.samples: npt.NDArray[np.uint32] = data

    def __add__(self, data):
        return self.__class__(np.append(self.samples, data.samples))

    @lru_cache
    def probability(self, state=0):
        """Returns the statistical frequency of the specified state (0 or 1)."""
        return abs(1 - state - np.mean(self.samples, axis=0))

    @property
    def serialize(self):
        """Serialize as a dictionary."""
        serialized_dict = {
            "0": self.probability(0).flatten(),
        }
        return serialized_dict

    @property
    def average(self):
        """Perform samples average"""
        average = self.probability(1)
        std = np.std(self.samples, axis=0, ddof=1) / np.sqrt(self.samples.shape[0])
        return AveragedSampleResults(average, self.samples, std=std)


class AveragedSampleResults(SampleResults):
    """
    Data structure to deal with the output of
    :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.CYCLIC
    or the averages of ``SampleResults``
    """

    def __init__(
        self, statistical_frequency: np.ndarray, samples: np.ndarray = np.array([]), std: np.ndarray = np.array([])
    ):
        super().__init__(samples)
        self.statistical_frequency: npt.NDArray[np.float64] = statistical_frequency
        self.std: Optional[npt.NDArray[np.float64]] = std

    def __add__(self, data):
        new_res = super().__add__(data)
        new_res.statistical_frequency = np.append(self.statistical_frequency, data.statistical_frequency)
        new_res.std = np.append(self.std, data.std)
        return new_res


ExecRes = np.dtype([("i", np.float64), ("q", np.float64)])


@dataclass
class ExecutionResults:
    """Data structure to deal with the output of :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`"""

    array: npt.NDArray[ExecRes]
    shots: Optional[npt.NDArray[np.uint32]] = None

    @classmethod
    def from_components(cls, is_, qs_, shots=None):
        ar = np.empty(is_.shape, dtype=ExecRes)
        ar["i"] = is_
        ar["q"] = qs_
        ar = np.rec.array(ar)
        return cls(ar, shots)

    @property
    def i(self):
        return self.array.i

    @property
    def q(self):
        return self.array.q

    def __add__(self, data):
        assert len(data.i.shape) == len(data.q.shape)
        if data.shots is not None:
            assert len(data.i.shape) == len(data.shots.shape)
        # concatenate on first dimension; if a scalar is passed, just append it
        axis = 0 if len(data.i.shape) > 0 else None
        i = np.append(self.i, data.i, axis=axis)
        q = np.append(self.q, data.q, axis=axis)
        if data.shots is not None:
            shots = np.append(self.shots, data.shots, axis=axis)
        else:
            shots = None

        new_execution_results = self.__class__.from_components(i, q, shots)

        return new_execution_results

    @cached_property
    def measurement(self):
        """Resonator signal voltage mesurement (MSR) in volts."""
        return np.sqrt(self.i**2 + self.q**2)

    @cached_property
    def phase(self):
        """Computes phase value."""
        phase = np.angle(self.i + 1.0j * self.q)
        return phase
        # return signal.detrend(np.unwrap(phase))

    @cached_property
    def ground_state_probability(self):
        """Computes ground state probability"""
        return 1 - np.mean(self.shots)

    def raw_probability(self, state=1):
        """Serialize probabilities in dict.
        Args:
            state (int): if 0 stores the probabilities of finding
                        the ground state. If 1 stores the
                        probabilities of finding the excited state.
        """
        if state == 1:
            return {"probability": 1 - self.ground_state_probability}
        elif state == 0:
            return {"probability": self.ground_state_probability}

    @property
    def average(self):
        """Perform average over i and q"""
        return AveragedResults.from_components(np.mean(self.i), np.mean(self.q))

    @property
    def raw(self):
        """Serialize output in dict."""

        return {
            "MSR[V]": self.measurement,
            "i[V]": self.i,
            "q[V]": self.q,
            "phase[rad]": self.phase,
        }

    def __len__(self):
        return len(self.i)


class AveragedResults(ExecutionResults):
    """Data structure containing averages of ``ExecutionResults``."""
