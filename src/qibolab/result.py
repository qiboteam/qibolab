from functools import cached_property, lru_cache
from typing import Optional

import numpy as np
import numpy.typing as npt


class IntegratedResults:
    """Data structure to deal with the output of :func:`qibolab.platforms.abstr
    act.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.INTEGRATION and
    AveragingMode.SINGLESHOT
    """

    def __init__(self, data: np.ndarray):
        self.voltage: npt.NDArray[np.complex128] = data

    def __add__(self, data):
        """Concatenates two results for sweeping variables."""
        return self.__class__(np.column_stack([self.voltage, data.voltage]))

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
        """Perform average over i and q."""
        average_data = np.mean(self.voltage, axis=0)
        std_data = np.std(self.voltage, axis=0, ddof=1) / np.sqrt(self.voltage.shape[0])
        return AveragedIntegratedResults(average_data, std_data)


class AveragedIntegratedResults(IntegratedResults):
    """Data structure to deal with the output of :func:`qibolab.platforms.abstr
    act.AbstractPlatform.execute_pulse_sequence`
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
    """Data structure to deal with the output of :func:`qibolab.platforms.abstr
    act.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.RAW and AveragingMode.SINGLESHOT may
    also be used to store the integration weights ?
    """


class AveragedRawWaveformResults(AveragedIntegratedResults):
    """Data structure to deal with the output of :func:`qibolab.platforms.abstr
    act.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.RAW and AveragingMode.CYCLIC
    or the averages of ``RawWaveformResults``
    """


class SampleResults:
    """Data structure to deal with the output of :func:`qibolab.platforms.abstr
    act.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and
    AveragingMode.SINGLESHOT
    """

    def __init__(self, data: np.ndarray):
        self.samples: npt.NDArray[np.uint32] = np.array(data).astype(np.uint32)

    def __add__(self, data):
        return self.__class__(np.append(self.samples, data.samples))

    @lru_cache
    def probability(self, state=0):
        """Returns the statistical frequency of the specified state (0 or
        1)."""
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
        """Perform samples average."""
        average = self.probability(1)
        std = np.std(self.samples, axis=0, ddof=1) / np.sqrt(self.samples.shape[0])
        return AveragedSampleResults(average, self.samples, std=std)


class AveragedSampleResults(SampleResults):
    """Data structure to deal with the output of :func:`qibolab.platforms.abstr
    act.AbstractPlatform.execute_pulse_sequence`
    :func:`qibolab.platforms.abstract.AbstractPlatform.sweep`

    Associated with AcquisitionType.DISCRIMINATION and AveragingMode.CYCLIC
    or the averages of ``SampleResults``
    """

    def __init__(
        self,
        statistical_frequency: np.ndarray,
        samples: np.ndarray = np.array([]),
        std: np.ndarray = np.array([]),
    ):
        super().__init__(samples)
        self.statistical_frequency: npt.NDArray[np.float64] = statistical_frequency
        self.std: Optional[npt.NDArray[np.float64]] = std

    def __add__(self, data):
        new_res = super().__add__(data)
        new_res.statistical_frequency = np.append(
            self.statistical_frequency, data.statistical_frequency
        )
        new_res.std = np.append(self.std, data.std)
        return new_res

    @lru_cache
    def probability(self, state=0):
        """Returns the statistical frequency of the specified state (0 or
        1)."""
        return abs(1 - state - self.statistical_frequency)
