from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

ExecRes = np.dtype([("i", np.float64), ("q", np.float64)])


@dataclass
class AveragedResults:
    """Data structure containing averages of ``ExecutionResults``."""

    i: npt.NDArray[np.float64]
    q: npt.NDArray[np.float64]

    def __add__(self, data):
        i = np.append(self.i, data.i)
        q = np.append(self.q, data.q)

        new_execution_results = self.__class__(i, q)

        return new_execution_results

    def to_dict(self):
        """Serialize output in dict"""

        return {
            "MSR[V]": np.sqrt(self.i**2 + self.q**2),
            "i[V]": self.i,
            "q[V]": self.q,
            "phase[rad]": np.angle(self.i + 1.0j * self.q),
        }

    def __len__(self):
        return len(self.i)


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
        i = np.append(self.i, data.i, axis=0)
        q = np.append(self.q, data.q, axis=0)

        new_execution_results = self.__class__.from_components(i, q)

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

    def to_dict_probability(self, state=1):
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

    def compute_average(self):
        """Perform average over i and q"""
        return AveragedResults(self.array.i.mean(), self.array.q.mean())

    def to_dict(self, average=True):
        """Serialize output in dict.
        Args:
            average (bool): If `True` returns a dictionary of the form
                            {'MSR[V]' : v, 'i[V]' : i, 'q[V]' : q, 'phase[rad]' : phase}.
                            Where each value is averaged over the number shots. If `False`
                            all the values for each shot are saved.
        """
        results = self.compute_average() if average else self

        return {
            "MSR[V]": np.sqrt(results.i**2 + results.q**2),
            "i[V]": results.i,
            "q[V]": results.q,
            "phase[rad]": np.angle(results.i + 1.0j * results.q),
        }

    def __len__(self):
        return len(self.i)
