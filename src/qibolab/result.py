from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

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
