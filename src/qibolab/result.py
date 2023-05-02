from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

iq_record_dtype = np.dtype([("i", np.float64), ("q", np.float64)])


class ExecutionResults:
    """Data structure to deal with the output of :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`"""

    def __init__(self, i:np.ndarray | list, q:np.ndarray | list, state:np.ndarray | list = None):
        if not isinstance(i, np.ndarray):
            i = np.array(i)
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        if i.shape != q.shape:
            raise ValueError("is_ and qs_ must have the same size")

        self.voltage: npt.NDArray[iq_record_dtype] = np.recarray(i.shape, dtype=iq_record_dtype)
        self.voltage["i"] = i
        self.voltage["q"] = q

        if state and not isinstance(state, np.ndarray):
            state = np.array(state)
        self.state: Optional[npt.NDArray[np.uint32]] = state

    @property
    def i(self):
        return self.voltage.i

    @property
    def q(self):
        return self.voltage.q

    @cached_property
    def magnitude(self):
        """Signal magnitude in volts."""
        return np.sqrt(self.i**2 + self.q**2)

    @cached_property
    def phase(self):
        """Signal phase in radiants."""
        phase = np.angle(self.i + 1.0j * self.q)
        return phase

    @cached_property
    def state_0_probability(self):
        return self.probability(0) if self.state else None
            
    @cached_property
    def state_1_probability(self):
        return self.probability(1) if self.state else None

    def probability(self, state=0):
        """Returns the statistical frequency of the specified state (0 or 1)."""
        return np.count_nonzero(self.state == state)/len(self.state)

    def __add__(self, data):# __add__(self, data:ExecutionResults) -> ExecutionResults
        axis = 0 if len(data.i.shape) > 0 else None
        i = np.append(self.i, data.i, axis=axis)
        q = np.append(self.q, data.q, axis=axis)
        if data.state:
            state = np.append(self.state, data.state, axis=axis)
        else:
            state = None
        return ExecutionResults(i, q, state)

    def __len__(self):
        return len(self.i)

    def serialize(self):
        """Serialize as a dictionary."""
        serialized_dict = {
            "magnitude[V]": self.magnitude,
            "i[V]": self.i,
            "q[V]": self.q,
            "phase[rad]": self.phase,
        }
        if self.state:
            serialized_dict["state"] = self.state_1_probability
        return serialized_dict

    @property
    def average(self):
        """Perform average over i and q"""
        return AveragedResults(np.mean(self.i), np.mean(self.q))
    

class AveragedResults(ExecutionResults):
    """Data structure containing averages of ``ExecutionResults``."""
