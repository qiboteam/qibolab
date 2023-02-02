from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import signal


@dataclass
class ExecutionResults:
    """Data structure to deal with the output of execute_pulse_sequence"""

    _i: npt.NDArray[np.float64]
    _q: npt.NDArray[np.float64]
    shots: Optional[npt.NDArray[np.uint32]] = None

    @classmethod
    def from_components(cls, is_, qs_, shots=None):
        return cls(is_, qs_, shots)

    @property
    def in_progress(self):
        """Placeholder for when we implement live fetching of data from instruments."""
        return False

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        self._i = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        self._q = value

    def __add__(self, data):
        i = np.append(self.i, data.i)
        q = np.append(self.q, data.q)

        new_execution_results = self.__class__.from_components(i, q)

        return new_execution_results

    @cached_property
    def msr(self):
        """Computes msr value."""
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
        self.i = self.i.mean()
        self.q = self.q.mean()

    def to_dict(self, average=False):
        """Serialize output in dict.
        Args:
            average (bool): If `True` returns a dictionary of the form
                            {'MSR[V]' : v, 'i[V]' : i, 'q[V]' : q, 'phase[rad]' : phase}.
                            Where each value is averaged over the number shots. If `False`
                            all the values for each shot are saved.
        """
        if average:
            self.compute_average()

        return {
            "MSR[V]": self.msr.ravel(),
            "i[V]": self.i.ravel(),
            "q[V]": self.q.ravel(),
            "phase[rad]": self.phase.ravel(),
        }
