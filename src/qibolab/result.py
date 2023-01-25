from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

ExecRes = np.dtype([("i", np.float64), ("q", np.float64), ("shots", np.uint32)])


@dataclass
class ExecutionResults:
    """Data structure to deal with the output of execute_pulse_sequence"""

    array: npt.NDArray[ExecRes]

    @classmethod
    def from_components(cls, is_, qs, shots):
        ar = np.empty(len(is_), dtype=ExecRes)
        ar["i"] = is_
        ar["q"] = qs
        ar["shots"] = shots
        ar = np.rec.array(ar)
        return cls(ar)

    @property
    def i(self):
        return self.array.i

    @property
    def q(self):
        return self.array.q

    @property
    def shots(self):
        return self.array.shots

    @cached_property
    def msr(self):
        """Computes msr value."""
        return np.sqrt(self.array.i**2 + self.array.q**2)

    @cached_property
    def phase(self):
        """Computes phase value."""
        return np.angle(self.array.i + 1.0j * self.array.q)

    @cached_property
    def ground_state_probability(self):
        """Computes ground state probability"""
        return 1 - self.array.shots.mean()

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

    def to_dict(self, average=True):
        """Serialize output in dict.
        Args:
            average (bool): If `True` returns a dictionary of the form
                            {'MSR[V]' : v, 'i[V]' : i, 'q[V]' : q, 'phase[rad]' : phase}.
                            Where each value is averaged over the number shots. If `False`
                            all the values for each shot are saved.
        """
        if average:
            return {
                "MSR[V]": self.msr.mean(),
                "i[V]": self.i.mean(),
                "q[V]": self.q.mean(),
                "phase[rad]": self.phase.mean(),
            }
        else:
            return {
                "MSR[V]": self.msr.ravel(),
                "i[V]": self.i.ravel(),
                "q[V]": self.q.ravel(),
                "phase[rad]": self.phase.ravel(),
            }
