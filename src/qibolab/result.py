from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import signal


@dataclass
class ExecutionResults:
    """Data structure to deal with the output of execute_pulse_sequence"""

    i: npt.NDArray[np.float64]
    q: npt.NDArray[np.float64]
    shots: Optional[npt.NDArray[np.uint32]] = None

    @classmethod
    def from_components(cls, is_, qs_, shots=None):
        return cls(is_, qs_, shots)

    @property
    def in_progress(self):
        """Placeholder for when we implement live fetching of data from instruments."""
        return False

    @cached_property
    def measurement(self):
        """Resonator signal voltage mesurement (MSR) in volts."""
        return np.sqrt(self.i**2 + self.q**2)

    @cached_property
    def phase(self):
        """Computes phase value."""
        phase = np.angle(self.i + 1.0j * self.q)
        return signal.detrend(np.unwrap(phase))

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
                "MSR[V]": self.measurement.mean(),
                "i[V]": self.i.mean(),
                "q[V]": self.q.mean(),
                "phase[rad]": self.phase.mean(),
            }
        else:
            return {
                "MSR[V]": self.measurement.ravel(),
                "i[V]": self.i.ravel(),
                "q[V]": self.q.ravel(),
                "phase[rad]": self.phase.ravel(),
            }
