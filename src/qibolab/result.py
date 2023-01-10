import numpy as np
from scipy import signal


class ExecutionResult:
    """Container returned by :meth:`qibolab.platforms.platform.Platform.execute_pulse_sequence`.

    Args:
        i_values (np.ndarray): Measured I values obtained from the experiment.
        q_values (np.ndarray): Measured Q values obtained from the experiment.
    """

    # TODO: Distinguish cases where we have single shots vs averaged values
    # TODO: Implement methods to return classified shots

    def __init__(self, i_values, q_values):
        self.I = i_values
        self.Q = q_values
        self.in_progress = False

    @property
    def MSR(self):
        return np.sqrt(self.I**2 + self.Q**2)

    @property
    def phase(self):
        phase = np.angle(self.I + 1j * self.Q)
        return signal.detrend(np.unwrap(phase))
