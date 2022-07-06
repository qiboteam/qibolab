"""Waveform class."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Waveforms:
    """Waveform class that containg the I and Q modulated waveforms."""

    i: np.ndarray = np.array([])
    q: np.ndarray = np.array([])  # pylint: disable=invalid-name

    def add(self, imod: np.ndarray, qmod: np.ndarray):
        """Add i and q arrays to the waveforms.

        Args:
            imod (np.ndarray): I modulated waveform to add.
            qmod (np.ndarray): Q modulated waveform to add.
        """
        self.i = np.append(self.i, imod)
        self.q = np.append(self.q, qmod)

    @property
    def values(self):
        """Return the waveform i and q values.

        Returns:
            np.ndarray: Array containing the i and q waveform values.
        """
        return np.array([self.i, self.q])

    def __add__(self, other):
        """Add two Waveform objects."""
        if not isinstance(other, Waveforms):
            raise NotImplementedError
        self.i = np.append(self.i, other.i)
        self.q = np.append(self.q, other.q)
        return self

    def __len__(self):
        """Length of the object."""
        if len(self.i) != len(self.q):
            raise ValueError("Both I and Q waveforms must have the same length.")
        return len(self.i)
