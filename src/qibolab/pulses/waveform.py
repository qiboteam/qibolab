"""Waveform class."""
import numpy as np


class Waveform:
    """A class to save pulse waveforms.

    A waveform is a list of samples, or discrete data points, used by the digital to analogue converters (DACs)
    to synthesise pulses.

    Attributes:
        data (np.ndarray): a numpy array containing the samples.
    """

    DECIMALS = 5

    def __init__(self, data):
        """Initialise the waveform with a of samples."""
        self.data: np.ndarray = np.array(data)

    def __len__(self):
        """Return the length of the waveform, the number of samples."""
        return len(self.data)

    def __hash__(self):
        """Hash the underlying data.

        .. todo::

            In order to make this reliable, we should set the data as immutable. This we
            could by making both the class frozen and the contained array readonly
            https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html#numpy.ndarray.flags
        """
        return hash(self.data.tobytes())

    def __eq__(self, other):
        """Compare two waveforms.

        Two waveforms are considered equal if their samples, rounded to
        `Waveform.DECIMALS` decimal places, are all equal.
        """
        return np.allclose(self.data, other.data)
