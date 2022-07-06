from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from qibo.config import raise_error
from qibolab.constants import RUNCARD
from qibolab.typings import PulseShapeName
from qibolab.typings.factory_element import FactoryElement

@dataclass
class PulseShape(FactoryElement):
    """Abstract class for pulse shapes"""

    name: PulseShapeName = field(init=False, repr=False)

    def envelope(self, duration: int, amplitude: float, resolution: float = 1.0) -> np.ndarray:
        """Compute the amplitudes of the pulse shape envelope.

        Args:
            duration (int): Duration of the pulse (ns).
            amplitude (float): Maximum amplitude of the pulse.

        Returns:
            ndarray: Amplitude of the envelope for each time step.
        """
        raise NotImplementedError

    def to_dict(self):
        """Return dictionary representation of the pulse shape.

        Returns:
            dict: Dictionary.
        """
        return {RUNCARD.NAME: self.name.value} | self.__dict__