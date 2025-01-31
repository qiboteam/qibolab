from abc import ABC, abstractmethod
from typing import List

import numpy.typing as npt

from ....serialize import Model

__all__ = ["Extractor"]


class Extractor(Model, ABC):
    @abstractmethod
    def num_raw_samples(self, n: int) -> int:
        """Number of raw QRNG samples that are needed to reach the required random floats.

        Args:
            n (int): Number of required random floats.
        """

    @abstractmethod
    def extract(self, raw: List[int]) -> npt.NDArray:
        """Extract uniformly distributed integers from the device samples."""
