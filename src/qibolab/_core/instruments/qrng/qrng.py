from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from serial import Serial

from ..abstract import Instrument
from .extractors import Extractor, ShaExtractor

__all__ = ["QRNG"]


class QRNG(Instrument):
    """Driver to sample numbers from a Quantum Random Number Generator (QRNG)."""

    address: str
    baudrate: int = 115200
    extractor: Extractor = ShaExtractor()

    port: Optional[Serial] = None

    samples_per_number: int = 4
    """Number of bytes to read from serial port to generate a raw sample."""

    def connect(self):
        if self.port is None:
            self.port = Serial(self.address, self.baudrate)

    def disconnect(self):
        if self.port is not None:
            self.port.close()
            self.port = None

    def _read(self) -> Optional[int]:
        num_str = ""
        while len(num_str) < self.samples_per_number:
            sample = self.port.read(1)
            if sample == b" ":
                break
            num_str += sample.decode("utf-8")
        try:
            return int(num_str)
        except ValueError:
            return None

    def read(self, n: int = 1) -> List[int]:
        """Read raw samples from the device serial output.

        In the entropy mode of the device, these typically follow a
        Gaussian distribution.

        Args:
            n: Number of samples to retrieve.
        """
        samples = []
        while len(samples) < n:
            number = self._read()
            if number is not None:
                samples.append(number)
        return samples

    def random(self, size: Optional[Union[int, Sequence[int]]] = None) -> npt.NDArray:
        """Returns random floats following uniform distribution in [0, 1].

        Args:
            size: Shape of the returned array (to behave similarly to ``np.random.random``).
        """
        n = np.prod(size)
        nraw = self.extractor.num_raw_samples(n)
        raw = self.read(nraw)
        extracted = self.extractor.extract(raw)[:n]
        return np.reshape(extracted, size)
