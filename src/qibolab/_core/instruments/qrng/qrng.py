from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from serial import Serial

from ..abstract import Instrument
from .extractors import Extractor, ShaExtractor

__all__ = ["QRNG"]


def read(port: Serial, n: int = 1, nbytes: int = 4) -> List[int]:
    """Read raw samples from the QRNG device serial output.

    In the entropy mode of the device, these typically follow a
    normal distribution.

    Args:
        n: Number of samples to retrieve.
        nbytes: Number of bytes to read from serial port to generate one raw sample.
    """
    samples = []
    while len(samples) < n:
        num_str = ""
        while len(num_str) < nbytes:
            sample = port.read(1)
            if sample == b" ":
                break
            num_str += sample.decode("utf-8")
        try:
            samples.append(int(num_str))
        except ValueError:
            pass
    return samples


class QRNG(Instrument):
    """Driver to sample numbers from a Quantum Random Number Generator (QRNG)."""

    address: str
    baudrate: int = 115200
    extractor: Extractor = ShaExtractor()
    port: Optional[Serial] = None

    bytes_per_number: int = 4
    """Number of bytes to read from serial port to generate one raw sample."""

    def connect(self):
        if self.port is None:
            self.port = Serial(self.address, self.baudrate)

    def disconnect(self):
        if self.port is not None:
            self.port.close()
            self.port = None

    def read(self, n: int) -> List[int]:
        return read(self.port, n, self.bytes_per_number)

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
