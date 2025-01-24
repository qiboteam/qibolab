from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg import toeplitz
from serial import Serial

from .abstract import Instrument

__all__ = ["QRNG"]


def unpackbits(x: npt.NDArray, num_bits: int) -> npt.NDArray:
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def generate_toeplitz(input_bits: int, extraction_ratio: int) -> npt.NDArray:
    """Generate a pseudo-random Toeplitz matrix of dimension ``(input_bits, extraction_ratio)``."""
    while True:
        c = np.mod(np.random.permutation(input_bits), 2)
        r = np.mod(np.random.permutation(extraction_ratio), 2)
        if np.sum(c) == 6 and np.sum(r) == (extraction_ratio // 2):
            return toeplitz(c, r)


def extractor(m1: npt.NDArray, input_bits: int, extraction_ratio: int) -> npt.NDArray:
    """Extract uniformly distributed integers from the device samples."""
    m2 = unpackbits(m1, input_bits)
    m2 = np.flip(m2)

    m3 = generate_toeplitz(input_bits, extraction_ratio)

    m4 = np.matmul(m2, m3)
    m4 = m4.astype("int16")
    m4 = np.mod(m4, 2)
    m4 = np.packbits(m4)
    m4 = np.right_shift(m4, 8 - extraction_ratio)
    return m4


def upscale(samples: npt.NDArray, input_bits=4, output_bits=32) -> npt.NDArray:
    """Increase size of random bit strings by concatenating them."""
    assert output_bits > input_bits
    assert output_bits % input_bits == 0

    factor = output_bits // input_bits
    assert len(samples) % factor == 0

    n = len(samples) // factor
    upscaled = np.zeros(n, dtype=int)
    for i, s in enumerate(np.reshape(samples, (factor, n))):
        upscaled += s << (input_bits * i)
    return upscaled


class QRNG(Instrument):
    """Driver to sample numbers from a Quantum Random Number Generator (QRNG).

    Note that if we are not connected to a physical QRNG device, this will
    return pseudo-random generated numbers using ``numpy.random``.
    """

    address: str
    baudrate: int = 115200

    port: Optional[Serial] = None

    samples_per_number: int = 4
    """Number of bytes to read from the device to generate a number."""
    raw_dimension: int = 12
    """Number of bits of the raw sampled numbers following normal distribution."""
    extracted_bits: int = 4
    """Number of bits of the uniformly distributed extracted samples."""

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

    def read(self, n: int = 1) -> npt.NDArray:
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
        return np.array(samples)

    def _extractor(self, samples: npt.NDArray) -> npt.NDArray:
        return extractor(samples, self.raw_dimension, self.extracted_bits)

    def extract(self, n: int) -> npt.NDArray:
        """Returns random ``extracted_bits``-bit integers following uniform distribution.

        Args:
            n: Number of samples to retrieve.
        """
        if self.port is None:
            return np.random.randint(0, 2**self.extracted_bits, size=(n,))

        samples = self.read(2 * n)
        extracted = self._extractor(samples)
        return extracted

    def random(
        self, size: Optional[Union[int, Iterable[int]]] = None, precision_bits: int = 32
    ) -> npt.NDArray:
        """Returns random floats following uniform distribution in [0, 1].

        Args:
            size: Shape of the returned array (to behave similarly to ``np.random.random``).
            precision_bits: Number of bits that is sampled to control precision.
        """
        if precision_bits % self.extracted_bits != 0:
            raise ValueError(
                f"Number of bits must be a multiple of the extracted bits {self.extracted_bits}."
            )

        n = np.prod(size) * (precision_bits // self.extracted_bits)
        samples = self.extract(n).astype(int)
        upscaled = upscale(samples, self.extracted_bits, precision_bits)
        upscaled = np.reshape(upscaled / (2**precision_bits - 1), size)
        return upscaled.astype(f"float{precision_bits}")
