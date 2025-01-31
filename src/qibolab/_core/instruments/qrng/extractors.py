import hashlib
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt
from scipy.linalg import toeplitz

from ...serialize import Model

__all__ = ["Extractor", "ToeplitzExtractor", "ShaExtractor"]


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


def toeplitz_extract(
    m1: npt.NDArray, input_bits: int, extraction_ratio: int
) -> npt.NDArray:
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


class ToeplitzExtractor(Extractor):
    """https://arxiv.org/pdf/2402.09481 appendix A.5"""

    input_bits: int = 12
    """Number of bits of the raw numbers sampled from the QRNG."""
    extraction_ratio: int = 4
    """Number of bits of the uniformly distributed extracted output samples."""
    precision_bits: int = 32

    def __post_init__(self):
        if self.precision_bits % self.extraction_ratio != 0:
            raise ValueError(
                f"Number of bits must be a multiple of the extracted bits {self.extraction_ratio}."
            )

    def num_raw_samples(self, n: int) -> int:
        return 2 * n * (self.precision_bits // self.extraction_ratio)

    def extract(self, raw: List[int]) -> npt.NDArray:
        extracted = toeplitz_extract(
            np.array(raw), self.input_bits, self.extraction_ratio
        ).astype(int)
        upscaled = upscale(extracted, self.extraction_ratio, self.precision_bits)
        return upscaled.astype(float) / (2**self.precision_bits - 1)


class ShaExtractor:
    """Extractor based on the SHA-256 hash algorithm."""

    def num_raw_samples(self, n: int) -> int:
        return 22 * (n // 4 + 1)

    def extract(self, raw: List[int]) -> npt.NDArray:
        extracted = []
        for i in range(len(raw) // 22):
            stream = "".join(
                format(sample, "012b") for sample in raw[22 * i : 22 * (i + 1)]
            )
            hash = hashlib.sha256(stream.encode("utf-8")).hexdigest()
            sha_bin = bin(int(hash, 16))[2:].zfill(256)
            for j in range(4):
                # Convert 53-bit chunk to integer
                uniform_int = int(sha_bin[53 * j : 53 * (j + 1)], 2)
                extracted.append(uniform_int / (2**53 - 1))
        return np.array(extracted)
