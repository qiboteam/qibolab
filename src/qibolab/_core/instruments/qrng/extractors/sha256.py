import hashlib
from typing import List

import numpy as np
import numpy.typing as npt

from .abstract import Extractor

__all__ = ["ShaExtractor"]


class ShaExtractor(Extractor):
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
