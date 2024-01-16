from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qibolab.qubits import QubitId


@dataclass
class Kernels:
    data: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: Path):
        return cls(data=dict(np.load(path)))

    def dump(self, path: Path):
        # This saves the QubitID as a str
        np.savez(path, **self.data)

    def __getitem__(self, qubit: QubitId):
        return self.data[qubit]
