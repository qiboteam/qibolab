import json
from pathlib import Path

import numpy as np

from qibolab.qubits import QubitId


class Kernels(dict[QubitId, np.ndarray]):
    """A dictionary subclass for handling Qubit Kernels.

    This class extends the built-in dict class and maps QubitId to numpy
    arrays. It provides methods to load and dump the kernels from and to
    a file.
    """

    @classmethod
    def load(cls, path: Path):
        """Class method to load kernels from a file.

        The file should contain a serialized dictionary where keys are
        serialized QubitId and values are numpy arrays.
        """
        return cls({json.loads(key): value for key, value in np.load(path).items()})

    def dump(self, path: Path):
        """Instance method to dump the kernels to a file.

        The keys (QubitId) are serialized to strings and the values
        (numpy arrays) are kept as is.
        """
        np.savez(
            path, **{json.dumps(qubit_id): value for qubit_id, value in self.items()}
        )
