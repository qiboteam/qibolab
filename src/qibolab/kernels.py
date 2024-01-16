from pathlib import Path
import json

import numpy as np
from qibolab.qubits import QubitId

class Kernels(dict[QubitId, np.ndarray]):

    """
    A dictionary subclass for handling Qubit Kernels.

    This class extends the built-in dict class and maps QubitId to numpy arrays.
    It provides methods to load and dump the kernels from and to a file.
    """
    
    @classmethod
    def load(cls, path: Path):
        """Class method to load kernels from a file. The file should contain a serialized dictionary
        where keys are serialized QubitId and values are numpy arrays."""
        raw_dict = dict(np.load(path)) 
        return cls({json.loads(key): value for key, value in raw_dict.items()})
    
    def dump(self, path: Path):
        """Instance method to dump the kernels to a file. The keys (QubitId) are serialized to strings
        and the values (numpy arrays) are kept as is."""
        serialzed_dict = {json.dumps(key): value for key, value in self.items()}
        np.savez(path, **serialzed_dict)
