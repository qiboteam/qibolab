from abc import ABC, abstractmethod
from typing import List, Tuple

from qibo.models import Circuit


class Transpiler(ABC):
    """A transpiler is a transformation from a circuit to another circuit."""

    @abstractmethod
    def is_satisfied(self, circuit: Circuit) -> bool:
        """Checks if the circuit satisfies this transpiler's requirements.

        In some cases this is computationally easier to check than applying
        the transpilation transformations.
        """

    @abstractmethod
    def transpile(self, circuit: Circuit) -> Tuple[Circuit, List[int]]:
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (qibo.models.Circuit): Circuit to transpile.

        Retruns:
            circuit (qibo.models.Circuit): Circuit after transpilation.
            qubit_map (list): Order of qubits in the transpiled circuit.
        """
        # TODO: Maybe use __call__?
